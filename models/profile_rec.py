import pandas as pd
import numpy as np
import torch as th
import warnings
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from dataset import ReviewDataset
from sklearn.neighbors import BallTree

warnings.filterwarnings('ignore')

class ProfileRecModel(object):
    def __init__(
        self, 
        train_dataset: ReviewDataset,
        test_dataset: ReviewDataset,
        device: th.device = th.device('cpu'),
        seed: int=123,
        batch_size: int=32,
        top_n: int=10
    ) -> None:
        self.users_train = train_dataset.users_df
        self.reviews_train = train_dataset.reviews_df
        self.users_test = test_dataset.users_df
        self.reviews_test = test_dataset.reviews_df
        self.top_n = top_n
        self.device = device
        self.seed = seed
        self.batch_size = batch_size
        self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2').to(self.device)
        
        # Dict to store user embeddings based on profile attributes
        self.user_embedding_dict = {}
        
        th.manual_seed(self.seed)
        np.random.seed(self.seed)
        
        print(f"Training users: {self.users_train.shape[0]}, Test users: {self.users_test.shape[0]}")
    
    def preprocess(self, reviews):
        for col in ['review_title', 'review_positive', 'review_negative']:
            reviews[col] = reviews[col].fillna('')
    
        reviews['combined_review'] = reviews.apply(
            lambda row: ' '.join([row['review_title'], row['review_positive'], row['review_negative']]), axis=1
        )
        
        return reviews.drop(columns=['review_title', 'review_positive', 'review_negative'])
    
    def generate_review_embeddings(self, reviews):
        reviews = self.preprocess(reviews)
        review_texts = reviews['combined_review'].tolist()
        embeddings = self.model.encode(review_texts, batch_size=self.batch_size, show_progress_bar=True, device=self.device)
        reviews['review_embedding'] = list(embeddings)
        print("Finished computing review embeddings...")
        return reviews
    
    def generate_user_embeddings(self):
        profile_columns = [
            'guest_type',
            'guest_country',
            'accommodation_type',
            'accommodation_country',
            'month',
            'room_nights'
        ]
        
        # Create user profiles by concatenating the selected columns
        user_profiles = self.users_train[profile_columns].apply(
            lambda row: tuple(row.astype(str)), # Convert profile columns to tuple for dict key
            axis=1
        ).tolist()
        
        embeddings = self.model.encode(user_profiles, batch_size=self.batch_size, show_progress_bar=True, device=self.device)
        
        # Store embeddings
        for profile, embedding in zip(user_profiles, embeddings):
            self.user_embedding_dict[profile] = embedding
            
        self.users_train['user_embedding'] = list(embeddings)
        print("Finished computing user embeddings...")
    
    def build_user_embedding_tree(self):
        # Stack all user embeddings into a matrix
        user_embeddings = np.vstack([np.array(embedding) for embedding in self.user_embedding_dict.values()])
        # Build a BallTree for efficient nearest neighbor search
        self.user_embedding_tree = BallTree(user_embeddings, metric='euclidean')
        self.user_embedding_profiles = list(self.user_embedding_dict.keys())
        print("Finished building user embedding tree...")

    def find_similar_user_embedding(self, user_profile):
        # Check for exact profile match first
        if user_profile in self.user_embedding_dict:
            return self.user_embedding_dict[user_profile]
        
        # Use BallTree for nearest neighbor search
        user_embedding_matrix = self.model.encode([user_profile], batch_size=1, device=self.device, show_progress_bar=False)
        dist, ind = self.user_embedding_tree.query(user_embedding_matrix, k=1)
        
        # Get the closest matching profile
        best_match_profile = self.user_embedding_profiles[ind[0][0]]
        return self.user_embedding_dict[best_match_profile]
    
    def rank_pair_reviews(self, user_embedding, accommodation_id):
        # Retrieve reviews for a specific accommodation
        reviews_for_accommodation = self.reviews_test[self.reviews_test['accommodation_id'] == accommodation_id].copy()
        
        if reviews_for_accommodation.empty:
            return []

        # Extract stored embeddings for the accommodation's reviews
        review_embeddings = np.vstack(reviews_for_accommodation['review_embedding'].values)

        # Calculate cosine similarity between the user embedding and each review embedding
        similarities = cosine_similarity([user_embedding], review_embeddings).flatten()

        # Add similarity scores to the reviews dataframe using .loc
        reviews_for_accommodation.loc[:, 'similarity'] = similarities

        # Sort reviews by similarity (descending) and get top N
        ranked_reviews = reviews_for_accommodation.sort_values(by='similarity', ascending=False)
        ranked_review_ids = ranked_reviews.head(self.top_n)['review_id'].tolist()
        
        return ranked_review_ids
    
    def recommend(self):
        self.reviews_test = self.generate_review_embeddings(self.reviews_test)
        self.generate_user_embeddings()
        self.build_user_embedding_tree()
        results = []
        
        self.users_test = self.users_test.reset_index(drop=True)
        total_pairs = self.users_test.shape[0]
        print(f"Starting to generate ranked reviews for {total_pairs} user-accommodation pairs.")
        
        # Iterate over each user-accommodation pair
        for index, row in self.users_test.iterrows():
            user_id = row['user_id']
            accommodation_id = row['accommodation_id']
            
            # Log progress every 10000 pairs
            if index % 10000 == 0:
                print(f"Processing pair {index + 1}/{total_pairs}")
            
            # Create a profile tuple for current user
            user_profile = tuple(row[['guest_type', 'guest_country', 'accommodation_type', 'accommodation_country', 'month', 'room_nights']].astype(str))
            user_embedding = self.find_similar_user_embedding(user_profile)
            
            if user_embedding is None:
                continue
            
            # Rank reviews for this user-accommodation pair
            ranked_review_ids = self.rank_pair_reviews(user_embedding, accommodation_id)
            
            # Append the result to the list
            results.append({
                'user_id': user_id,
                'accommodation_id': accommodation_id,
                'ranked_review_ids': ranked_review_ids
            })
        
        print("Finished generating ranked reviews.")
        
        # Convert the results to a DataFrame
        results_df = pd.DataFrame(results)
        return results_df
    
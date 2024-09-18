import pandas as pd
import numpy as np
import torch as th
import warnings
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings('ignore')

class EmbeddingModel(object):
    def __init__(
        self, 
        users_df: pd.DataFrame,
        reviews_df: pd.DataFrame,
        device: th.device = th.device('cpu'),
        seed: int=123,
        batch_size: int=32,
        top_n: int=10
    ) -> None:
        self.users = users_df
        self.reviews = reviews_df
        self.top_n = top_n
        self.device = device
        self.seed = seed
        self.batch_size = batch_size
        self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2').to(self.device)
        
        th.manual_seed(self.seed)
        np.random.seed(self.seed)
    
    def preprocess(self):
        for col in ['review_title', 'review_positive', 'review_negative']:
            self.reviews[col] = self.reviews[col].fillna('')
    
        self.reviews['combined_review'] = self.reviews.apply(
            lambda row: ' '.join([row['review_title'], row['review_positive'], row['review_negative']]), axis=1
        )
        self.reviews = self.reviews.drop(columns=['review_title', 'review_positive', 'review_negative'])
    
    def generate_review_embeddings(self):
        self.preprocess()
        review_texts = self.reviews['combined_review'].tolist()
        embeddings = self.model.encode(review_texts, batch_size=self.batch_size, show_progress_bar=True, device=self.device)
        self.reviews['review_embedding'] = list(embeddings)
        print("Finished computing review embeddings...")
    
    def generate_user_embeddings(self):
        user_profiles = self.users.apply(
            lambda row: f"{row['guest_type']} {row['guest_country']} {row['accommodation_type']} {row['accommodation_country']}", 
            axis=1
        ).tolist()
        embeddings = self.model.encode(user_profiles, batch_size=self.batch_size, show_progress_bar=True, device=self.device)
        self.users['user_embedding'] = list(embeddings)
        print("Finished computing user embeddings...")
    
    def rank_pair_reviews(self, user_embedding, accommodation_id):
        # Retrieve reviews for a specific accommodation
        reviews_for_accommodation = self.reviews[self.reviews['accommodation_id'] == accommodation_id].copy()
        
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
        self.generate_review_embeddings()
        self.generate_user_embeddings()
        results = []
        
        self.users = self.users.reset_index(drop=True)
        total_pairs = self.users.shape[0]
        print(f"Starting to generate ranked reviews for {total_pairs} user-accommodation pairs.")
        
        # Iterate over each user-accommodation pair
        for index, row in self.users.iterrows():
            user_id = row['user_id']
            accommodation_id = row['accommodation_id']
            
            # Log progress every 10000 pairs
            if index % 10000 == 0:
                print(f"Processing pair {index + 1}/{total_pairs}")
            
            user_embedding = row['user_embedding']  # Get precomputed user embedding
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
    
import pandas as pd

class PopularityModel:
    def __init__(self, users_df, reviews_df, top_n=10):
        self.users = users_df
        self.reviews = reviews_df
        self.top_n = top_n
    
    def rank_pair_reviews(self, user_id, accommodation_id):
        reviews_for_accommodation = self.reviews[self.reviews['accommodation_id'] == accommodation_id]
        # Sort reviews by review scores in descending order
        ranked_reviews = reviews_for_accommodation.sort_values(by=['review_score'], ascending=[False])
        # Limit to the top_n reviews
        top_ranked_reviews = ranked_reviews.head(self.top_n)
        # Generate a list of review_ids in ranked order
        ranked_review_ids = top_ranked_reviews['review_id'].tolist()
        return ranked_review_ids
    
    def generate_ranked_reviews(self):
        results = []
        
        total_pairs = self.users.shape[0]
        print(f"Starting to generate ranked reviews for {total_pairs} user-accommodation pairs.")
        
        # Iterate over each unique user_id and accommodation_id pair in matches_df
        for index, row in self.users.iterrows():
            user_id = row['user_id']
            accommodation_id = row['accommodation_id']
            
            # Log progress every 100 pairs
            if index % 10000 == 0:
                print(f"Processing pair {index + 1}/{total_pairs}")
            
            # Get ranked reviews for this pair
            ranked_review_ids = self.rank_pair_reviews(user_id, accommodation_id)
            
            # Append results to the list
            results.append({
                'user_id': user_id,
                'accommodation_id': accommodation_id,
                'ranked_review_ids': ranked_review_ids
            })
        
        print("Finished generating ranked reviews.")
        
        # Convert results to a DataFrame
        results_df = pd.DataFrame(results)
        return results_df
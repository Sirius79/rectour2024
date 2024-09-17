import pandas as pd

def calculate_mrr_at_k(ranked_reviews_df, matches_df, k=10):
    mrr_sum = 0
    num_pairs = 0
    
    print(f"Starting MRR@{k} calculation")
    
    # Merge ranked reviews with matches to get ideal review for each pair
    merged_df = pd.merge(ranked_reviews_df, matches_df, on=['user_id', 'accommodation_id'], how='left')
    
    # Iterate over each pair to calculate MRR@10
    for index, row in merged_df.iterrows():
        ideal_review_id = row['review_id']
        ranked_review_ids = row['ranked_review_ids']
        
        if index % 10000 == 0:
            print(f"Processing pair{index + 1}/{len(merged_df)}")
        
        # Find the position of the ideal review
        if ideal_review_id in ranked_review_ids:
            rank_of_ideal = ranked_review_ids.index(ideal_review_id) + 1  # Convert to 1-based index
            if rank_of_ideal <= k:
                mrr_sum += 1.0 / rank_of_ideal
        num_pairs += 1
    
    # Calculate MRR@10
    mrr_at_k = mrr_sum / num_pairs
    return mrr_at_k

def format_ranked_reviews(ranked_reviews_df, k=10):
    # Create a DataFrame where each row contains a list of review IDs
    ranked_reviews_expanded = ranked_reviews_df['ranked_review_ids'].apply(pd.Series)
    
    # Rename columns to Review 1, Review 2, ..., Review k
    ranked_reviews_expanded.columns = [f'review_{i+1}' for i in range(ranked_reviews_expanded.shape[1])]
    
    # Concatenate with the original DataFrame
    formatted_df = pd.concat([ranked_reviews_df[['user_id', 'accommodation_id']], ranked_reviews_expanded], axis=1)
    # Rename user_id and accommodation_id columns
    # formatted_df.rename(columns={'user_id': 'User id', 'accommodation_id': 'Accommodation id'}, inplace=True)
    
    return formatted_df
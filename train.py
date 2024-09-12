import pandas as pd
import logging
from models import PopularityModel
from dataset import ReviewDataset
from utils import calculate_mrr_at_k, format_ranked_reviews

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == "__main__":
    val_dataset = ReviewDataset(mode='val')
    popularity_model = PopularityModel(val_dataset.users_df, val_dataset.reviews_df)
    ranked_reviews_df = popularity_model.generate_ranked_reviews()

    mrr_10 = calculate_mrr_at_k(ranked_reviews_df, val_dataset.matches_df, k=10)
    print(f"MRR@10: {mrr_10}")

    formatted_ranked_reviews_df = format_ranked_reviews(ranked_reviews_df, k=10)
    formatted_ranked_reviews_df.to_csv('val_predictions.csv', index=False)
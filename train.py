import logging
import argparse
import torch as th
from models import PopularityModel, EmbeddingModel, ProfileRecModel
from dataset import ReviewDataset
from utils import calculate_mrr_at_k, format_ranked_reviews

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algo",
        default="pop",
        choices=["pop", "emb", "profile"],
        help="Use pop for popularity based model"
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu","mps","gpu"],
        help="Use gpu for NVIDIA gpus"
    )
    args = parser.parse_args()
    mode = 'test'
    
    # Set device
    if args.device == "mps" and th.has_mps:
        device = th.device("mps")
    elif args.device == "gpu" and th.cuda.is_available():
        device = th.device("cuda")
    else:
        device = th.device("cpu")
    
    print(f"Using device: {device}")
    
    # Load the training dataset using the ReviewDataset class
    train_dataset = ReviewDataset(sample=False, mode='train')
    val_dataset = ReviewDataset(sample=False, mode='test')
    
    # Create the recommendation system based on selected algo
    if args.algo == "pop":
        recommender = PopularityModel(train_dataset.users_df, train_dataset.reviews_df)
    elif args.algo == "emb":
        recommender = EmbeddingModel(train_dataset.users_df, train_dataset.reviews_df, device)
    elif args.algo == "profile":
        recommender = ProfileRecModel(train_dataset, val_dataset, device)
    
    ranked_reviews_df = recommender.recommend()
    
    if mode != 'test':
        mrr_10 = calculate_mrr_at_k(ranked_reviews_df, val_dataset.matches_df)
        print(f"MRR@10: {mrr_10}")

    formatted_ranked_reviews_df = format_ranked_reviews(ranked_reviews_df, k=10)
    formatted_ranked_reviews_df.to_csv('test2_predictions.csv', index=False)
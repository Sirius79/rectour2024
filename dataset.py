import pandas as pd
import numpy as np

class ReviewDataset(object):
    def __init__(
        self,
        mode,
        seed=123,
        sample=False,
        sample_size=0.1
    ) -> None:
        self.reviews_df = pd.read_csv(f"data/{mode}_reviews.csv")
        self.users_df = pd.read_csv(f"data/{mode}_users.csv")
        
        if mode in ['train', 'val']:
            self.matches_df = pd.read_csv(f"data/{mode}_matches.csv")
        
        if sample:
            np.random.seed(seed)
            sampled_accoIds = np.random.choice(self.users_df['accommodation_id'].unique(),
                                               size=int(len(self.users_df['accommodation_id'].unique()) * sample_size),
                                               replace=False)
            self.reviews_df = self.reviews_df[self.reviews_df['accommodation_id'].isin(sampled_accoIds)]
            self.users_df = self.users_df[self.users_df['accommodation_id'].isin(sampled_accoIds)]
            
            if mode in ['train', 'val']:
                self.matches_df = self.matches_df[self.matches_df['accommodation_id'].isin(sampled_accoIds)]

if __name__ == "__main__":
    review_dataset = ReviewDataset(mode='val', sample=True)
    print(review_dataset.matches_df.head())
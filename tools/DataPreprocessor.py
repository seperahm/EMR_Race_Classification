from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

RANDOM_STATE = 42

RACE_SAMPLE_SIZE = 100 # sample size for each race
RACE_ABSENT_MULTIPLIER = 9 # sample size multiplier for the 'absent' label
TEST_SIZE = 0.2

class DataPreprocessor:
    # Function to upsample data with balanced repetition
    def balanced_upsample(self, df, label, n_samples):
        df_label = df[df['label'] == label]
        count = len(df_label)
    
        if count >= n_samples:
            return df_label.sample(n_samples, random_state=42)
        else:
            multiplier = n_samples // count
            remainder = n_samples % count
    
            df_repeated = pd.concat([df_label] * multiplier, ignore_index=True)
            df_remainder = df_label.sample(remainder, random_state=42)
    
            return pd.concat([df_repeated, df_remainder], ignore_index=True)
    def sample_data(self, df, sample_size=RACE_SAMPLE_SIZE, absent_multiplier=RACE_ABSENT_MULTIPLIER):
        # Sample 32 entries from each label, upsampling if necessary
        labels = df['label'].unique()
        sampled_dfs = []
        
        for label in labels:
            if label == 'absent':
                sampled_dfs.append(self.balanced_upsample(df, label, absent_multiplier * RACE_SAMPLE_SIZE))
            else:
                sampled_dfs.append(self.balanced_upsample(df, label, sample_size))
        
        # Combine all sampled data
        sampled_df = pd.concat(sampled_dfs, ignore_index=True)
        
        return sampled_df

    def split_data(self, df):
        # Split the data into train and test sets with stratification
        train_df, test_df = train_test_split(df, test_size=TEST_SIZE, stratify=df['label'], random_state=42)
        
        # Show the result
        print('==================TRAIN DATA====================')
        print(train_df.groupby(['label']).count())
        
        print('==================TEST DATA====================')
        print(test_df.groupby(['label']).count())

        return train_df, test_df

    def sample_and_split_data(self, df, sample_size=RACE_SAMPLE_SIZE, absent_multiplier=RACE_ABSENT_MULTIPLIER):
        sampled_df = self.sample_data(df)
        return self.split_data(sampled_df)

    def prepare_data(self, train_df, test_df):
        # Prepare your data
        X_train = train_df['text'].values
        X_test = test_df['text'].values
        y_train = train_df['label'].values
        y_test = test_df['label'].values
        return X_train, X_test, y_train, y_test

        
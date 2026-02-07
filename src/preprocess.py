import pandas as pd
import numpy as np
import os

TRAIN_PATH = os.path.join('data', 'train_Kaggle_Comp.csv')
TEST_PATH = os.path.join('data', 'test_Kaggle_Comp.csv')

def load_data():
    if not os.path.exists(TRAIN_PATH):
        return None, None
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    return train_df, test_df

def clean_data(df, is_train=True):
    if 'employee_id' in df.columns:
        df = df.drop('employee_id', axis=1)
    
    if is_train and 'burnout_score' in df.columns:
        df = df.dropna(subset=['burnout_score'])
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
            
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if len(df[col].mode()) > 0:
            df[col] = df[col].fillna(df[col].mode()[0])
        
    return df

def engineer_features(df):
    if 'Date of Joining' in df.columns:
        df = df.drop('Date of Joining', axis=1)
        
    categorical_cols = df.select_dtypes(include=['object']).columns
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    return df

def main():
    train_df, test_df = load_data()
    if train_df is None:
        return

    train_df = clean_data(train_df, is_train=True)
    test_df = clean_data(test_df, is_train=False)
    
    train_df = engineer_features(train_df)
    test_df = engineer_features(test_df)
    
    train_cols = [c for c in train_df.columns if c != 'burnout_score']
    
    for col in train_cols:
        if col not in test_df.columns:
            test_df[col] = 0
            
    test_df = test_df[train_cols] 
    
    train_df.to_csv(os.path.join('data', 'processed_train.csv'), index=False)
    test_df.to_csv(os.path.join('data', 'processed_test.csv'), index=False)

if __name__ == "__main__":
    main()

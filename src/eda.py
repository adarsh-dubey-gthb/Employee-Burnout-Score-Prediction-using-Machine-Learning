import pandas as pd
import numpy as np
import os

DATA_DIR = 'data'
TRAIN_PATH = os.path.join(DATA_DIR, 'train_Kaggle_Comp.csv')
TEST_PATH = os.path.join(DATA_DIR, 'test_Kaggle_Comp.csv')

def load_data():
    if not os.path.exists(TRAIN_PATH):
        print(f"Error: {TRAIN_PATH} not found.")
        return None, None
    
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    return train_df, test_df

def inspect_data(df, name="Data"):
    print(f"\n--- Inspecting {name} ---")
    print(f"Shape: {df.shape}")
    print("\nMissing Values:")
    print(df.isnull().sum()[df.isnull().sum() > 0])
    
    print("\nData Types:")
    print(df.dtypes)
    
    print("\nSummary Statistics:")
    print(df.describe())

def analyze_target(df):
    print("\n--- Target Variable Analysis (burnout_score) ---")
    print(df['burnout_score'].describe())

def correlation_analysis(df):
    print("\n--- Correlation with Burnout Score ---")
    numeric_df = df.select_dtypes(include=[np.number])
    if 'burnout_score' in numeric_df.columns:
        corr = numeric_df.corr()['burnout_score'].sort_values(ascending=False)
        print(corr)

def main():
    train_df, test_df = load_data()
    if train_df is None:
        return

    inspect_data(train_df, "Train Set")
    
    analyze_target(train_df)
    correlation_analysis(train_df)

if __name__ == "__main__":
    main()

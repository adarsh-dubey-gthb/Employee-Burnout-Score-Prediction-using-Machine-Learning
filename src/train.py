import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

DATA_DIR = 'data'
TRAIN_PATH = os.path.join(DATA_DIR, 'processed_train.csv')
TEST_PATH = os.path.join(DATA_DIR, 'processed_test.csv')
SAMPLE_SUBMISSION_PATH = os.path.join(DATA_DIR, 'sample_submission_Kaggle_Comp.csv')
SUBMISSION_PATH = os.path.join(DATA_DIR, 'submission.csv')

def load_processed_data():
    if not os.path.exists(TRAIN_PATH) or not os.path.exists(TEST_PATH):
        print("Error: Processed data not found. Run preprocess.py first.")
        return None, None, None, None, None

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    
    X = train_df.drop('burnout_score', axis=1)
    y = train_df['burnout_score']
    X_test = test_df
    
    return X, y, X_test, train_df, test_df

def train_models(X_train, y_train, X_val, y_val):
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    }
    
    results = {}
    trained_models = {}
    
    print("\n--- Model Evaluation (RMSE) ---")
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        print(f"{name}: {rmse:.4f}")
        results[name] = rmse
        trained_models[name] = model
        
    return results, trained_models

def make_submission(model, X_test, original_test_path):
    print("\n--- Generating Submission ---")
    predictions = model.predict(X_test)
    
    predictions = np.clip(predictions, 0, 1)
    
    if os.path.exists(SAMPLE_SUBMISSION_PATH):
        submission = pd.read_csv(SAMPLE_SUBMISSION_PATH)
        submission['burnout_score'] = predictions
        submission.to_csv(SUBMISSION_PATH, index=False)
        print(f"Submission saved to {SUBMISSION_PATH}")
    else:
        print(f"Warning: {SAMPLE_SUBMISSION_PATH} not found. Submission file creation might fail if IDs are needed from it.")
        raw_test_path = os.path.join(DATA_DIR, 'test_Kaggle_Comp.csv')
        if os.path.exists(raw_test_path):
             raw_test = pd.read_csv(raw_test_path)
             submission = pd.DataFrame({
                 'employee_id': raw_test['employee_id'],
                 'burnout_score': predictions
             })
             submission.to_csv(SUBMISSION_PATH, index=False)
             print(f"Submission saved to {SUBMISSION_PATH} (using IDs from raw test file)")

def main():
    X, y, X_test, _, _ = load_processed_data()
    if X is None:
        return

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    results, models = train_models(X_train, y_train, X_val, y_val)
    
    best_model_name = min(results, key=results.get)
    best_model = models[best_model_name]
    print(f"\nBest Model: {best_model_name}")
    
    print("Retraining best model on full dataset...")
    best_model.fit(X, y)
    
    make_submission(best_model, X_test, SAMPLE_SUBMISSION_PATH)

if __name__ == "__main__":
    main()

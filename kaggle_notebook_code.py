import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

TRAIN_PATH = 'data/train_Kaggle_Comp.csv'
TEST_PATH = 'data/test_Kaggle_Comp.csv'
SAMPLE_SUBMISSION_PATH = 'data/sample_submission_Kaggle_Comp.csv'

if not os.path.exists(TRAIN_PATH):
    TRAIN_PATH = 'data/train_Kaggle_Comp.csv' 
    TEST_PATH = 'test_Kaggle_Comp.csv'
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)
print("Data Loaded.")
print(f"Train Shape: {train_df.shape}")
print(f"Test Shape: {test_df.shape}")
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
        
    if 'mental_fatigue_score' in df.columns and 'work_pressure_score' in df.columns:
        df['Interaction_Fatigue_Pressure'] = df['mental_fatigue_score'] * df['work_pressure_score']
    
    if 'mental_fatigue_score' in df.columns and 'resource_allocation' in df.columns:
        df['Interaction_Fatigue_Allocation'] = df['mental_fatigue_score'] * df['resource_allocation']
        
    categorical_cols = df.select_dtypes(include=['object']).columns
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    return df

train_df = clean_data(train_df, is_train=True)
test_df = clean_data(test_df, is_train=False)

train_df = engineer_features(train_df)
test_df = engineer_features(test_df)
X = train_df.drop('burnout_score', axis=1)
y = train_df['burnout_score']
for col in X.columns:
    if col not in test_df.columns:
        test_df[col] = 0

X_test = test_df[X.columns]

print("Preprocessing Complete.")

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.model_selection import GridSearchCV

print("Starting Hyperparameter Tuning for XGBoost...")
xgb = XGBRegressor(random_state=42, n_jobs=-1)

param_grid = {
    'n_estimators': [100, 300, 500],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'subsample': [0.7, 0.8, 1.0]
}

grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, 
                           cv=3, scoring='neg_root_mean_squared_error', verbose=1)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best CV RMSE: {-grid_search.best_score_:.4f}")

y_pred = best_model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print(f"Validation RMSE with Tuned XGBoost: {rmse:.4f}")


print("\nTraining Linear Regression for Ensemble...")
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_val)
lr_rmse = np.sqrt(mean_squared_error(y_val, lr_pred))
print(f"Linear Regression Value RMSE: {lr_rmse:.4f}")


print("\nCreating Ensemble Predictions...")
ensemble_pred_val = (0.5 * lr_pred) + (0.5 * y_pred)
ensemble_rmse = np.sqrt(mean_squared_error(y_val, ensemble_pred_val))
print(f"Ensemble (50% LR + 50% XGB) Validation RMSE: {ensemble_rmse:.4f}")

print("\nRetraining models on full dataset for submission...")
best_model.fit(X, y) 
lr.fit(X, y)         


xgb_test_pred = best_model.predict(X_test)
lr_test_pred = lr.predict(X_test)


final_predictions = (0.5 * lr_test_pred) + (0.5 * xgb_test_pred)
final_predictions = np.clip(final_predictions, 0, 1) 

try:
    submission = pd.read_csv(SAMPLE_SUBMISSION_PATH)
except:
    submission = pd.read_csv(TEST_PATH)[['employee_id']]

submission['burnout_score'] = final_predictions
submission.to_csv('submission.csv', index=False)

print("submission.csv created (Ensemble Version).")
print(submission.head())



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


from sklearn.model_selection import GridSearchCV

    
xgb = XGBRegressor(random_state=42, n_jobs=-1)
param_grid = {
    'n_estimators': [100, 300],        
    'learning_rate': [0.05, 0.1],      
    'max_depth': [3, 5, 7],            
    'subsample': [0.8, 1.0]           
}

print("Starting Hyperparameter Tuning for XGBoost...")
grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, 
                           cv=3, scoring='neg_root_mean_squared_error', verbose=1)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best CV RMSE: {-grid_search.best_score_:.4f}")


y_pred = best_model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print(f"Validation RMSE with Tuned XGBoost: {rmse:.4f}")

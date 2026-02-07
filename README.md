# Employee Burnout Analysis & Prediction

This project aims to predict employee burnout scores using machine learning techniques as part of a Kaggle competition.

## Project Structure
- `data/`: Contains dataset files (train.csv, test.csv, sample_submission.csv)
- `notebooks/`: Jupyter notebooks for EDA and modeling
- `src/`: Source code for reproducible pipelines
- `README.md`: Project documentation

## Getting Started
1.  Install dependencies: `pip install -r requirements.txt`
2.  Place the competition data in the `data/` directory.
3.  Run EDA notebooks in `notebooks/`.
4.  Train models using scripts in `src/` or notebooks.

## Goal
Predict `burnout_score` (target) based on employee attributes.
Evaluation metric: RMSE.

## Model Approach
-   EDA & Preprocessing
-   Feature Engineering
-   Baseline: Linear Regression
-   Advanced: Random Forest, Gradient Boosting

## Requirements
-   Python 3.8+
-   pandas, numpy, scikit-learn, matplotlib, seaborn, xgboost/lightgbm

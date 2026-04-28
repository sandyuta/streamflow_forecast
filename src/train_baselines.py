import pandas as pd
import numpy as np
import os
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit
from sklearn.metrics import mean_squared_error

# Import our custom tools
from utils import create_tabular_lags

def tune_and_train(csv_path, catchment_name):
    print(f"--- Tuning & Training Baselines for {catchment_name.upper()} ---")
    
    # 1. Load Data & Create Lags
    df = pd.read_csv(csv_path, index_col='Date', parse_dates=True)
    X, y = create_tabular_lags(df, target_col='discharge', lag_days=7)
    
    # 2. Implement the 10/3/2 Split Boundaries
    train_end = '2008-12-31'
    val_end = '2011-12-31'
    
    # Filter out the final test set (2012-2013) entirely so the models never see it
    X_working = X.loc[:val_end]
    y_working = y.loc[:val_end]
    
    # Scale features
    scaler = StandardScaler()
    X_working_scaled = scaler.fit_transform(X_working)
    
    os.makedirs('outputs/models', exist_ok=True)
    joblib.dump(scaler, f'outputs/models/{catchment_name}_scaler.joblib')

    # 3. Create the Predefined Split for Validation (10 years Train, 3 years Val)
    # -1 means Train, 0 means Validation
    test_fold = np.where(X_working.index <= train_end, -1, 0)
    ps = PredefinedSplit(test_fold)

    # 4. Define Hyperparameter Search Spaces
    # MLR has no hyperparameters to tune, so we just train it directly
    mlr = LinearRegression()
    mlr.fit(X_working_scaled[test_fold == -1], y_working[test_fold == -1])
    joblib.dump(mlr, f'outputs/models/{catchment_name}_MLR_ARX.joblib')
    print("  Trained and saved MLR_ARX (No tuning required).")

    search_spaces = {
        'RandomForest': {
            'model': RandomForestRegressor(random_state=42, n_jobs=-1),
            'params': {
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10]
            }
        },
        'XGBoost': {
            'model': XGBRegressor(random_state=42),
            'params': {
                'n_estimators': [50, 100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7, 9]
            }
        }
    }

    # 5. Run Randomized Search for RF and XGBoost
    tuning_results = []
    
    for name, config in search_spaces.items():
        print(f"  Tuning {name} (Trying 10 random parameter combinations)...")
        search = RandomizedSearchCV(
            estimator=config['model'],
            param_distributions=config['params'],
            n_iter=10, # Number of random combinations to try
            cv=ps,     # Strictly enforces our 10/3 validation split
            scoring='neg_root_mean_squared_error',
            random_state=42,
            n_jobs=-1
        )
        
        # This automatically searches, finds the best, and refits on the whole working set
        search.fit(X_working_scaled, y_working)
        
        # Save the best model
        best_model = search.best_estimator_
        joblib.dump(best_model, f'outputs/models/{catchment_name}_{name}.joblib')
        
        # Log the results
        best_rmse = -search.best_score_
        print(f"    Best Params: {search.best_params_}")
        print(f"    Validation RMSE: {best_rmse:.2f}")
        
        tuning_results.append({
            'Catchment': catchment_name,
            'Model': name,
            'Best_Params': str(search.best_params_),
            'Validation_RMSE': best_rmse
        })
        
    # 6. Save Tuning Report
    os.makedirs('outputs/tables', exist_ok=True)
    report_df = pd.DataFrame(tuning_results)
    report_path = f'outputs/tables/{catchment_name}_tuning_report.csv'
    report_df.to_csv(report_path, index=False)
    print(f"  Saved tuning report to {report_path}\n")

if __name__ == "__main__":
    processed_dir = "data/processed"
    tune_and_train(os.path.join(processed_dir, "snow_processed.csv"), "snow")
    tune_and_train(os.path.join(processed_dir, "rain_processed.csv"), "rain")
import pandas as pd
import os
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler

# Import our custom tools from utils.py
from utils import create_tabular_lags

def train_and_save_baselines(csv_path, catchment_name):
    print(f"--- Training Baselines for {catchment_name} ---")
    
    # 1. Load Data
    df = pd.read_csv(csv_path, index_col='Date', parse_dates=True)
    
    # 2. Create Lagged Features (7-day memory)
    # This effectively turns our Linear Regression into an ARX model
    X, y = create_tabular_lags(df, target_col='discharge', lag_days=7)
    
    # 3. Chronological Train/Test Split
    # We train on the past to predict the future (1999-2008 Train, 2009-2013 Test)
    split_date = '2008-12-31'
    X_train = X.loc[:split_date]
    y_train = y.loc[:split_date]
    
    # 4. Scale the features (Crucial for Linear Regression so large numbers don't dominate)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Save the scaler so we can scale test data exactly the same way during evaluation
    os.makedirs('outputs/models', exist_ok=True)
    joblib.dump(scaler, f'outputs/models/{catchment_name}_scaler.joblib')

    # 5. Initialize Models with chosen hyperparameters
    models = {
        'MLR_ARX': LinearRegression(),
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'XGBoost': XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    }

    # 6. Train and Save Models
    for model_name, model in models.items():
        print(f"  Training {model_name}...")
        model.fit(X_train_scaled, y_train)
        
        # Save the trained model artifact
        model_path = f'outputs/models/{catchment_name}_{model_name}.joblib'
        joblib.dump(model, model_path)
        print(f"  Saved to {model_path}")

    print("\n")

if __name__ == "__main__":
    processed_dir = "data/processed"
    
    # Train Snow Catchment Models
    train_and_save_baselines(
        csv_path=os.path.join(processed_dir, "snow_processed.csv"),
        catchment_name="snow"
    )

    # Train Rain Catchment Models
    train_and_save_baselines(
        csv_path=os.path.join(processed_dir, "rain_processed.csv"),
        catchment_name="rain"
    )
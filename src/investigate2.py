import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Import our custom tools
from utils import create_tabular_lags, evaluate_model

def test_soil_moisture_hypothesis(csv_path):
    print("--- Testing Antecedent Moisture (API) Hypothesis on RAIN Catchment ---\n")
    
    # 1. Load Data
    df = pd.read_csv(csv_path, index_col='Date', parse_dates=True)
    
    # 2. Find the precipitation column dynamically
    prcp_col = [col for col in df.columns if 'prcp' in col.lower() or 'precip' in col.lower()][0]
    print(f"  Found precipitation column: {prcp_col}")
    
    # 3. Calculate Antecedent Precipitation Index (API)
    # Formula: API_today = (0.9 * API_yesterday) + Precipitation_today
    print("  Engineering new feature: 'API' (Proxy for Soil Moisture)...")
    k = 0.90 
    api_values = np.zeros(len(df))
    prcp_values = df[prcp_col].values
    
    api_values[0] = prcp_values[0]
    for i in range(1, len(df)):
        api_values[i] = (k * api_values[i-1]) + prcp_values[i]
        
    df['API'] = api_values
    
    # 4. Create Lags with the new API feature included
    X, y = create_tabular_lags(df, target_col='discharge', lag_days=7)
    
    # 5. Split Data (Train: 1999-2011, Test: 2012-2013)
    train_end = '2011-12-31'
    test_start = '2012-01-01'
    test_end = '2013-12-31'
    
    X_train = X.loc[:train_end]
    y_train = y.loc[:train_end]
    
    X_test = X.loc[test_start:test_end]
    y_test = y.loc[test_start:test_end]
    
    # 6. Scale Data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 7. Train a fresh Random Forest
    print("  Training new Random Forest with API feature included...")
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train_scaled, y_train.values)
    
    # 8. Predict and Evaluate
    preds = rf.predict(X_test_scaled)
    preds = np.maximum(preds, 0) # No negative flow
    
    results = evaluate_model(y_test.values, preds)
    
    # 9. Extract New Feature Importance
    importances = rf.feature_importances_
    fi_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    print("\n=========================================")
    print(" RESULTS WITH FAKE SOIL MOISTURE (API)")
    print("=========================================")
    print(f"  Old NSE: 0.004  |  New NSE:  {results['NSE']:.3f}")
    print(f"  Old RMSE: 76.08 |  New RMSE: {results['RMSE']:.3f}\n")
    
    print("  Top 5 Features used by the new model:")
    for idx, row in fi_df.head(5).iterrows():
        print(f"   - {row['Feature']}: {row['Importance']:.4f}")

if __name__ == "__main__":
    processed_dir = "data/processed"
    test_soil_moisture_hypothesis(os.path.join(processed_dir, "rain_processed.csv"))
import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Import custom tool to recreate the exact feature set
from utils import create_tabular_lags

def run_investigations(csv_path, catchment_name):
    print(f"\n--- Running Investigations for {catchment_name.upper()} Catchment ---")
    
    # 1. Rebuild the exact Test Set (2012-2013)
    df = pd.read_csv(csv_path, index_col='Date', parse_dates=True)
    test_start = '2012-01-01'
    test_end = '2013-12-31'
    
    # Need late-2011 context for 7-day lags
    df_context = df.loc['2011-12-20':test_end] 
    X_tab, y_tab = create_tabular_lags(df_context, target_col='discharge', lag_days=7)
    
    X_test = X_tab.loc[test_start:test_end]
    y_test = y_tab.loc[test_start:test_end]
    
    # Load Scaler and Model (Using Random Forest as our white-box model)
    scaler = joblib.load(f'outputs/models/{catchment_name}_scaler.joblib')
    rf_model = joblib.load(f'outputs/models/{catchment_name}_RandomForest.joblib')
    
    X_test_scaled = scaler.transform(X_test)
    preds = rf_model.predict(X_test_scaled)
    preds = np.maximum(preds, 0)
    
    # ==========================================
    # INVESTIGATION 1: Feature Importance
    # ==========================================
    # Extract importances and map to column names
    importances = rf_model.feature_importances_
    feature_names = X_test.columns
    
    fi_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    # Save Table
    os.makedirs('outputs/tables', exist_ok=True)
    fi_df.to_csv(f'outputs/tables/{catchment_name}_feature_importance.csv', index=False)
    print(f"  Top 3 Features: {fi_df['Feature'].head(3).tolist()}")
    
    # Plot Top 15 Features
    plt.figure(figsize=(10, 6))
    top_fi = fi_df.head(15)
    plt.barh(top_fi['Feature'][::-1], top_fi['Importance'][::-1], color='teal')
    plt.title(f'{catchment_name.capitalize()}: Top 15 Feature Importances (Random Forest)')
    plt.xlabel('Relative Importance')
    plt.tight_layout()
    
    os.makedirs('outputs/figures', exist_ok=True)
    plt.savefig(f'outputs/figures/{catchment_name}_feature_importance.png', dpi=300)
    plt.close()

    # ==========================================
    # INVESTIGATION 2: Extreme Event Errors
    # ==========================================
    # Define "Extreme" as the top 10% of observed flows
    threshold = np.percentile(y_test.values, 90)
    
    # Split indices into Extreme vs Normal
    extreme_mask = y_test.values >= threshold
    normal_mask = ~extreme_mask
    
    # Calculate RMSE for both sets
    rmse_extreme = np.sqrt(mean_squared_error(y_test.values[extreme_mask], preds[extreme_mask]))
    rmse_normal = np.sqrt(mean_squared_error(y_test.values[normal_mask], preds[normal_mask]))
    
    # Create a summary dictionary
    extreme_results = {
        'Catchment': catchment_name.capitalize(),
        '90th_Percentile_Threshold': threshold,
        'Normal_Days_Count': normal_mask.sum(),
        'Extreme_Days_Count': extreme_mask.sum(),
        'RMSE_Normal_Flows': rmse_normal,
        'RMSE_Extreme_Flows': rmse_extreme,
        'Error_Multiplier': rmse_extreme / rmse_normal if rmse_normal > 0 else 0
    }
    
    extreme_df = pd.DataFrame([extreme_results])
    extreme_df.to_csv(f'outputs/tables/{catchment_name}_extreme_errors.csv', index=False)
    print(f"  Normal RMSE: {rmse_normal:.2f} | Extreme RMSE: {rmse_extreme:.2f}")

if __name__ == "__main__":
    processed_dir = "data/processed"
    run_investigations(os.path.join(processed_dir, "snow_processed.csv"), "snow")
    run_investigations(os.path.join(processed_dir, "rain_processed.csv"), "rain")
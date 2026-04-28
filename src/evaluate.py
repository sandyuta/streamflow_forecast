import pandas as pd
import numpy as np
import os
import joblib
import torch
import matplotlib.pyplot as plt

# Import our custom tools and architectures
from utils import create_tabular_lags, create_sequences, evaluate_model
from train_dl_models import SimpleRNN, SimpleLSTM

def evaluate_catchment(csv_path, catchment_name):
    print(f"\n=========================================")
    print(f" FINAL EVALUATION: {catchment_name.upper()} CATCHMENT")
    print(f"=========================================")
    
    # 1. Load Data & Isolate the Unseen Test Set (2012 - 2013)
    df = pd.read_csv(csv_path, index_col='Date', parse_dates=True)
    test_start = '2012-01-01'
    test_end = '2013-12-31'
    df_test = df.loc[test_start:test_end]
    
    # We need late-2011 data to create the 7-day lags for Jan 1, 2012
    df_context = df.loc['2011-12-20':test_end] 
    
    # --- A. EVALUATE CLASSICAL ML BASELINES ---
    X_tab, y_tab = create_tabular_lags(df_context, target_col='discharge', lag_days=7)
    X_test_tab = X_tab.loc[test_start:test_end]
    y_test_tab = y_tab.loc[test_start:test_end]
    
    # Load scaler and scale test features
    tab_scaler = joblib.load(f'outputs/models/{catchment_name}_scaler.joblib')
    X_test_tab_scaled = tab_scaler.transform(X_test_tab)
    
    results = {}
    predictions_dict = {'Observed': y_test_tab.values}
    
    baseline_models = ['MLR_ARX', 'RandomForest', 'XGBoost']
    for model_name in baseline_models:
        model = joblib.load(f'outputs/models/{catchment_name}_{model_name}.joblib')
        preds = model.predict(X_test_tab_scaled)
        
        # Hydrology constraint: streamflow cannot be negative
        preds = np.maximum(preds, 0)
        
        predictions_dict[model_name] = preds
        results[model_name] = evaluate_model(y_test_tab.values, preds)

    # --- B. EVALUATE DEEP LEARNING MODELS ---
    X_raw = df_context.drop(columns=['discharge']).values
    y_raw = df_context['discharge'].values.reshape(-1, 1)
    
    dl_X_scaler = joblib.load(f'outputs/models/{catchment_name}_dl_X_scaler.joblib')
    dl_y_scaler = joblib.load(f'outputs/models/{catchment_name}_dl_y_scaler.joblib')
    
    X_scaled = dl_X_scaler.transform(X_raw)
    y_scaled = dl_y_scaler.transform(y_raw)
    
    X_seq, _ = create_sequences(X_scaled, y_scaled, seq_length=7)
    X_tensor = torch.tensor(X_seq, dtype=torch.float32)
    
    dl_models = {'RNN': SimpleRNN, 'LSTM': SimpleLSTM}
    input_size = X_scaled.shape[1]
    
    for model_name, ModelClass in dl_models.items():
        model = ModelClass(input_size=input_size)
        model.load_state_dict(torch.load(f'outputs/models/{catchment_name}_{model_name}.pth'))
        model.eval()
        
        with torch.no_grad():
            preds_scaled = model(X_tensor).numpy()
            
        # Inverse transform to get real-world discharge values (m3/s)
        preds = dl_y_scaler.inverse_transform(preds_scaled).flatten()
        preds = np.maximum(preds, 0)
        
        # Trim to match exact length of the test set
        preds = preds[-len(y_test_tab):]
        
        predictions_dict[model_name] = preds
        results[model_name] = evaluate_model(y_test_tab.values, preds)

    # --- C. GENERATE FINAL OUTPUTS ---
    # 1. Save Results Table
    results_df = pd.DataFrame(results).T
    results_df = results_df.round(3) # Round to 3 decimal places for a clean table
    os.makedirs('outputs/tables', exist_ok=True)
    results_df.to_csv(f'outputs/tables/{catchment_name}_final_metrics.csv')
    print("\n--- Final Performance Metrics ---")
    print(results_df)

    # 2. Plot Final Comparison Hydrograph
    plt.figure(figsize=(14, 6))
    plt.plot(y_test_tab.index, predictions_dict['Observed'], label='Observed (True)', color='black', linewidth=1.5, alpha=0.8)
    plt.plot(y_test_tab.index, predictions_dict['LSTM'], label='LSTM', color='blue', linewidth=1.2, alpha=0.7)
    plt.plot(y_test_tab.index, predictions_dict['RandomForest'], label='Random Forest', color='orange', linewidth=1.2, alpha=0.7)
    
    plt.title(f'{catchment_name.capitalize()} Catchment: Unseen Test Period (2012-2013)', fontsize=14, fontweight='bold')
    plt.ylabel('Streamflow Discharge', fontsize=12)
    plt.xlabel('Date', fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    os.makedirs('outputs/figures', exist_ok=True)
    plt.savefig(f'outputs/figures/{catchment_name}_final_predictions.png', dpi=300)
    plt.close()
    print(f"\n  Saved hydrograph to outputs/figures/{catchment_name}_final_predictions.png\n")

if __name__ == "__main__":
    processed_dir = "data/processed"
    
    evaluate_catchment(os.path.join(processed_dir, "snow_processed.csv"), "snow")
    evaluate_catchment(os.path.join(processed_dir, "rain_processed.csv"), "rain")
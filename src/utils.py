import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

# ==========================================
# 1. EVALUATION METRICS
# ==========================================

def calculate_rmse(y_true, y_pred):
    """Calculates Root Mean Square Error."""
    return np.sqrt(mean_squared_error(y_true, y_pred))

def calculate_nse(y_true, y_pred):
    """
    Calculates Nash-Sutcliffe Efficiency (NSE).
    NSE = 1 indicates a perfect match.
    NSE < 0 indicates model is worse than predicting the mean.
    """
    denominator = np.sum((y_true - np.mean(y_true)) ** 2)
    if denominator == 0:
        return np.nan # Avoid division by zero
    
    numerator = np.sum((y_true - y_pred) ** 2)
    nse = 1 - (numerator / denominator)
    return nse

def evaluate_model(y_true, y_pred):
    """Returns a dictionary of all metrics."""
    return {
        'RMSE': calculate_rmse(y_true, y_pred),
        'NSE': calculate_nse(y_true, y_pred)
    }

# ==========================================
# 2. DATA SHAPING FOR CLASSICAL ML
# ==========================================

def create_tabular_lags(df, target_col='discharge', lag_days=7):
    """
    Creates lagged features for classical ML models (RF, XGBoost, MLR).
    
    Parameters:
    - df: Processed pandas DataFrame
    - target_col: Name of the column to predict
    - lag_days: How many days of past data to include as features
    
    Returns:
    - X: Feature matrix (DataFrame)
    - y: Target vector (Series) aligned with t+1
    """
    df_lagged = df.copy()
    feature_cols = df.columns.tolist()

    # Create lagged features for all variables (Precip, Temp, Discharge)
    for i in range(1, lag_days + 1):
        for col in feature_cols:
            df_lagged[f'{col}_lag_{i}'] = df_lagged[col].shift(i)

    # The target is the next day's streamflow (t+1)
    df_lagged['target_t_plus_1'] = df_lagged[target_col].shift(-1)

    # Drop rows with NaNs introduced by shifting
    df_lagged = df_lagged.dropna()

    # Separate X (features) and y (target)
    # We drop the original un-lagged columns to strictly use past data to predict the future
    X = df_lagged.drop(columns=feature_cols + ['target_t_plus_1'])
    y = df_lagged['target_t_plus_1']
    
    return X, y

# ==========================================
# 3. DATA SHAPING FOR DEEP LEARNING
# ==========================================

def create_sequences(features_array, target_array, seq_length=7):
    """
    Creates overlapping 3D sequences for PyTorch models (RNN/LSTM).
    
    Parameters:
    - features_array: Numpy array of scaled features
    - target_array: Numpy array of scaled targets
    - seq_length: Size of the sliding window (TimeSteps)
    
    Returns:
    - X_seq: 3D Numpy array [Samples, TimeSteps, Features]
    - y_seq: 1D Numpy array [Target at t+1]
    """
    X_list = []
    y_list = []
    
    # Slide the window across the data
    for i in range(len(features_array) - seq_length):
        X_list.append(features_array[i:(i + seq_length)])
        y_list.append(target_array[i + seq_length])
        
    return np.array(X_list), np.array(y_list)
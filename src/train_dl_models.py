import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import joblib

from utils import create_sequences

# ==========================================
# 1. DEFINE PYTORCH ARCHITECTURES
# ==========================================
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :]) 
        return out

class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# ==========================================
# 2. TRAINING LOOP WITH VALIDATION
# ==========================================
def train_dl_models(csv_path, catchment_name, seq_length=7, epochs=50, batch_size=32, lr=0.001):
    print(f"--- Training Deep Learning Models for {catchment_name.upper()} ---")
    
    # 1. Load Data
    df = pd.read_csv(csv_path, index_col='Date', parse_dates=True)
    
    # 10/3/2 Split Boundaries
    train_end = '2008-12-31'
    val_end = '2011-12-31'
    
    df_train = df.loc[:train_end]
    # We include 7 days prior to 2009 to have complete sequences for the start of validation
    df_val = df.loc['2008-12-25':val_end] 
    
    # Separate Features and Target
    X_train_raw = df_train.drop(columns=['discharge']).values
    y_train_raw = df_train['discharge'].values.reshape(-1, 1)
    
    X_val_raw = df_val.drop(columns=['discharge']).values
    y_val_raw = df_val['discharge'].values.reshape(-1, 1)
    
    # 2. Scale Data (Fit ONLY on training data to prevent data leakage)
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()
    
    X_train_scaled = X_scaler.fit_transform(X_train_raw)
    y_train_scaled = y_scaler.fit_transform(y_train_raw)
    
    X_val_scaled = X_scaler.transform(X_val_raw)
    y_val_scaled = y_scaler.transform(y_val_raw)
    
    # Save scalers
    os.makedirs('outputs/models', exist_ok=True)
    joblib.dump(X_scaler, f'outputs/models/{catchment_name}_dl_X_scaler.joblib')
    joblib.dump(y_scaler, f'outputs/models/{catchment_name}_dl_y_scaler.joblib')

    # 3. Create Sequences & DataLoaders
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, seq_length)
    X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val_scaled, seq_length)
    
    train_loader = DataLoader(TensorDataset(torch.tensor(X_train_seq, dtype=torch.float32), 
                                            torch.tensor(y_train_seq, dtype=torch.float32)), 
                              batch_size=batch_size, shuffle=True)
    
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val_seq, dtype=torch.float32), 
                                          torch.tensor(y_val_seq, dtype=torch.float32)), 
                            batch_size=batch_size, shuffle=False)
    
    # 4. Initialize Models
    input_size = X_train_scaled.shape[1]
    models = {
        'RNN': SimpleRNN(input_size=input_size),
        'LSTM': SimpleLSTM(input_size=input_size)
    }
    
    loss_function = nn.MSELoss()

    # 5. Train with Validation (Early Stopping logic)
    for model_name, model in models.items():
        print(f"  Training {model_name}...")
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Training Phase
            model.train()
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                predictions = model(batch_X)
                loss = loss_function(predictions, batch_y)
                loss.backward()
                optimizer.step()
                
            # Validation Phase
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    predictions = model(batch_X)
                    loss = loss_function(predictions, batch_y)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            
            # Save the model if validation loss improves
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), f'outputs/models/{catchment_name}_{model_name}.pth')
                
            if (epoch + 1) % 10 == 0:
                print(f"    Epoch {epoch+1}/{epochs} | Val Loss: {avg_val_loss:.4f}")

        print(f"  Saved best {model_name} weights (Val Loss: {best_val_loss:.4f}).\n")

if __name__ == "__main__":
    processed_dir = "data/processed"
    train_dl_models(os.path.join(processed_dir, "snow_processed.csv"), "snow")
    train_dl_models(os.path.join(processed_dir, "rain_processed.csv"), "rain")
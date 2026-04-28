import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import joblib

# Import custom sequence generator
from utils import create_sequences

# ==========================================
# 1. DEFINE PYTORCH ARCHITECTURES
# ==========================================
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1) # Output a single discharge value

    def forward(self, x):
        out, hidden = self.rnn(x)
        # We only want the output from the very last time step in the sequence
        out = self.fc(out[:, -1, :]) 
        return out

class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, (hn, cn) = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# ==========================================
# 2. TRAINING LOOP
# ==========================================
def train_dl_models(csv_path, catchment_name, seq_length=7, epochs=50, batch_size=32, lr=0.001):
    print(f"--- Training Deep Learning Models for {catchment_name} ---")
    
    # 1. Load Data
    df = pd.read_csv(csv_path, index_col='Date', parse_dates=True)
    
    # Chronological Split (1999-2008 Train)
    split_date = '2008-12-31'
    df_train = df.loc[:split_date]
    
    # Separate Features and Target
    X_train_raw = df_train.drop(columns=['discharge']).values
    y_train_raw = df_train['discharge'].values.reshape(-1, 1)
    
    # 2. Scale Data (Neural networks require targets to be scaled too)
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()
    
    X_train_scaled = X_scaler.fit_transform(X_train_raw)
    y_train_scaled = y_scaler.fit_transform(y_train_raw)
    
    # Save scalers for evaluation
    os.makedirs('outputs/models', exist_ok=True)
    joblib.dump(X_scaler, f'outputs/models/{catchment_name}_dl_X_scaler.joblib')
    joblib.dump(y_scaler, f'outputs/models/{catchment_name}_dl_y_scaler.joblib')

    # 3. Create 3D Sequences [Samples, TimeSteps, Features]
    X_seq, y_seq = create_sequences(X_train_scaled, y_train_scaled, seq_length)
    
    # Convert to PyTorch Tensors
    X_tensor = torch.tensor(X_seq, dtype=torch.float32)
    y_tensor = torch.tensor(y_seq, dtype=torch.float32)
    
    # Create DataLoader
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 4. Initialize Models
    input_size = X_train_scaled.shape[1]
    models = {
        'RNN': SimpleRNN(input_size=input_size),
        'LSTM': SimpleLSTM(input_size=input_size)
    }
    
    loss_function = nn.MSELoss()

    # 5. Train
    for model_name, model in models.items():
        print(f"  Training {model_name}...")
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                predictions = model(batch_X)
                loss = loss_function(predictions, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                
            if (epoch + 1) % 10 == 0:
                print(f"    Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/len(dataloader):.4f}")

        # Save the model weights
        torch.save(model.state_dict(), f'outputs/models/{catchment_name}_{model_name}.pth')
        print(f"  Saved {model_name} weights.\n")

if __name__ == "__main__":
    processed_dir = "data/processed"
    
    # Train Snow Catchment DL
    train_dl_models(
        csv_path=os.path.join(processed_dir, "snow_processed.csv"),
        catchment_name="snow"
    )

    # Train Rain Catchment DL
    train_dl_models(
        csv_path=os.path.join(processed_dir, "rain_processed.csv"),
        catchment_name="rain"
    )
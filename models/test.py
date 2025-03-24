import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from scipy.fftpack import fft, ifft
import test_model
import matplotlib.pyplot as plt

# Load ETTh Dataset
def load_etth_dataset(file_path):
    data = pd.read_csv(file_path)
    data['date'] = pd.to_datetime(data['date'])
    return data

# Extract Time-Related Features
def extract_time_features(data):
    data['hour'] = data['date'].dt.hour
    data['day'] = data['date'].dt.dayofweek
    data['month'] = data['date'].dt.month
    data['season'] = (data['month'] % 12 + 3) // 3  # 1: Winter, 2: Spring, 3: Summer, 4: Fall
    return data[['hour', 'day', 'month', 'season']]

# Normalize Features
def normalize_features(features):
    scaler = MinMaxScaler(feature_range=(-0.5, 0.5))
    return scaler.fit_transform(features)

# Cyclic Feature Extraction using FFT
def extract_cyclic_features(time_series, top_freq=5):
    fft_values = fft(time_series)
    frequencies = np.argsort(np.abs(fft_values))[-top_freq:]  # Select top frequencies
    cyclic_features = np.real(ifft(fft_values[frequencies]))
    return cyclic_features

# Prepare Dataset
def prepare_dataset(data, look_back=24, horizon=12):
    X, y = [], []
    for i in range(len(data) - look_back - horizon):
        X.append(data[i:i+look_back])
        y.append(data[i+look_back:i+look_back+horizon])
    return np.array(X), np.array(y)

# Lightweight Forecasting Model
class LightweightForecaster(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LightweightForecaster, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        return self.linear2(x)

# Train Model
def train_model(model, X_train, y_train, epochs=100, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Explain Model using SHAP
def explain_model(model, X_test):
    explainer = test_model.DeepExplainer(model, X_test[:100])  # Use a subset for faster computation
    shap_values = explainer.shap_values(X_test[:10])  # Explain first 10 samples
    test_model.summary_plot(shap_values, X_test[:10], feature_names=['hour', 'day', 'month', 'season', 'cyclic'])

# Main Function
def main():
    # Load and Preprocess Data
    data = load_etth_dataset('.\dataset\ETT-small\ETTh1.csv')
    time_features = extract_time_features(data)
    time_features = normalize_features(time_features)
    cyclic_features = extract_cyclic_features(data['OT'].values)  # Use Oil Temperature as target
    features = np.hstack((time_features, cyclic_features.reshape(-1, 1)))

    # Prepare Dataset
    X, y = prepare_dataset(features)
    X_train, y_train = torch.FloatTensor(X[:int(0.8*len(X))]), torch.FloatTensor(y[:int(0.8*len(y))])
    X_test, y_test = torch.FloatTensor(X[int(0.8*len(X)):]), torch.FloatTensor(y[int(0.8*len(y)):])

    # Initialize and Train Model
    model = LightweightForecaster(input_dim=X_train.shape[2], hidden_dim=64, output_dim=y_train.shape[2])
    train_model(model, X_train, y_train)

    # Evaluate Model
    with torch.no_grad():
        predictions = model(X_test)
        mse = nn.MSELoss()(predictions, y_test).item()
        print(f'Test MSE: {mse:.4f}')

    # Explain Model
    explain_model(model, X_test)

if __name__ == '__main__':
    main()
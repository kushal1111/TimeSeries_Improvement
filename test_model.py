import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from scipy.fftpack import fft, ifft
import shap # Explicitly import DeepExplainer
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import lime
import lime.lime_tabular

def explain_with_lime(model, X_test, feature_names, num_features=5):
    """
    Explain individual predictions using LIME.
    :param model: Trained model.
    :param X_test: Test data (n_samples, n_features).
    :param feature_names: List of feature names.
    :param num_features: Number of features to display in the explanation.
    """
    # Wrap the model to return a single-dimensional output
    def model_wrapper(x):
        x_tensor = torch.FloatTensor(x)
        predictions = model(x_tensor)  # Shape: (n_samples, horizon, output_dim)
        return predictions.mean(dim=1).detach().numpy()  # Aggregate across horizon

    # Create LIME explainer
    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_test, feature_names=feature_names, mode='regression'
    )

    # Explain a single instance
    instance_idx = 0  # Explain the first instance
    explanation = explainer.explain_instance(X_test[instance_idx], model_wrapper, num_features=num_features)

    # Print explanation in text format
    print("LIME Explanation:")
    for feature, weight in explanation.as_list():
        print(f"{feature}: {weight}")

def visualize_tsne(model, X_test):
    """
    Visualize feature embeddings using t-SNE.
    :param model: Trained model.
    :param X_test: Test data (n_samples, look_back, n_features).
    """
    # Flatten the input tensor: [n_samples, look_back, n_features] -> [n_samples, look_back * n_features]
    X_test_flat = X_test.view(X_test.size(0), -1)

    # Pass the flattened input through the first linear layer to get feature embeddings
    feature_embeddings = model.linear1(X_test_flat).detach().numpy()

    # Apply t-SNE to reduce dimensionality to 2D
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    embeddings_2d = tsne.fit_transform(feature_embeddings)

    # Plot the t-SNE visualization
    plt.figure(figsize=(8, 6))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.6)
    plt.title('t-SNE Visualization of Feature Embeddings')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.show()

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

# Lightweight Forecasting Model
class LightweightForecaster(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, horizon=12):
        super(LightweightForecaster, self).__init__()
        self.input_dim = input_dim
        self.horizon = horizon
        self.output_dim = output_dim
        look_back=24
        self.linear1 = nn.Linear(input_dim * look_back, hidden_dim)  # Flatten input
        self.linear2 = nn.Linear(hidden_dim, output_dim * horizon)  # Predict horizon steps
        self.relu = nn.ReLU()

    def forward(self, x):
        batch_size = x.shape[0]  # Get batch size
        # Flatten the input: [batch_size, look_back, input_dim] -> [batch_size, look_back * input_dim]
        x = x.view(batch_size, -1)
        x = self.relu(self.linear1(x))  # Pass through first linear layer
        x = self.linear2(x)  # Shape: [batch_size, output_dim * horizon]
        # Reshape to [batch_size, horizon, output_dim]
        x = x.view(batch_size, self.horizon, self.output_dim)
        return x

# Prepare Dataset
def prepare_dataset(data, look_back=24, horizon=12):
    X, y = [], []
    for i in range(len(data) - look_back - horizon):
        X.append(data[i:i+look_back])
        y.append(data[i+look_back:i+look_back+horizon])
    return np.array(X), np.array(y)

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

def explain_model(model, X_test):
    # Convert PyTorch tensor to NumPy array
    X_test_np = X_test.numpy()
    look_back=24
    # Flatten the input data to 2 dimensions: [n_samples, look_back * n_features]
    X_test_np_flat = X_test_np.reshape(X_test_np.shape[0], -1)

    # Wrap the model to return a scalar output
    def model_wrapper(x):
        # Reshape the input back to 3 dimensions: [n_samples, look_back, n_features]
        x_reshaped = x.reshape(x.shape[0], look_back, -1)
        x_tensor = torch.FloatTensor(x_reshaped)  # Convert NumPy array back to PyTorch tensor
        return model(x_tensor).mean(dim=1).detach().numpy()  # Aggregate outputs to a scalar

    # Use KernelExplainer with flattened input
    explainer = shap.KernelExplainer(model_wrapper, X_test_np_flat[:100])  # Use a subset for faster computation
    shap_values = explainer.shap_values(X_test_np_flat[:50])  # Explain first 10 samples

    # Generate feature names for LIME
    feature_names = [f't{i+1}_f{j+1}' for i in range(look_back) for j in range(X_test_np.shape[2])]

    # Explain predictions using LIME
    explain_with_lime(model, X_test_np_flat[:10], feature_names)

    # Handle single-output and multi-output models
    if isinstance(shap_values, list):
        # Multi-output model: shap_values is a list of arrays
        for i, sv in enumerate(shap_values):
            # Reshape SHAP values to match the input data shape: [n_samples, look_back * n_features]
            shap_values_reshaped = sv.reshape(50, look_back, -1)
            shap_values_flat = shap_values_reshaped.reshape(50, -1)

            # Generate feature names for each time step and feature
            feature_names = [f't{i+1}_f{j+1}' for i in range(look_back) for j in range(X_test_np.shape[2])]

            # Visualize SHAP values for each output
            print(f"Visualizing SHAP values for output {i+1}")
            shap.summary_plot(shap_values_flat, X_test_np_flat[:50], feature_names=feature_names)
    else:
        # Single-output model: shap_values is a single array
        # Reshape SHAP values to match the input data shape: [n_samples, look_back * n_features]
        shap_values_reshaped = np.mean(shap_values, axis=2)
        shap_values_flat = shap_values_reshaped.reshape(50, -1)

        # Generate feature names for each time step and feature
        feature_names = [f't{i+1}_f{j+1}' for i in range(look_back) for j in range(X_test_np.shape[2])]

        # Visualize SHAP values
        shap.summary_plot(shap_values_flat, X_test_np_flat[:50], feature_names=feature_names)

def ablation_study(model, X_test, y_test, feature_groups):
    """
    Perform ablation studies by removing specific feature groups.
    :param model: Trained model.
    :param X_test: Test data (n_samples, look_back, n_features).
    :param y_test: Test labels (n_samples, horizon, output_dim).
    :param feature_groups: Dictionary of feature groups to ablate.
    """
    results = {}
    for group_name, feature_indices in feature_groups.items():
        # Create a copy of the test data and remove the feature group
        X_test_ablated = X_test.clone()
        X_test_ablated[:, :, feature_indices] = 0  # Set feature values to 0

        # Evaluate the model on the ablated data
        with torch.no_grad():
            predictions = model(X_test_ablated)
            mse = nn.MSELoss()(predictions, y_test).item()

        # Store the results
        results[group_name] = mse
        print(f'Ablation: {group_name}, Test MSE: {mse:.4f}')

    return results

# Main Function
# Cyclic Feature Extraction using FFT for each time step
def extract_cyclic_features(time_series, window_size=24, top_freq=5):
    cyclic_features = []
    for i in range(len(time_series) - window_size):
        segment = time_series[i:i+window_size]
        fft_values = fft(segment)
        frequencies = np.argsort(np.abs(fft_values))[-top_freq:]  # Select top frequencies
        cyclic_feature = np.real(ifft(fft_values[frequencies])).mean()  # Mean of top frequencies
        cyclic_features.append(cyclic_feature)
    return np.array(cyclic_features)

# Prepare Dataset
def prepare_dataset(data, look_back=24, horizon=12):
    X, y = [], []
    for i in range(len(data) - look_back - horizon):
        X.append(data[i:i+look_back])
        y.append(data[i+look_back:i+look_back+horizon])
    return np.array(X), np.array(y)

# Main Function
# Main Function
# Main Function
# Main Function
def main():
    # Load and Preprocess Data
    data = load_etth_dataset('.\dataset\ETT-small\ETTh1.csv')
    time_features = extract_time_features(data)
    time_features = normalize_features(time_features)

    # Extract cyclic features for each time step
    cyclic_features = extract_cyclic_features(data['OT'].values)  # Use Oil Temperature as target
    cyclic_features = cyclic_features.reshape(-1, 1)  # Reshape to (n_samples, 1)

    # Ensure both time_features and cyclic_features have the same number of samples
    min_samples = min(len(time_features), len(cyclic_features))
    time_features = time_features[:min_samples]
    cyclic_features = cyclic_features[:min_samples]

    # Combine features
    features = np.hstack((time_features, cyclic_features))

    # Prepare Dataset
    look_back = 24  # Match this with the model's look_back
    horizon = 12  # Match this with the model's horizon
    X, y = prepare_dataset(features, look_back=look_back, horizon=horizon)
    X_train, y_train = torch.FloatTensor(X[:int(0.8*len(X))]), torch.FloatTensor(y[:int(0.8*len(y))])
    X_test, y_test = torch.FloatTensor(X[int(0.8*len(X)):]), torch.FloatTensor(y[int(0.8*len(y)):])

    # Verify shapes
    print(f"X_train shape: {X_train.shape}")  # Should be (batch_size, look_back, input_dim)
    print(f"y_train shape: {y_train.shape}")  # Should be (batch_size, horizon, output_dim)

    # Initialize and Train Model
    model = LightweightForecaster(input_dim=X_train.shape[2], hidden_dim=64, output_dim=y_train.shape[2], horizon=horizon)
    outputs = model(X_train)
    print(f"Outputs shape: {outputs.shape}")  # Should be (batch_size, output_dim)

    train_model(model, X_train, y_train)

    # Evaluate Model
    with torch.no_grad():
        predictions = model(X_test)
        mse = nn.MSELoss()(predictions, y_test).item()
        print(f'Test MSE: {mse:.4f}')

    # Explain Model
    explain_model(model, X_test)

    # t-SNE Visualization
    visualize_tsne(model, X_test)

    # Ablation Studies
    feature_groups = {
        'Time Features': [0, 1, 2, 3],  # hour, day, month, season
        'Cyclic Feature': [4],           # cyclic
        'All Features': list(range(5))   # all features
    }
    ablation_results = ablation_study(model, X_test, y_test, feature_groups)
    print(ablation_results)

if __name__ == '__main__':
    main()
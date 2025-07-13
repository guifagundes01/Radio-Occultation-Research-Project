import torch
import numpy as np
import os
from model.chapman_hybrid import ChapmanPredictor, ProfileCorrector, generate_chapman_profiles
import matplotlib.pyplot as plt
import random
from torch.utils.data import TensorDataset, DataLoader

# Paths
model_dir = './data/fit_results/hybrid_model'
X_scaler_path = os.path.join(model_dir, 'X_scaler.npy')
y_scaler_path = os.path.join(model_dir, 'y_scaler.npy')
model_ckpt_path = os.path.join(model_dir, 'best_hybrid_model.pth')
data_path = './data/filtered/electron_density_profiles_2023_with_fits.h5'

import h5py

def load_test_data():
    with h5py.File(data_path, 'r') as f:
        latitude = f['latitude'][:]
        longitude = f['longitude'][:]
        local_time = f['local_time'][:]
        f107 = f['f107'][:]
        kp = f['kp'][:]
        dip = f['dip'][:]
        altitude = f['altitude'][:]
        electron_density = f['electron_density'][:]
        chapman_params = f['fit_results/chapman_params'][:]
        is_good_fit = f['fit_results/is_good_fit'][:]
        # Filter
        latitude = latitude[is_good_fit]
        longitude = longitude[is_good_fit]
        local_time = local_time[is_good_fit]
        f107 = f107[is_good_fit]
        kp = kp[is_good_fit]
        dip = dip[is_good_fit]
        chapman_params = chapman_params[is_good_fit]
        electron_density = electron_density[is_good_fit]
        return latitude, longitude, local_time, f107, kp, dip, altitude, chapman_params, electron_density

def circular_encode(value, max_value):
    sin_val = np.sin(2 * np.pi * value / max_value)
    cos_val = np.cos(2 * np.pi * value / max_value)
    return sin_val, cos_val

def convert_seconds_to_datetime(seconds):
    import pandas as pd
    local_time_date = pd.to_datetime(seconds, unit='s')
    return {
        'year': local_time_date.year,
        'month': local_time_date.month,
        'doy': local_time_date.dayofyear,
        'hour': local_time_date.hour,
        'minute': local_time_date.minute
    }

def prepare_features(latitude, longitude, local_time, f107, kp, dip):
    temporal_features = []
    for seconds in local_time:
        dt = convert_seconds_to_datetime(seconds)
        month_sin, month_cos = circular_encode(dt['month'], 12)
        doy_sin, doy_cos = circular_encode(dt['doy'], 365)
        hour_sin, hour_cos = circular_encode(dt['hour'], 24)
        temporal_features.append([
            month_sin, month_cos, doy_sin, doy_cos, hour_sin, hour_cos
        ])
    temporal_features = np.array(temporal_features)
    X = np.column_stack([
        latitude, longitude, temporal_features, f107, kp, dip
    ])
    return X

def plot_random_profiles(altitude, true_profiles, chapman_profiles, corrected_profiles, n=5, save_path=None):
    idxs = random.sample(range(true_profiles.shape[0]), n)
    plt.figure(figsize=(10, 6))
    for i, idx in enumerate(idxs):
        plt.plot(altitude, true_profiles[idx], label=f'True #{idx}', color='black', alpha=0.5 if n > 1 else 1)
        plt.plot(altitude, chapman_profiles[idx], label=f'Chapman #{idx}', linestyle='--', color='blue', alpha=0.5 if n > 1 else 1)
        plt.plot(altitude, corrected_profiles[idx], label=f'Corrected #{idx}', linestyle='-.', color='green', alpha=0.5 if n > 1 else 1)
    plt.xlabel('Altitude (km)')
    plt.ylabel('Electron Density')
    plt.title('Random Profile Comparisons')
    if n == 1:
        plt.legend()
    else:
        plt.legend(['True', 'Chapman', 'Corrected'])
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_error_histograms(true_profiles, chapman_profiles, corrected_profiles, save_path=None):
    errors_chapman = (chapman_profiles - true_profiles).flatten()
    errors_corrected = (corrected_profiles - true_profiles).flatten()
    plt.figure(figsize=(10, 5))
    plt.hist(errors_chapman, bins=100, alpha=0.5, label='Chapman Error', color='blue')
    plt.hist(errors_corrected, bins=100, alpha=0.5, label='Corrected Error', color='green')
    plt.xlabel('Error (Predicted - True)')
    plt.ylabel('Count')
    plt.title('Error Histogram')
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_true_vs_pred(true_profiles, corrected_profiles, save_path=None):
    plt.figure(figsize=(6, 6))
    plt.scatter(true_profiles.flatten(), corrected_profiles.flatten(), s=1, alpha=0.3, color='purple')
    minval = min(true_profiles.min(), corrected_profiles.min())
    maxval = max(true_profiles.max(), corrected_profiles.max())
    plt.plot([minval, maxval], [minval, maxval], 'k--', lw=2)
    plt.xlabel('True Electron Density')
    plt.ylabel('Predicted (Corrected) Electron Density')
    plt.title('True vs. Predicted (Corrected) Electron Density')
    if save_path:
        plt.savefig(save_path)
    plt.close()

def main():
    # Load test data
    latitude, longitude, local_time, f107, kp, dip, altitude, chapman_params, electron_density = load_test_data()
    X = prepare_features(latitude, longitude, local_time, f107, kp, dip)
    # Load scalers
    X_scaler = np.load(X_scaler_path, allow_pickle=True).item() if X_scaler_path.endswith('.npy') else np.load(X_scaler_path, allow_pickle=True)
    y_scaler = np.load(y_scaler_path, allow_pickle=True).item() if y_scaler_path.endswith('.npy') else np.load(y_scaler_path, allow_pickle=True)
    X_scaled = X_scaler.transform(X)
    # Load models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_stage1 = ChapmanPredictor(input_size=11, hidden_sizes=[128, 64]).to(device)
    model_stage2 = ProfileCorrector(profile_size=len(altitude), feature_size=11, hidden_sizes=[128, 64]).to(device)
    checkpoint = torch.load(model_ckpt_path, map_location=device)
    model_stage1.load_state_dict(checkpoint['stage1_state_dict'])
    model_stage2.load_state_dict(checkpoint['stage2_state_dict'])
    model_stage1.eval()
    model_stage2.eval()
    # Prepare tensors
    X_tensor = torch.FloatTensor(X_scaled)
    altitude_tensor = torch.FloatTensor(altitude)
    batch_size = 32
    test_dataset = TensorDataset(X_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    all_chapman_profiles = []
    all_corrections = []
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            X_batch = batch[0].to(device)
            chapman_params_pred = model_stage1(X_batch)
            altitude_batch = altitude_tensor.unsqueeze(0).repeat(X_batch.shape[0], 1).to(device)
            chapman_profiles = generate_chapman_profiles(chapman_params_pred, altitude_batch)
            if chapman_profiles.dim() == 3 and chapman_profiles.shape[1] == 1:
                chapman_profiles = chapman_profiles.squeeze(1)
            corrections = model_stage2(chapman_profiles, X_batch, chapman_params_pred)
            # Robust shape checks
            expected_shape = (X_batch.shape[0], altitude_tensor.shape[0])
            # if chapman_profiles.shape != expected_shape or corrections.shape != expected_shape:
            #     print(f"Skipping batch {i}: chapman_profiles shape = {chapman_profiles.shape}, corrections shape = {corrections.shape}, expected = {expected_shape}")
            #     continue
            # if chapman_profiles.shape[0] == 0 or corrections.shape[0] == 0:
            #     print(f"Skipping batch {i}: zero-size batch.")
            #     continue
            # If corrections has shape [batch_size, batch_size, num_altitude_points], take the diagonal
            if corrections.dim() == 3 and corrections.shape[0] == corrections.shape[1]:
                corrections = corrections.diagonal(dim1=0, dim2=1).permute(2, 0, 1).squeeze(1).T
            all_chapman_profiles.append(chapman_profiles.cpu().numpy())
            all_corrections.append(corrections.cpu().numpy())
    chapman_profiles = np.concatenate(all_chapman_profiles, axis=0)
    corrections = np.concatenate(all_corrections, axis=0)
    corrected_profiles = chapman_profiles + corrections
    # Compute metrics
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    mse = mean_squared_error(electron_density.flatten(), corrected_profiles.flatten())
    mae = mean_absolute_error(electron_density.flatten(), corrected_profiles.flatten())
    r2 = r2_score(electron_density.flatten(), corrected_profiles.flatten())
    print(f"Hybrid Model - Corrected Profile: MSE={mse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")
    mse_chapman = mean_squared_error(electron_density.flatten(), chapman_profiles.flatten())
    mae_chapman = mean_absolute_error(electron_density.flatten(), chapman_profiles.flatten())
    r2_chapman = r2_score(electron_density.flatten(), chapman_profiles.flatten())
    print(f"Chapman Only: MSE={mse_chapman:.4f}, MAE={mae_chapman:.4f}, R2={r2_chapman:.4f}")
    # Save predictions
    np.save(os.path.join(model_dir, 'corrected_profiles_test.npy'), corrected_profiles)
    np.save(os.path.join(model_dir, 'chapman_profiles_test.npy'), chapman_profiles)
    print("Predictions saved.")
    # Plot random profiles
    plot_random_profiles(altitude, electron_density, chapman_profiles, corrected_profiles, n=5, save_path=os.path.join(model_dir, 'profile_comparison.png'))
    print("Random profile comparison plot saved.")
    # Plot error histograms
    plot_error_histograms(electron_density, chapman_profiles, corrected_profiles, save_path=os.path.join(model_dir, 'error_histogram.png'))
    print("Error histogram plot saved.")
    # Plot true vs predicted
    plot_true_vs_pred(electron_density, corrected_profiles, save_path=os.path.join(model_dir, 'true_vs_pred.png'))
    print("True vs. predicted plot saved.")

if __name__ == "__main__":
    main() 
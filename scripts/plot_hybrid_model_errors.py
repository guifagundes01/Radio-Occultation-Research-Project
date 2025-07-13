import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from model.chapman_hybrid_v2 import ChapmanPredictor, ProfileCorrector, HybridDataset, generate_chapman_profiles
from torch.utils.data import DataLoader
import h5py
import os

MODEL_DIR = './data/fit_results/hybrid_model'
DATA_PATH = './data/filtered/electron_density_profiles_2023_with_fits.h5'
BATCH_SIZE = 32
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Data Loading (reuse from evaluate_chapman_hybrid_v2.py) ---
def load_test_data():
    with h5py.File(DATA_PATH, 'r') as f:
        latitude = f['latitude'][:]
        longitude = f['longitude'][:]
        local_time = f['local_time'][:]
        f107 = f['f107'][:]
        kp = f['kp'][:]
        dip = f['dip'][:]
        electron_density = f['electron_density'][:]
        altitude = f['altitude'][:]
        chapman_params = f['fit_results/chapman_params'][:]
        is_good_fit = f['fit_results/is_good_fit'][:]
        latitude = latitude[is_good_fit]
        longitude = longitude[is_good_fit]
        local_time = local_time[is_good_fit]
        f107 = f107[is_good_fit]
        kp = kp[is_good_fit]
        dip = dip[is_good_fit]
        electron_density = electron_density[is_good_fit]
        chapman_params = chapman_params[is_good_fit]
    # Load scalers
    X_scaler = np.load(f'{MODEL_DIR}/X_scaler.npy', allow_pickle=True).item()
    y_scaler = np.load(f'{MODEL_DIR}/y_scaler.npy', allow_pickle=True).item()
    # Process temporal features
    import pandas as pd
    def convert_seconds_to_datetime(seconds):
        local_time_date = pd.to_datetime(seconds, unit='s')
        return {
            'year': local_time_date.year,
            'month': local_time_date.month,
            'doy': local_time_date.dayofyear,
            'hour': local_time_date.hour,
            'minute': local_time_date.minute
        }
    def circular_encode(value, max_value):
        sin_val = np.sin(2 * np.pi * value / max_value)
        cos_val = np.cos(2 * np.pi * value / max_value)
        return sin_val, cos_val
    temporal_features = []
    for seconds in local_time:
        dt = convert_seconds_to_datetime(seconds)
        month_sin, month_cos = circular_encode(dt['month'], 12)
        doy_sin, doy_cos = circular_encode(dt['doy'], 365)
        hour_sin, hour_cos = circular_encode(dt['hour'], 24)
        temporal_features.append([
            month_sin, month_cos,
            doy_sin, doy_cos,
            hour_sin, hour_cos
        ])
    temporal_features = np.array(temporal_features)
    X = np.column_stack([
        latitude,
        longitude,
        temporal_features,
        f107,
        kp,
        dip
    ])
    # Scale
    X_scaled = X_scaler.transform(X)
    y_scaled = y_scaler.transform(chapman_params)
    # Split
    from sklearn.model_selection import train_test_split
    indices = np.arange(len(X_scaled))
    _, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    X_test = X_scaled[test_idx]
    y_test = y_scaled[test_idx]
    profiles_test = electron_density[test_idx]
    altitude = altitude
    test_dataset = HybridDataset(X_test, y_test, profiles_test, altitude)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    return test_loader, altitude, y_scaler

# --- Model Loading ---
def load_model(altitude):
    model_stage1 = ChapmanPredictor(input_size=11, hidden_sizes=[128, 64])
    model_stage2 = ProfileCorrector(profile_size=len(altitude), feature_size=11, hidden_sizes=[128, 64])
    checkpoint = torch.load(f'{MODEL_DIR}/best_hybrid_model.pth', map_location=DEVICE)
    model_stage1.load_state_dict(checkpoint['stage1_state_dict'])
    model_stage2.load_state_dict(checkpoint['stage2_state_dict'])
    model_stage1.to(DEVICE)
    model_stage2.to(DEVICE)
    model_stage1.eval()
    model_stage2.eval()
    return model_stage1, model_stage2

# --- Evaluation ---
def get_predictions(model_stage1, model_stage2, test_loader, altitude, y_scaler):
    all_true_profiles = []
    all_pred_profiles = []
    all_chapman_profiles = []
    all_chapman_true_profiles = []
    with torch.no_grad():
        for X_batch, y_batch, profiles_batch, altitude_batch in test_loader:
            X_batch = X_batch.to(DEVICE)
            profiles_batch = profiles_batch.to(DEVICE)
            altitude_batch = altitude_batch.to(DEVICE)
            # Inverse-transform predicted Chapman parameters
            chapman_params_scaled = model_stage1(X_batch).cpu().numpy()
            chapman_params_pred = y_scaler.inverse_transform(chapman_params_scaled)
            chapman_profiles = generate_chapman_profiles(torch.tensor(chapman_params_pred).float().to(DEVICE), altitude_batch)
            # True Chapman fit
            chapman_params_true = y_scaler.inverse_transform(y_batch.cpu().numpy())
            chapman_true_profiles = generate_chapman_profiles(torch.tensor(chapman_params_true).float().to(DEVICE), altitude_batch)
            # Corrections
            chapman_params_pred_tensor = torch.tensor(chapman_params_pred).float().to(DEVICE)
            corrections = model_stage2(X_batch, chapman_params_pred_tensor)
            profile_scale = torch.mean(torch.abs(chapman_profiles), dim=1, keepdim=True)
            max_correction_scale = 0.1
            corrections = corrections * profile_scale * max_correction_scale
            predicted_profiles = chapman_profiles + corrections
            all_true_profiles.append(profiles_batch.cpu().numpy())
            all_pred_profiles.append(predicted_profiles.cpu().numpy())
            all_chapman_profiles.append(chapman_profiles.cpu().numpy())
            all_chapman_true_profiles.append(chapman_true_profiles.cpu().numpy())
    all_true_profiles = np.concatenate(all_true_profiles, axis=0)
    all_pred_profiles = np.concatenate(all_pred_profiles, axis=0)
    all_chapman_profiles = np.concatenate(all_chapman_profiles, axis=0)
    all_chapman_true_profiles = np.concatenate(all_chapman_true_profiles, axis=0)
    return all_true_profiles, all_pred_profiles, all_chapman_profiles, all_chapman_true_profiles

# --- Error Metrics ---
def compute_errors(true_profiles, pred_profiles):
    # Per-profile errors
    mae = np.mean(np.abs(pred_profiles - true_profiles), axis=1)
    rmse = np.sqrt(np.mean((pred_profiles - true_profiles) ** 2, axis=1))
    rel_error = np.abs(pred_profiles - true_profiles) / (np.abs(true_profiles) + 1e-8)
    mean_rel_error = 100 * np.mean(rel_error, axis=1)  # percent
    return mae, rmse, mean_rel_error

# --- Plotting ---
def plot_error_histograms(mae_hybrid, rmse_hybrid, rel_hybrid, mae_chapman, rmse_chapman, rel_chapman, save_path=None):
    plt.figure(figsize=(18, 5))
    titles = ['Mean Absolute Error', 'Root Mean Squared Error', 'Mean Relative Error (%)']
    for i, (hyb, chap, title) in enumerate(zip([mae_hybrid, rmse_hybrid, rel_hybrid], [mae_chapman, rmse_chapman, rel_chapman], titles)):
        plt.subplot(1, 3, i+1)
        sns.histplot(chap, bins=40, color='blue', label='Chapman Only', stat='density', kde=True, alpha=0.5)
        sns.histplot(hyb, bins=40, color='green', label='Hybrid Model', stat='density', kde=True, alpha=0.5)
        plt.title(title)
        plt.xlabel(title)
        plt.ylabel('Density')
        plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

if __name__ == '__main__':
    test_loader, altitude, y_scaler = load_test_data()
    model_stage1, model_stage2 = load_model(altitude)
    all_true, all_pred, all_chapman, _ = get_predictions(model_stage1, model_stage2, test_loader, altitude, y_scaler)
    # Hybrid model errors
    mae_hybrid, rmse_hybrid, rel_hybrid = compute_errors(all_true, all_pred)
    # Chapman-only errors
    mae_chapman, rmse_chapman, rel_chapman = compute_errors(all_true, all_chapman)
    # Plot
    plot_error_histograms(mae_hybrid, rmse_hybrid, rel_hybrid, mae_chapman, rmse_chapman, rel_chapman, save_path=f'{MODEL_DIR}/error_histograms.png') 
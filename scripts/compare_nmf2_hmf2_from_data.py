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

# --- Get Profiles ---
def get_profiles(model_stage1, model_stage2, test_loader, altitude, y_scaler):
    all_true_profiles = []
    all_pred_profiles = []
    with torch.no_grad():
        for X_batch, y_batch, profiles_batch, altitude_batch in test_loader:
            X_batch = X_batch.to(DEVICE)
            profiles_batch = profiles_batch.to(DEVICE)
            altitude_batch = altitude_batch.to(DEVICE)
            # Inverse-transform predicted Chapman parameters
            chapman_params_scaled = model_stage1(X_batch).cpu().numpy()
            chapman_params_pred = y_scaler.inverse_transform(chapman_params_scaled)
            chapman_profiles = generate_chapman_profiles(torch.tensor(chapman_params_pred).float().to(DEVICE), altitude_batch)
            # Corrections
            chapman_params_pred_tensor = torch.tensor(chapman_params_pred).float().to(DEVICE)
            corrections = model_stage2(X_batch, chapman_params_pred_tensor)
            profile_scale = torch.mean(torch.abs(chapman_profiles), dim=1, keepdim=True)
            max_correction_scale = 0.1
            corrections = corrections * profile_scale * max_correction_scale
            predicted_profiles = chapman_profiles + corrections
            all_true_profiles.append(profiles_batch.cpu().numpy())
            all_pred_profiles.append(predicted_profiles.cpu().numpy())
    all_true_profiles = np.concatenate(all_true_profiles, axis=0)
    all_pred_profiles = np.concatenate(all_pred_profiles, axis=0)
    return all_true_profiles, all_pred_profiles

# --- Infer NmF2 and hmF2 from profiles ---
def infer_nmf2_hmf2(profiles, altitude):
    nmf2 = np.max(profiles, axis=1)
    hmf2 = altitude[np.argmax(profiles, axis=1)]
    return nmf2, hmf2

# --- Main ---
if __name__ == '__main__':
    test_loader, altitude, y_scaler = load_test_data()
    model_stage1, model_stage2 = load_model(altitude)
    all_true_profiles, all_pred_profiles = get_profiles(model_stage1, model_stage2, test_loader, altitude, y_scaler)
    # Infer NmF2 and hmF2
    nmf2_true, hmf2_true = infer_nmf2_hmf2(all_true_profiles, altitude)
    nmf2_pred, hmf2_pred = infer_nmf2_hmf2(all_pred_profiles, altitude)
    # Errors
    nmf2_abs_error = np.abs(nmf2_pred - nmf2_true)
    hmf2_abs_error = np.abs(hmf2_pred - hmf2_true)
    nmf2_pct_error = 100 * nmf2_abs_error / (np.abs(nmf2_true) + 1e-8)
    hmf2_pct_error = 100 * hmf2_abs_error / (np.abs(hmf2_true) + 1e-8)
    # Plot absolute error histograms
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(nmf2_abs_error, bins=40, kde=True)
    plt.title('NmF2 Absolute Error (Hybrid, from data)')
    plt.xlabel('NmF2 Error')
    plt.subplot(1, 2, 2)
    sns.histplot(hmf2_abs_error, bins=40, kde=True)
    plt.title('hmF2 Absolute Error (Hybrid, from data)')
    plt.xlabel('hmF2 Error')
    plt.tight_layout()
    plt.savefig(f'{MODEL_DIR}/nmf2_hmf2_error_histograms_from_data.png')
    plt.show()
    # Plot percentage error histograms
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(nmf2_pct_error, bins=40, kde=True)
    plt.title('NmF2 Percentage Error (Hybrid, from data)')
    plt.xlabel('NmF2 Error (%)')
    plt.subplot(1, 2, 2)
    sns.histplot(hmf2_pct_error, bins=40, kde=True)
    plt.title('hmF2 Percentage Error (Hybrid, from data)')
    plt.xlabel('hmF2 Error (%)')
    plt.tight_layout()
    plt.savefig(f'{MODEL_DIR}/nmf2_hmf2_pct_error_histograms_from_data.png')
    plt.show()
    # Scatter plot predicted vs. true
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(nmf2_true, nmf2_pred, alpha=0.5)
    plt.plot([nmf2_true.min(), nmf2_true.max()], [nmf2_true.min(), nmf2_true.max()], 'r--')
    plt.xlabel('True NmF2')
    plt.ylabel('Predicted NmF2')
    plt.title('NmF2: True vs. Predicted (from data)')
    plt.subplot(1, 2, 2)
    plt.scatter(hmf2_true, hmf2_pred, alpha=0.5)
    plt.plot([hmf2_true.min(), hmf2_true.max()], [hmf2_true.min(), hmf2_true.max()], 'r--')
    plt.xlabel('True hmF2')
    plt.ylabel('Predicted hmF2')
    plt.title('hmF2: True vs. Predicted (from data)')
    plt.tight_layout()
    plt.savefig(f'{MODEL_DIR}/nmf2_hmf2_scatter_from_data.png')
    plt.show() 
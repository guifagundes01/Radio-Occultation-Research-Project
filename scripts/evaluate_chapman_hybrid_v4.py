import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from model.chapman_hybrid_v4 import ChapmanPredictor, ProfileCorrector, HybridDataset
from model.chapman_function import generate_chapman_profiles
from torch.utils.data import DataLoader
import h5py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- Configuration ---
MODEL_DIR = './data/fit_results/hybrid_model_v4'
DATA_PATH = './data/filtered/electron_density_profiles_2023_with_fits.h5'
BATCH_SIZE = 32
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

# --- Load Data ---
def load_test_data():
    print("Loading data...")
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
        print(f"Number of profiles after filtering: {len(electron_density)}")
        print(f"Number of altitude points: {len(altitude)}")
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
    # Load scalers
    X_scaler = np.load(f'{MODEL_DIR}/X_scaler.npy', allow_pickle=True)
    y_scaler = np.load(f'{MODEL_DIR}/y_scaler.npy', allow_pickle=True)
    # If saved as dict, use .item()
    if hasattr(X_scaler, 'item'):
        X_scaler = X_scaler.item()
    if hasattr(y_scaler, 'item'):
        y_scaler = y_scaler.item()
    X_scaled = X_scaler.transform(X)
    y_scaled = y_scaler.transform(chapman_params)
    indices = np.arange(len(X_scaled))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    X_test = X_scaled[test_idx]
    y_test = y_scaled[test_idx]
    profiles_test = electron_density[test_idx]
    local_time_test = local_time[test_idx]
    f107_test = f107[test_idx]
    kp_test = kp[test_idx]
    dip_test = dip[test_idx]
    print(f"Test set size: {len(X_test)}")
    test_dataset = HybridDataset(X_test, y_test, profiles_test, altitude)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    return test_loader, altitude, local_time_test, f107_test, kp_test, dip_test, y_scaler

# --- Load Model ---
def load_model(altitude):
    print("Loading model...")
    model_stage1 = ChapmanPredictor(input_size=11, hidden_sizes=[128, 64])
    model_stage2 = ProfileCorrector(
        profile_size=len(altitude),
        feature_size=11,
        hidden_sizes=[128, 64]
    )
    # Load checkpoint from fine-tuning phase
    checkpoint = torch.load(f'{MODEL_DIR}/finetune_best.pth', map_location=DEVICE)
    model_stage1.load_state_dict(checkpoint['stage1_state_dict'])
    model_stage2.load_state_dict(checkpoint['stage2_state_dict'])
    model_stage1.to(DEVICE)
    model_stage2.to(DEVICE)
    model_stage1.eval()
    model_stage2.eval()
    print("Model loaded successfully!")
    return model_stage1, model_stage2

# --- Calculate Metrics ---
def calculate_metrics(all_true_profiles, all_pred_profiles, all_chapman_profiles):
    mse = np.mean((all_true_profiles - all_pred_profiles) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(all_true_profiles - all_pred_profiles))
    rel_error = np.abs(all_true_profiles - all_pred_profiles) / (np.abs(all_true_profiles) + 1e-8)
    mean_rel_error = np.mean(rel_error)
    median_rel_error = np.median(rel_error)
    chapman_rel_error = np.abs(all_true_profiles - all_chapman_profiles) / (np.abs(all_true_profiles) + 1e-8)
    chapman_mean_rel_error = np.mean(chapman_rel_error)
    improvement = (chapman_mean_rel_error - mean_rel_error) / chapman_mean_rel_error * 100
    metrics = {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'mean_rel_error': float(mean_rel_error),
        'median_rel_error': float(median_rel_error),
        'chapman_mean_rel_error': float(chapman_mean_rel_error),
        'improvement_over_chapman': float(improvement)
    }
    return metrics, rel_error

# --- Evaluation and Plotting ---
def evaluate_and_plot(model_stage1, model_stage2, test_loader, altitude, local_time, f107, kp, dip, y_scaler):
    all_true_profiles = []
    all_pred_profiles = []
    all_chapman_profiles = []
    all_altitudes = []
    with torch.no_grad():
        for X_batch, y_batch, profiles_batch, altitude_batch in test_loader:
            X_batch = X_batch.to(DEVICE)
            profiles_batch = profiles_batch.to(DEVICE)
            altitude_batch = altitude_batch.to(DEVICE)
            chapman_params = model_stage1(X_batch)
            chapman_profiles = generate_chapman_profiles(chapman_params, altitude_batch)
            corrections = model_stage2(X_batch, chapman_params)
            profile_scale = torch.mean(torch.abs(chapman_profiles), dim=1, keepdim=True)
            max_correction_scale = 0.1
            corrections = corrections * profile_scale * max_correction_scale
            predicted_profiles = chapman_profiles + corrections
            all_true_profiles.append(profiles_batch.cpu().numpy())
            all_pred_profiles.append(predicted_profiles.cpu().numpy())
            all_chapman_profiles.append(chapman_profiles.cpu().numpy())
            all_altitudes.append(altitude_batch.cpu().numpy())
    all_true_profiles = np.concatenate(all_true_profiles, axis=0)
    all_pred_profiles = np.concatenate(all_pred_profiles, axis=0)
    all_chapman_profiles = np.concatenate(all_chapman_profiles, axis=0)
    all_altitudes = np.concatenate(all_altitudes, axis=0)
    metrics, rel_error = calculate_metrics(all_true_profiles, all_pred_profiles, all_chapman_profiles)
    # Save metrics
    with open(f'{MODEL_DIR}/test_metrics_eval_script.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    print("\nTest Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4e}")
    # Plot error distribution
    plt.figure(figsize=(8, 5))
    plt.hist(rel_error.flatten(), bins=50, alpha=0.7)
    plt.xlabel('Relative Error')
    plt.ylabel('Frequency')
    plt.title('Error Distribution')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{MODEL_DIR}/error_distribution_eval_script.png')
    plt.close()
    # Plot mean profile comparison
    plt.figure(figsize=(8, 5))
    mean_true = np.mean(all_true_profiles, axis=0)
    mean_pred = np.mean(all_pred_profiles, axis=0)
    mean_chapman = np.mean(all_chapman_profiles, axis=0)
    std_true = np.std(all_true_profiles, axis=0)
    std_pred = np.std(all_pred_profiles, axis=0)
    plt.semilogy(all_altitudes[0], mean_true, 'b-', label='True Mean')
    plt.semilogy(all_altitudes[0], mean_pred, 'r--', label='Hybrid Mean')
    plt.semilogy(all_altitudes[0], mean_chapman, 'g-.', label='Chapman Mean')
    plt.fill_between(all_altitudes[0], mean_true - std_true, mean_true + std_true, alpha=0.2, color='blue')
    plt.fill_between(all_altitudes[0], mean_pred - std_pred, mean_pred + std_pred, alpha=0.2, color='red')
    plt.xlabel('Altitude (km)')
    plt.ylabel('Electron Density (cm⁻³)')
    plt.title('Mean Profile Comparison')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{MODEL_DIR}/mean_profile_comparison_eval_script.png')
    plt.close()

if __name__ == "__main__":
    test_loader, altitude, local_time, f107, kp, dip, y_scaler = load_test_data()
    model_stage1, model_stage2 = load_model(altitude)
    evaluate_and_plot(model_stage1, model_stage2, test_loader, altitude, local_time, f107, kp, dip, y_scaler) 
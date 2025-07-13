import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from model.chapman_hybrid_v3 import ChapmanPredictor, ProfileCorrector, HybridDataset
from model.chapman_function import generate_chapman_profiles
from torch.utils.data import DataLoader
import h5py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- Configuration ---
MODEL_DIR = './data/fit_results/hybrid_model_v3'
DATA_PATH = './data/filtered/electron_density_profiles_2023_with_fits.h5'
BATCH_SIZE = 32
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def convert_seconds_to_datetime(seconds):
    """Convert seconds since start of year to datetime components"""
    local_time_date = pd.to_datetime(seconds, unit='s')
    return {
        'year': local_time_date.year,
        'month': local_time_date.month,
        'doy': local_time_date.dayofyear,
        'hour': local_time_date.hour,
        'minute': local_time_date.minute
    }

def circular_encode(value, max_value):
    """Apply circular encoding to a value"""
    sin_val = np.sin(2 * np.pi * value / max_value)
    cos_val = np.cos(2 * np.pi * value / max_value)
    return sin_val, cos_val

# --- Load Data ---
def load_test_data():
    """Load and preprocess test data"""
    print("Loading data...")
    
    with h5py.File(DATA_PATH, 'r') as f:
        # Get base features
        latitude = f['latitude'][:]
        longitude = f['longitude'][:]
        local_time = f['local_time'][:]
        f107 = f['f107'][:]
        kp = f['kp'][:]
        dip = f['dip'][:]
        electron_density = f['electron_density'][:]
        altitude = f['altitude'][:]  # This is a single array of height points
        
        # Get Chapman parameters
        chapman_params = f['fit_results/chapman_params'][:]
        
        # Only use good fits
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
    
    # Process temporal features
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
    
    # Combine all features
    X = np.column_stack([
        latitude,
        longitude,
        temporal_features,
        f107,
        kp,
        dip
    ])
    
    # Load scalers
    X_scaler = np.load(f'{MODEL_DIR}/X_scaler.npy', allow_pickle=True).item()
    y_scaler = np.load(f'{MODEL_DIR}/y_scaler.npy', allow_pickle=True).item()
    
    # Scale the data
    X_scaled = X_scaler.transform(X)
    y_scaled = y_scaler.transform(chapman_params)
    
    # Split the data (same split as training)
    indices = np.arange(len(X_scaled))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    
    X_test = X_scaled[test_idx]
    y_test = y_scaled[test_idx]
    profiles_test = electron_density[test_idx]
    
    # Get original features for plotting
    local_time_test = local_time[test_idx]
    f107_test = f107[test_idx]
    kp_test = kp[test_idx]
    dip_test = dip[test_idx]
    
    print(f"Test set size: {len(X_test)}")
    
    # Create dataset and dataloader
    test_dataset = HybridDataset(X_test, y_test, profiles_test, altitude)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    return test_loader, altitude, local_time_test, f107_test, kp_test, dip_test, y_scaler

# --- Load Model ---
def load_model(altitude):
    """Load the trained hybrid model v3"""
    print("Loading model...")
    
    model_stage1 = ChapmanPredictor(input_size=11, hidden_sizes=[128, 64])
    model_stage2 = ProfileCorrector(
        profile_size=len(altitude),
        feature_size=11,
        hidden_sizes=[128, 64]
    )
    
    # Load checkpoint
    checkpoint = torch.load(f'{MODEL_DIR}/best_hybrid_model_v3.pth', map_location=DEVICE)
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
    """Calculate comprehensive error metrics"""
    
    # Basic error metrics
    mse = np.mean((all_true_profiles - all_pred_profiles) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(all_true_profiles - all_pred_profiles))
    
    # Relative error metrics
    rel_error = np.abs(all_true_profiles - all_pred_profiles) / (np.abs(all_true_profiles) + 1e-8)
    mean_rel_error = np.mean(rel_error)
    median_rel_error = np.median(rel_error)
    
    # Chapman vs Hybrid comparison
    chapman_rel_error = np.abs(all_true_profiles - all_chapman_profiles) / (np.abs(all_true_profiles) + 1e-8)
    chapman_mean_rel_error = np.mean(chapman_rel_error)
    
    # Improvement over Chapman
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

# --- Plot Best/Worst Profiles ---
def plot_best_worst_profiles(
    all_true_profiles, all_pred_profiles, all_chapman_profiles, all_altitudes,
    local_time, f107, kp, dip, n_samples=5
):
    """Plot best and worst performing profiles"""
    
    # Compute mean relative error for each profile
    rel_error = np.abs(all_pred_profiles - all_true_profiles) / (np.abs(all_true_profiles) + 1e-8)
    errors = 100 * np.mean(rel_error, axis=1)  # percent
    
    best_indices = np.argsort(errors)[:n_samples]
    worst_indices = np.argsort(errors)[-n_samples:][::-1]
    
    fig, axes = plt.subplots(2, n_samples, figsize=(4*n_samples, 8), sharey=True)
    
    for row, indices, label in zip([0, 1], [best_indices, worst_indices], ['Best', 'Worst']):
        for i, idx in enumerate(indices):
            ax = axes[row, i]
            
            # Real data
            ax.scatter(all_true_profiles[idx], all_altitudes[idx], 
                      label='Data', alpha=0.6, s=20, color='black')
            
            # Chapman profile (predicted)
            ax.plot(all_chapman_profiles[idx], all_altitudes[idx], 
                   'r-', label='Chapman (pred)', linewidth=2)
            
            # Model prediction
            ax.plot(all_pred_profiles[idx], all_altitudes[idx], 
                   'g--', label='Hybrid Model', linewidth=2)
            
            # Info
            dt = convert_seconds_to_datetime(local_time[idx])
            param_info = (f'F10.7: {f107[idx]:.1f}\nKp: {kp[idx]:.1f}\nDip: {dip[idx]:.1f}\n'
                         f'Month: {dt["month"]}\nDOY: {dt["doy"]}\nHour: {dt["hour"]}')
            ax.text(0.05, 0.95, param_info, transform=ax.transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax.set_title(f'{label} {i+1}\nError: {errors[idx]:.1f}%')
            ax.set_xlabel('Electron Density (cm⁻³)')
            if i == 0:
                ax.set_ylabel('Altitude (km)')
            ax.legend()
            ax.grid(True)
            ax.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig(f'{MODEL_DIR}/best_worst_profiles_v3.png', dpi=300, bbox_inches='tight')
    plt.close()

# --- Plot Error Distribution ---
def plot_error_distribution(rel_error, metrics):
    """Plot error distribution and statistics"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Error distribution
    axes[0, 0].hist(rel_error.flatten(), bins=50, alpha=0.7, color='blue')
    axes[0, 0].set_xlabel('Relative Error')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Error Distribution')
    axes[0, 0].grid(True)
    
    # Error vs altitude
    mean_error_by_altitude = np.mean(rel_error, axis=0)
    median_error_by_altitude = np.median(rel_error, axis=0)
    altitude_points = np.arange(len(mean_error_by_altitude))
    
    axes[0, 1].plot(altitude_points, mean_error_by_altitude, 'b-', label='Mean Error')
    axes[0, 1].plot(altitude_points, median_error_by_altitude, 'r--', label='Median Error')
    axes[0, 1].set_xlabel('Altitude Index')
    axes[0, 1].set_ylabel('Relative Error')
    axes[0, 1].set_title('Error vs Altitude')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Error statistics
    axes[1, 0].bar(['Mean', 'Median', 'Std'], 
                   [metrics['mean_rel_error'], metrics['median_rel_error'], np.std(rel_error.flatten())])
    axes[1, 0].set_ylabel('Relative Error')
    axes[1, 0].set_title('Error Statistics')
    axes[1, 0].grid(True)
    
    # Metrics summary
    metrics_text = f"MSE: {metrics['mse']:.2e}\n"
    metrics_text += f"RMSE: {metrics['rmse']:.2e}\n"
    metrics_text += f"MAE: {metrics['mae']:.2e}\n"
    metrics_text += f"Mean Rel Error: {metrics['mean_rel_error']:.2%}\n"
    metrics_text += f"Median Rel Error: {metrics['median_rel_error']:.2%}\n"
    metrics_text += f"Chapman Mean Rel Error: {metrics['chapman_mean_rel_error']:.2%}\n"
    metrics_text += f"Improvement: {metrics['improvement_over_chapman']:.1f}%"
    
    axes[1, 1].text(0.1, 0.5, metrics_text, transform=axes[1, 1].transAxes, 
                    fontsize=12, verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    axes[1, 1].set_title('Performance Metrics')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{MODEL_DIR}/error_analysis_v3.png', dpi=300, bbox_inches='tight')
    plt.close()

# --- Plot Mean Profile Comparison ---
def plot_mean_profile_comparison(all_true_profiles, all_pred_profiles, all_chapman_profiles, all_altitudes):
    """Plot mean profile comparison with standard deviations"""
    
    plt.figure(figsize=(12, 8))
    
    # Calculate statistics
    mean_true = np.mean(all_true_profiles, axis=0)
    mean_pred = np.mean(all_pred_profiles, axis=0)
    mean_chapman = np.mean(all_chapman_profiles, axis=0)
    
    std_true = np.std(all_true_profiles, axis=0)
    std_pred = np.std(all_pred_profiles, axis=0)
    std_chapman = np.std(all_chapman_profiles, axis=0)
    
    # Plot mean profiles
    plt.semilogy(all_altitudes[0], mean_true, 'b-', label='True Mean', linewidth=2)
    plt.semilogy(all_altitudes[0], mean_pred, 'g--', label='Hybrid Model Mean', linewidth=2)
    plt.semilogy(all_altitudes[0], mean_chapman, 'r:', label='Chapman Mean', linewidth=2)
    
    # Plot standard deviations
    plt.fill_between(all_altitudes[0], 
                     mean_true - std_true, 
                     mean_true + std_true, 
                     alpha=0.2, color='blue', label='True ±1σ')
    plt.fill_between(all_altitudes[0], 
                     mean_pred - std_pred, 
                     mean_pred + std_pred, 
                     alpha=0.2, color='green', label='Hybrid ±1σ')
    plt.fill_between(all_altitudes[0], 
                     mean_chapman - std_chapman, 
                     mean_chapman + std_chapman, 
                     alpha=0.2, color='red', label='Chapman ±1σ')
    
    plt.xlabel('Altitude (km)')
    plt.ylabel('Electron Density (cm⁻³)')
    plt.title('Mean Profile Comparison')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{MODEL_DIR}/mean_profile_comparison_v3.png', dpi=300, bbox_inches='tight')
    plt.close()

# --- Main Evaluation Function ---
def evaluate_and_plot(model_stage1, model_stage2, test_loader, altitude, 
                     local_time, f107, kp, dip, y_scaler):
    """Main evaluation function"""
    
    print("Evaluating model...")
    
    all_true_profiles = []
    all_pred_profiles = []
    all_chapman_profiles = []
    all_altitudes = []
    
    with torch.no_grad():
        for X_batch, y_batch, profiles_batch, altitude_batch in test_loader:
            X_batch = X_batch.to(DEVICE)
            profiles_batch = profiles_batch.to(DEVICE)
            altitude_batch = altitude_batch.to(DEVICE)
            
            # Get predictions
            chapman_params = model_stage1(X_batch)
            chapman_profiles = generate_chapman_profiles(chapman_params, altitude_batch)
            corrections = model_stage2(X_batch, chapman_params)
            
            # Scale corrections
            profile_scale = torch.mean(torch.abs(chapman_profiles), dim=1, keepdim=True)
            max_correction_scale = 0.1
            corrections = corrections * profile_scale * max_correction_scale
            
            # Get final predictions
            predicted_profiles = chapman_profiles + corrections
            
            # Store results
            all_true_profiles.append(profiles_batch.cpu().numpy())
            all_pred_profiles.append(predicted_profiles.cpu().numpy())
            all_chapman_profiles.append(chapman_profiles.cpu().numpy())
            all_altitudes.append(altitude_batch.cpu().numpy())
    
    # Convert to numpy arrays
    all_true_profiles = np.concatenate(all_true_profiles, axis=0)
    all_pred_profiles = np.concatenate(all_pred_profiles, axis=0)
    all_chapman_profiles = np.concatenate(all_chapman_profiles, axis=0)
    all_altitudes = np.concatenate(all_altitudes, axis=0)
    
    print(f"Evaluated {len(all_true_profiles)} profiles")
    
    # Calculate metrics
    metrics, rel_error = calculate_metrics(all_true_profiles, all_pred_profiles, all_chapman_profiles)
    
    # Print metrics
    print("\n=== Performance Metrics ===")
    print(f"MSE: {metrics['mse']:.2e}")
    print(f"RMSE: {metrics['rmse']:.2e}")
    print(f"MAE: {metrics['mae']:.2e}")
    print(f"Mean Relative Error: {metrics['mean_rel_error']:.2%}")
    print(f"Median Relative Error: {metrics['median_rel_error']:.2%}")
    print(f"Chapman Mean Relative Error: {metrics['chapman_mean_rel_error']:.2%}")
    print(f"Improvement over Chapman: {metrics['improvement_over_chapman']:.1f}%")
    
    # Create plots
    print("\nCreating plots...")
    
    # Plot best/worst profiles
    plot_best_worst_profiles(
        all_true_profiles, all_pred_profiles, all_chapman_profiles, all_altitudes,
        local_time, f107, kp, dip, n_samples=5
    )
    
    # Plot error distribution
    plot_error_distribution(rel_error, metrics)
    
    # Plot mean profile comparison
    plot_mean_profile_comparison(all_true_profiles, all_pred_profiles, all_chapman_profiles, all_altitudes)
    
    # Save metrics
    with open(f'{MODEL_DIR}/evaluation_metrics_v3.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"\nResults saved to {MODEL_DIR}")
    print("Evaluation completed!")

if __name__ == '__main__':
    # Load test data
    test_loader, altitude, local_time, f107, kp, dip, y_scaler = load_test_data()
    
    # Load model
    model_stage1, model_stage2 = load_model(altitude)
    
    # Evaluate and plot
    evaluate_and_plot(model_stage1, model_stage2, test_loader, altitude, 
                     local_time, f107, kp, dip, y_scaler) 
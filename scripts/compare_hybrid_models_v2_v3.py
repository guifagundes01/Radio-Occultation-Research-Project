import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import pandas as pd
from torch.utils.data import DataLoader
import h5py
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Import both model versions
from model.chapman_hybrid_v2 import ChapmanPredictor as ChapmanPredictorV2, ProfileCorrector as ProfileCorrectorV2, HybridDataset
from model.chapman_hybrid_v3 import ChapmanPredictor as ChapmanPredictorV3, ProfileCorrector as ProfileCorrectorV3
from model.chapman_function import generate_chapman_profiles

# --- Configuration ---
MODEL_DIR_V2 = './data/fit_results/hybrid_model'
MODEL_DIR_V3 = './data/fit_results/hybrid_model_v3'
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
        altitude = f['altitude'][:]
        
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
    
    # Load scalers (use V3 scalers for consistency)
    X_scaler = np.load(f'{MODEL_DIR_V3}/X_scaler.npy', allow_pickle=True).item()
    y_scaler = np.load(f'{MODEL_DIR_V3}/y_scaler.npy', allow_pickle=True).item()
    
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

# --- Load Models ---
def load_models(altitude):
    """Load both V2 and V3 models"""
    print("Loading models...")
    
    # Load V2 model
    model_stage1_v2 = ChapmanPredictorV2(input_size=11, hidden_sizes=[128, 64])
    model_stage2_v2 = ProfileCorrectorV2(profile_size=len(altitude), feature_size=11, hidden_sizes=[128, 64])
    
    checkpoint_v2 = torch.load(f'{MODEL_DIR_V2}/best_hybrid_model.pth', map_location=DEVICE)
    model_stage1_v2.load_state_dict(checkpoint_v2['stage1_state_dict'])
    model_stage2_v2.load_state_dict(checkpoint_v2['stage2_state_dict'])
    
    model_stage1_v2.to(DEVICE)
    model_stage2_v2.to(DEVICE)
    model_stage1_v2.eval()
    model_stage2_v2.eval()
    
    # Load V3 model
    model_stage1_v3 = ChapmanPredictorV3(input_size=11, hidden_sizes=[128, 64])
    model_stage2_v3 = ProfileCorrectorV3(profile_size=len(altitude), feature_size=11, hidden_sizes=[128, 64])
    
    checkpoint_v3 = torch.load(f'{MODEL_DIR_V3}/best_hybrid_model_v3.pth', map_location=DEVICE)
    model_stage1_v3.load_state_dict(checkpoint_v3['stage1_state_dict'])
    model_stage2_v3.load_state_dict(checkpoint_v3['stage2_state_dict'])
    
    model_stage1_v3.to(DEVICE)
    model_stage2_v3.to(DEVICE)
    model_stage1_v3.eval()
    model_stage2_v3.eval()
    
    print("Both models loaded successfully!")
    
    return (model_stage1_v2, model_stage2_v2), (model_stage1_v3, model_stage2_v3)

# --- Evaluate Model ---
def evaluate_model(model_stage1, model_stage2, test_loader, model_name):
    """Evaluate a single model"""
    print(f"Evaluating {model_name}...")
    
    all_true_profiles = []
    all_pred_profiles = []
    all_chapman_profiles = []
    
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
    
    # Convert to numpy arrays
    all_true_profiles = np.concatenate(all_true_profiles, axis=0)
    all_pred_profiles = np.concatenate(all_pred_profiles, axis=0)
    all_chapman_profiles = np.concatenate(all_chapman_profiles, axis=0)
    
    return all_true_profiles, all_pred_profiles, all_chapman_profiles

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

# --- Plot Comparison ---
def plot_model_comparison(metrics_v2, metrics_v3, rel_error_v2, rel_error_v3):
    """Create comprehensive comparison plots"""
    
    # Create comparison directory
    comparison_dir = './data/fit_results/model_comparison'
    os.makedirs(comparison_dir, exist_ok=True)
    
    # 1. Metrics comparison bar plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Relative error comparison
    metrics_names = ['Mean Rel Error', 'Median Rel Error', 'Chapman Rel Error']
    v2_values = [metrics_v2['mean_rel_error'], metrics_v2['median_rel_error'], metrics_v2['chapman_mean_rel_error']]
    v3_values = [metrics_v3['mean_rel_error'], metrics_v3['median_rel_error'], metrics_v3['chapman_mean_rel_error']]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, v2_values, width, label='V2', alpha=0.8)
    axes[0, 0].bar(x + width/2, v3_values, width, label='V3', alpha=0.8)
    axes[0, 0].set_xlabel('Metrics')
    axes[0, 0].set_ylabel('Relative Error')
    axes[0, 0].set_title('Relative Error Comparison')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(metrics_names)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Improvement over Chapman
    axes[0, 1].bar(['V2', 'V3'], 
                   [metrics_v2['improvement_over_chapman'], metrics_v3['improvement_over_chapman']],
                   color=['blue', 'green'], alpha=0.8)
    axes[0, 1].set_ylabel('Improvement (%)')
    axes[0, 1].set_title('Improvement over Chapman Function')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Error distribution comparison
    axes[1, 0].hist(rel_error_v2.flatten(), bins=50, alpha=0.7, label='V2', density=True)
    axes[1, 0].hist(rel_error_v3.flatten(), bins=50, alpha=0.7, label='V3', density=True)
    axes[1, 0].set_xlabel('Relative Error')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('Error Distribution Comparison')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Absolute error metrics
    abs_metrics = ['MSE', 'RMSE', 'MAE']
    v2_abs = [metrics_v2['mse'], metrics_v2['rmse'], metrics_v2['mae']]
    v3_abs = [metrics_v3['mse'], metrics_v3['rmse'], metrics_v3['mae']]
    
    axes[1, 1].bar(x - width/2, v2_abs, width, label='V2', alpha=0.8)
    axes[1, 1].bar(x + width/2, v3_abs, width, label='V3', alpha=0.8)
    axes[1, 1].set_xlabel('Metrics')
    axes[1, 1].set_ylabel('Absolute Error')
    axes[1, 1].set_title('Absolute Error Comparison')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(abs_metrics)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(f'{comparison_dir}/metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Error vs altitude comparison
    plt.figure(figsize=(12, 8))
    
    mean_error_v2_by_altitude = np.mean(rel_error_v2, axis=0)
    mean_error_v3_by_altitude = np.mean(rel_error_v3, axis=0)
    altitude_points = np.arange(len(mean_error_v2_by_altitude))
    
    plt.plot(altitude_points, mean_error_v2_by_altitude, 'b-', label='V2', linewidth=2)
    plt.plot(altitude_points, mean_error_v3_by_altitude, 'g-', label='V3', linewidth=2)
    plt.xlabel('Altitude Index')
    plt.ylabel('Mean Relative Error')
    plt.title('Error vs Altitude Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{comparison_dir}/error_vs_altitude.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Sample profile comparison
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Select random sample profiles
    sample_indices = np.random.choice(len(rel_error_v2), 3, replace=False)
    
    for i, idx in enumerate(sample_indices):
        # Get error for this profile
        error_v2 = np.mean(rel_error_v2[idx]) * 100
        error_v3 = np.mean(rel_error_v3[idx]) * 100
        
        # Plot V2
        axes[0, i].scatter(all_true_profiles_v2[idx], np.arange(len(all_true_profiles_v2[idx])), 
                          label='True', alpha=0.6, s=20, color='black')
        axes[0, i].plot(all_pred_profiles_v2[idx], np.arange(len(all_pred_profiles_v2[idx])), 
                       'b--', label=f'V2 (Error: {error_v2:.1f}%)', linewidth=2)
        axes[0, i].set_xlabel('Electron Density (cm⁻³)')
        axes[0, i].set_ylabel('Altitude Index')
        axes[0, i].set_title(f'Sample {i+1} - V2')
        axes[0, i].legend()
        axes[0, i].grid(True, alpha=0.3)
        axes[0, i].set_xscale('log')
        
        # Plot V3
        axes[1, i].scatter(all_true_profiles_v3[idx], np.arange(len(all_true_profiles_v3[idx])), 
                          label='True', alpha=0.6, s=20, color='black')
        axes[1, i].plot(all_pred_profiles_v3[idx], np.arange(len(all_pred_profiles_v3[idx])), 
                       'g--', label=f'V3 (Error: {error_v3:.1f}%)', linewidth=2)
        axes[1, i].set_xlabel('Electron Density (cm⁻³)')
        axes[1, i].set_ylabel('Altitude Index')
        axes[1, i].set_title(f'Sample {i+1} - V3')
        axes[1, i].legend()
        axes[1, i].grid(True, alpha=0.3)
        axes[1, i].set_xscale('log')
    
    plt.tight_layout()
    plt.savefig(f'{comparison_dir}/sample_profiles_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return comparison_dir

# --- Print Comparison Summary ---
def print_comparison_summary(metrics_v2, metrics_v3):
    """Print detailed comparison summary"""
    
    print("\n" + "="*60)
    print("HYBRID MODEL V2 vs V3 COMPARISON")
    print("="*60)
    
    print("\n--- PERFORMANCE METRICS ---")
    print(f"{'Metric':<25} {'V2':<15} {'V3':<15} {'Improvement':<15}")
    print("-" * 70)
    
    # Calculate improvements
    mse_improvement = (metrics_v2['mse'] - metrics_v3['mse']) / metrics_v2['mse'] * 100
    rmse_improvement = (metrics_v2['rmse'] - metrics_v3['rmse']) / metrics_v2['rmse'] * 100
    mae_improvement = (metrics_v2['mae'] - metrics_v3['mae']) / metrics_v2['mae'] * 100
    rel_error_improvement = (metrics_v2['mean_rel_error'] - metrics_v3['mean_rel_error']) / metrics_v2['mean_rel_error'] * 100
    
    print(f"{'MSE':<25} {metrics_v2['mse']:<15.2e} {metrics_v3['mse']:<15.2e} {mse_improvement:<15.1f}%")
    print(f"{'RMSE':<25} {metrics_v2['rmse']:<15.2e} {metrics_v3['rmse']:<15.2e} {rmse_improvement:<15.1f}%")
    print(f"{'MAE':<25} {metrics_v2['mae']:<15.2e} {metrics_v3['mae']:<15.2e} {mae_improvement:<15.1f}%")
    print(f"{'Mean Rel Error':<25} {metrics_v2['mean_rel_error']:<15.2%} {metrics_v3['mean_rel_error']:<15.2%} {rel_error_improvement:<15.1f}%")
    print(f"{'Median Rel Error':<25} {metrics_v2['median_rel_error']:<15.2%} {metrics_v3['median_rel_error']:<15.2%}")
    print(f"{'Chapman Rel Error':<25} {metrics_v2['chapman_mean_rel_error']:<15.2%} {metrics_v3['chapman_mean_rel_error']:<15.2%}")
    print(f"{'Improvement over Chapman':<25} {metrics_v2['improvement_over_chapman']:<15.1f}% {metrics_v3['improvement_over_chapman']:<15.1f}%")
    
    print("\n--- KEY DIFFERENCES ---")
    print("V2: Two-stage training with separate optimizers")
    print("V3: End-to-end training with single optimizer")
    print(f"V3 shows {rel_error_improvement:.1f}% improvement in mean relative error")
    print(f"V3 shows {mse_improvement:.1f}% improvement in MSE")
    
    # Save comparison to file
    comparison_data = {
        'v2_metrics': metrics_v2,
        'v3_metrics': metrics_v3,
        'improvements': {
            'mse_improvement': mse_improvement,
            'rmse_improvement': rmse_improvement,
            'mae_improvement': mae_improvement,
            'rel_error_improvement': rel_error_improvement
        }
    }
    
    comparison_dir = './data/fit_results/model_comparison'
    with open(f'{comparison_dir}/comparison_summary.json', 'w') as f:
        json.dump(comparison_data, f, indent=4)

# --- Main Function ---
def main():
    """Main comparison function"""
    print("Starting hybrid model V2 vs V3 comparison...")
    
    # Load test data
    test_loader, altitude, local_time, f107, kp, dip, y_scaler = load_test_data()
    
    # Load both models
    (model_stage1_v2, model_stage2_v2), (model_stage1_v3, model_stage2_v3) = load_models(altitude)
    
    # Evaluate V2 model
    global all_true_profiles_v2, all_pred_profiles_v2, all_chapman_profiles_v2
    all_true_profiles_v2, all_pred_profiles_v2, all_chapman_profiles_v2 = evaluate_model(
        model_stage1_v2, model_stage2_v2, test_loader, "V2"
    )
    
    # Evaluate V3 model
    global all_true_profiles_v3, all_pred_profiles_v3, all_chapman_profiles_v3
    all_true_profiles_v3, all_pred_profiles_v3, all_chapman_profiles_v3 = evaluate_model(
        model_stage1_v3, model_stage2_v3, test_loader, "V3"
    )
    
    # Calculate metrics for both models
    metrics_v2, rel_error_v2 = calculate_metrics(all_true_profiles_v2, all_pred_profiles_v2, all_chapman_profiles_v2)
    metrics_v3, rel_error_v3 = calculate_metrics(all_true_profiles_v3, all_pred_profiles_v3, all_chapman_profiles_v3)
    
    # Create comparison plots
    comparison_dir = plot_model_comparison(metrics_v2, metrics_v3, rel_error_v2, rel_error_v3)
    
    # Print comparison summary
    print_comparison_summary(metrics_v2, metrics_v3)
    
    print(f"\nComparison completed! Results saved to {comparison_dir}")

if __name__ == '__main__':
    main() 
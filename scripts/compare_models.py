import torch
import numpy as np
import matplotlib.pyplot as plt
from model.chapman_mlp import ChapmanMLP, convert_seconds_to_datetime, circular_encode
from sklearn.preprocessing import StandardScaler
import h5py
from analyze.functions import *
from analyze.fits import *
from scripts.evaluate_chapman_mlp import chapmanF2F1
import os

def calculate_density_error(y_pred, electron_density, altitude):
    """Calculate the error between predicted and true electron density profiles"""
    errors = np.zeros(len(y_pred))
    for i in range(len(y_pred)):
        # Calculate predicted density profile
        pred_density = chapmanF2F1(altitude, *y_pred[i])
        true_density = electron_density[i]
        
        # Calculate mean absolute percentage error
        error = np.abs(pred_density - true_density) / np.abs(true_density)
        # Handle division by zero
        error[~np.isfinite(error)] = np.abs(pred_density[~np.isfinite(error)] - true_density[~np.isfinite(error)])
        errors[i] = np.mean(error) * 100  # Convert to percentage
    return errors

def load_all_models():
    """Load all three models and their scalers"""
    models = {}
    scalers = {}
    
    # Model configurations
    model_configs = {
        'original': {
            'path': './data/fit_results/best_chapman_mlp.pth',
            'input_size': 6,  # Basic features: lat, lon, time, f107, kp, dip
            'scaler_path': './data/fit_results'
        },
        'filtered_negative': {
            'path': './data/fit_results/filter_negative/best_chapman_mlp.pth',
            'input_size': 8,  # Same as original: lat, lon, time, f107, kp, dip
            'scaler_path': './data/fit_results/filter_negative'
        },
        'circular_encoding': {
            'path': './data/fit_results/circular_encoding/best_chapman_mlp.pth',
            'input_size': 11,  # lat, lon, month_sin, month_cos, doy_sin, doy_cos, hour_sin, hour_cos, f107, kp, dip
            'scaler_path': './data/fit_results/circular_encoding'
        }
    }
    
    for model_name, config in model_configs.items():
        try:
            # Load model
            model = ChapmanMLP(input_size=config['input_size'])
            checkpoint = torch.load(config['path'])
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            models[model_name] = model
            
            # Load scalers
            X_scaler = np.load(f'{config["scaler_path"]}/X_scaler.npy', allow_pickle=True).item()
            y_scaler = np.load(f'{config["scaler_path"]}/y_scaler.npy', allow_pickle=True).item()
            scalers[model_name] = {'X': X_scaler, 'y': y_scaler}
            print(f"Successfully loaded {model_name} model and scalers")
        except Exception as e:
            print(f"Error loading {model_name} model: {str(e)}")
            continue
    
    return models, scalers

def plot_training_history_comparison():
    """Plot training history for all three models"""
    plt.figure(figsize=(12, 6))
    
    model_configs = {
        'original': './data/fit_results/best_chapman_mlp.pth',
        'filtered_negative': './data/fit_results/filter_negative/best_chapman_mlp.pth',
        'circular_encoding': './data/fit_results/circular_encoding/best_chapman_mlp.pth'
    }
    
    for model_name, path in model_configs.items():
        try:
            checkpoint = torch.load(path)
            # Handle different checkpoint structures
            if 'train_losses' in checkpoint:
                train_losses = checkpoint['train_losses']
                val_losses = checkpoint['val_losses']
            else:
                # For older checkpoints that might not have the full history
                train_losses = [checkpoint['train_loss']]
                val_losses = [checkpoint['val_loss']]
            
            plt.plot(train_losses, label=f'{model_name} (Train)', linestyle='-', alpha=0.7)
            plt.plot(val_losses, label=f'{model_name} (Val)', linestyle='--', alpha=0.7)
        except Exception as e:
            print(f"Error plotting training history for {model_name}: {str(e)}")
            continue
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig('./data/fit_results/model_comparison/training_history_comparison.png')
    plt.close()

def prepare_input_features(local_time, latitude, longitude, f107, kp, dip):
    """Prepare input features for all models"""
    features = {}
    
    # Basic features for original model
    basic_features = np.column_stack([latitude, longitude, local_time, f107, kp, dip])
    features['original'] = basic_features
    
    # Process temporal features for filter_negative model (no circular encoding)
    temporal_features = []
    for seconds in local_time:
        dt = convert_seconds_to_datetime(seconds)
        temporal_features.append([
            dt['month'],
            dt['doy'],
            dt['hour']
        ])
    
    temporal_features = np.array(temporal_features)
    
    # Non-circular encoding model uses processed temporal features
    features['filtered_negative'] = np.column_stack([
        latitude, longitude, temporal_features, f107, kp, dip
    ])
    
    # Process temporal features for circular_encoding model
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
    
    # Circular encoding model uses processed temporal features
    features['circular_encoding'] = np.column_stack([
        latitude, longitude, temporal_features, f107, kp, dip
    ])
    
    return features

def plot_profile_comparison(models, scalers, electron_density, altitude, features, indices, n_samples=3, title_prefix=''):
    """Plot electron density profiles for all models side by side"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create figure with subplots for each sample
    fig, axes = plt.subplots(n_samples, 1, figsize=(10, 5*n_samples))
    if n_samples == 1:
        axes = [axes]
    
    for i, idx in enumerate(indices):
        ax = axes[i]
        
        # Plot data points
        ax.scatter(electron_density[idx], altitude, label='Data', alpha=0.6, s=20, color='gray')
        
        # Get predictions from each model
        for model_name, model in models.items():
            try:
                model = model.to(device)
                # Get the correct features for this model
                X = torch.FloatTensor(scalers[model_name]['X'].transform(features[model_name][idx:idx+1])).to(device)
                
                with torch.no_grad():
                    y_pred_scaled = model(X)
                    y_pred = scalers[model_name]['y'].inverse_transform(y_pred_scaled.cpu().numpy())
                
                # Plot predicted profile
                y_pred_curve = chapmanF2F1(altitude, *y_pred[0])
                ax.plot(y_pred_curve, altitude, label=f'{model_name}', linestyle='--')
            except Exception as e:
                print(f"Error making prediction with {model_name} model: {str(e)}")
                continue
        
        # Add parameter information
        dt = convert_seconds_to_datetime(features['original'][idx, 2])  # local_time is at index 2
        param_info = (f'F10.7: {features["original"][idx, 3]:.1f}\n'
                     f'Kp: {features["original"][idx, 4]:.1f}\n'
                     f'Dip: {features["original"][idx, 5]:.1f}\n'
                     f'Month: {dt["month"]}\n'
                     f'DOY: {dt["doy"]}\n'
                     f'Hour: {dt["hour"]}')
        
        ax.text(0.05, 0.95, param_info, transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_title(f'{title_prefix}Sample {i+1}')
        ax.set_xlabel('Electron Density (cm⁻³)')
        ax.set_ylabel('Altitude (km)')
        ax.set_xscale('log')  # Set x-axis to logarithmic scale
        ax.legend()
        ax.grid(True, which="both", ls="-", alpha=0.2)  # Add grid for both major and minor ticks
    
    plt.tight_layout()
    plt.savefig(f'./data/fit_results/model_comparison/profile_comparison_{title_prefix.lower().replace(" ", "_")}.png')
    plt.close()

def plot_error_distribution_comparison(models, scalers, electron_density, altitude, features):
    """Plot error distribution comparison for all models"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Calculate errors for each model
    errors = {}
    for model_name, model in models.items():
        try:
            model = model.to(device)
            # Get the correct features for this model
            X = torch.FloatTensor(scalers[model_name]['X'].transform(features[model_name])).to(device)
            
            with torch.no_grad():
                y_pred_scaled = model(X)
                y_pred = scalers[model_name]['y'].inverse_transform(y_pred_scaled.cpu().numpy())
            
            # Calculate errors
            errors[model_name] = calculate_density_error(y_pred, electron_density, altitude)
        except Exception as e:
            print(f"Error calculating errors for {model_name} model: {str(e)}")
            continue
    
    if not errors:
        print("No models produced valid predictions. Cannot plot error distributions.")
        return
    
    # Plot error distributions (only errors > 5%)
    plt.figure(figsize=(10, 6))
    for model_name, model_errors in errors.items():
        # Filter errors > 5%
        filtered_errors = model_errors[model_errors > 5]
        if len(filtered_errors) > 0:
            plt.hist(filtered_errors, bins=50, alpha=0.5, label=model_name)
    
    plt.xlabel('Error (%)')
    plt.ylabel('Frequency')
    plt.title('Error Distribution Comparison (Errors > 5%)')
    plt.legend()
    plt.grid(True)
    plt.savefig('./data/fit_results/model_comparison/error_distribution_comparison.png')
    plt.close()
    
    # Print error statistics
    print("\nError Statistics Comparison:")
    print("-" * 50)
    for model_name, model_errors in errors.items():
        print(f"\n{model_name}:")
        print(f"Mean Error: {np.mean(model_errors):.2f}%")
        print(f"Median Error: {np.median(model_errors):.2f}%")
        print(f"Std Error: {np.std(model_errors):.2f}%")
        print(f"Errors > 5%: {np.mean(model_errors > 5)*100:.2f}% of samples")

def find_best_worst_profiles(models, scalers, electron_density, altitude, features):
    """Find indices of best and worst profiles for each model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Calculate errors for each model
    all_errors = {}
    for model_name, model in models.items():
        try:
            model = model.to(device)
            X = torch.FloatTensor(scalers[model_name]['X'].transform(features[model_name])).to(device)
            
            with torch.no_grad():
                y_pred_scaled = model(X)
                y_pred = scalers[model_name]['y'].inverse_transform(y_pred_scaled.cpu().numpy())
            
            all_errors[model_name] = calculate_density_error(y_pred, electron_density, altitude)
        except Exception as e:
            print(f"Error calculating errors for {model_name} model: {str(e)}")
            continue
    
    if not all_errors:
        return None, None
    
    # Calculate average error across all models for each profile
    avg_errors = np.zeros(len(electron_density))
    for errors in all_errors.values():
        avg_errors += errors
    avg_errors /= len(all_errors)
    
    # Get indices of best and worst profiles
    best_indices = np.argsort(avg_errors)[:5]
    worst_indices = np.argsort(avg_errors)[-5:][::-1]
    
    return best_indices, worst_indices

def main():
    # Create output directory
    os.makedirs('./data/fit_results/model_comparison', exist_ok=True)
    
    # Load all models and scalers
    models, scalers = load_all_models()
    
    if not models:
        print("No models were successfully loaded. Exiting.")
        return
    
    # Load data
    electron_density, latitude, longitude, altitude, local_time, f107, kp, dip = load_data('./data/filtered/electron_density_profiles_2023_with_fits.h5')
    
    # Prepare input features for all models
    features = prepare_input_features(local_time, latitude, longitude, f107, kp, dip)
    
    # Plot training history comparison
    plot_training_history_comparison()
    
    # Find best and worst profiles
    best_indices, worst_indices = find_best_worst_profiles(models, scalers, electron_density, altitude, features)
    
    if best_indices is not None and worst_indices is not None:
        # Plot best profiles
        plot_profile_comparison(models, scalers, electron_density, altitude, features, best_indices, 
                              n_samples=5, title_prefix='Best ')
        
        # Plot worst profiles
        plot_profile_comparison(models, scalers, electron_density, altitude, features, worst_indices, 
                              n_samples=5, title_prefix='Worst ')
    
    # Plot error distribution comparison
    plot_error_distribution_comparison(models, scalers, electron_density, altitude, features)
    
    print("Comparison completed! All plots have been saved to data/fit_results/model_comparison/")

if __name__ == "__main__":
    main() 
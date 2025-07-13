import torch
import numpy as np
import matplotlib.pyplot as plt
from model.chapman_mlp_no_circular import ChapmanMLP, convert_seconds_to_datetime
from sklearn.preprocessing import StandardScaler
import h5py
from analyze.functions import *

def load_model_and_scalers(model_path):
    # Load the best model
    model = ChapmanMLP(input_size=8)  # Updated input size for non-circular encoding
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load scalers
    X_scaler = np.load('./data/fit_results/filter_negative/X_scaler.npy', allow_pickle=True).item()
    y_scaler = np.load('./data/fit_results/filter_negative/y_scaler.npy', allow_pickle=True).item()
    
    return model, X_scaler, y_scaler

def chapmanF2F1(x, Nmax, hmax, H, a1, Nm, a2, hm):
    """Chapman function for F2 and F1 layers"""
    z = (x - hmax) / H
    zf1 = (x - hm) / H
    F2 = Nmax * np.exp(1 - z - np.exp(-z))
    F1 = Nm * np.exp(1 - zf1 - np.exp(-zf1))
    return F2 + a1 * F1

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

def plot_parameter_correlations(errors, latitude, longitude, local_time, f107, kp, dip):
    """Plot how errors correlate with different parameters"""
    # Convert local_time back to hours for better visualization
    hours = np.array([convert_seconds_to_datetime(lt)['hour'] for lt in local_time])
    
    parameters = {
        'Latitude': latitude,
        'Longitude': longitude,
        'Local Hour': hours,
        'F10.7': f107,
        'Kp': kp,
        'Dip': dip
    }
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (name, values) in enumerate(parameters.items()):
        ax = axes[i]
        scatter = ax.scatter(values, errors, alpha=0.5, c=errors, cmap='viridis')
        ax.set_xlabel(name)
        ax.set_ylabel('Error (%)')
        ax.set_title(f'Error vs {name}')
        ax.grid(True)
        
        # Add trend line
        z = np.polyfit(values, errors, 1)
        p = np.poly1d(z)
        ax.plot(values, p(values), "r--", alpha=0.8)
        
        # Calculate correlation coefficient
        corr = np.corrcoef(values, errors)[0,1]
        ax.text(0.05, 0.95, f'Correlation: {corr:.2f}', 
                transform=ax.transAxes, verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig('./data/fit_results/filter_negative/error_correlations.png')
    plt.close()

def plot_predictions(model, X_scaler, y_scaler, electron_density, altitude, latitude, longitude, local_time, f107, kp, dip, n_samples=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Process temporal features
    temporal_features = []
    for seconds in local_time:
        dt = convert_seconds_to_datetime(seconds)
        temporal_features.append([
            dt['month'],
            dt['doy'],
            dt['hour']
        ])
    
    temporal_features = np.array(temporal_features)
    
    # Prepare all input features
    X = np.column_stack([
        latitude,
        longitude,
        temporal_features,
        f107,
        kp,
        dip
    ])
    
    X_scaled = X_scaler.transform(X)
    X_tensor = torch.FloatTensor(X_scaled).to(device)
    
    # Get predictions for all data
    with torch.no_grad():
        y_pred_scaled = model(X_tensor)
        y_pred = y_scaler.inverse_transform(y_pred_scaled.cpu().numpy())
    
    # Load true parameters
    with h5py.File("./data/filtered/electron_density_profiles_2023_with_fits.h5", 'r') as f:
        true_params = f['fit_results/chapman_params'][:]
    
    # Calculate errors based on actual electron density profiles
    errors = calculate_density_error(y_pred, electron_density, altitude)
    
    # Plot parameter correlations
    plot_parameter_correlations(errors, latitude, longitude, local_time, f107, kp, dip)
    
    # Get indices of best and worst predictions
    best_indices = np.argsort(errors)[:n_samples]
    worst_indices = np.argsort(errors)[-n_samples:][::-1]  # Reverse to get worst first
    
    # Create figure with two rows: best and worst predictions
    fig, axes = plt.subplots(2, n_samples, figsize=(4*n_samples, 8))
    
    # Plot best predictions
    for i, idx in enumerate(best_indices):
        ax = axes[0, i]
        # Plot data points
        ax.scatter(electron_density[idx], altitude, label='Data', alpha=0.6, s=20)
        
        # Plot true Chapman fit (used in training)
        y_true = chapmanF2F1(altitude, *true_params[idx])
        ax.plot(y_true, altitude, 'r-', label='True Chapman')
        
        # Plot predicted fit
        y_pred_curve = chapmanF2F1(altitude, *y_pred[idx])
        ax.plot(y_pred_curve, altitude, 'g--', label='Predicted Chapman')
        
        # Add parameter information
        dt = convert_seconds_to_datetime(local_time[idx])
        param_info = (f'F10.7: {f107[idx]:.1f}\nKp: {kp[idx]:.1f}\nDip: {dip[idx]:.1f}\n'
                     f'Month: {dt["month"]}\nDOY: {dt["doy"]}\nHour: {dt["hour"]}')
        ax.text(0.05, 0.95, param_info, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_title(f'Best {i+1}\nError: {errors[idx]:.1f}%')
        ax.set_xlabel('Electron Density (cm⁻³)')
        ax.set_ylabel('Altitude (km)')
        ax.legend()
        ax.grid(True)
    
    # Plot worst predictions
    for i, idx in enumerate(worst_indices):
        ax = axes[1, i]
        # Plot data points
        ax.scatter(electron_density[idx], altitude, label='Data', alpha=0.6, s=20)
        
        # Plot true Chapman fit (used in training)
        y_true = chapmanF2F1(altitude, *true_params[idx])
        ax.plot(y_true, altitude, 'r-', label='True Chapman')
        
        # Plot predicted fit
        y_pred_curve = chapmanF2F1(altitude, *y_pred[idx])
        ax.plot(y_pred_curve, altitude, 'g--', label='Predicted Chapman')
        
        # Add parameter information
        dt = convert_seconds_to_datetime(local_time[idx])
        param_info = (f'F10.7: {f107[idx]:.1f}\nKp: {kp[idx]:.1f}\nDip: {dip[idx]:.1f}\n'
                     f'Month: {dt["month"]}\nDOY: {dt["doy"]}\nHour: {dt["hour"]}')
        ax.text(0.05, 0.95, param_info, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_title(f'Worst {i+1}\nError: {errors[idx]:.1f}%')
        ax.set_xlabel('Electron Density (cm⁻³)')
        ax.set_ylabel('Altitude (km)')
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('./data/fit_results/filter_negative/mlp_predictions_no_circular.png')
    plt.close()

def plot_training_history(model_path):
    """Plot the training and validation loss history"""
    checkpoint = torch.load(model_path)
    train_losses = checkpoint['train_losses']
    val_losses = checkpoint['val_losses']
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss History')
    plt.legend()
    plt.grid(True)
    plt.savefig('./data/fit_results/filter_negative/training_history_no_circular.png')
    plt.close()

def plot_prediction_statistics(model, X_scaler, y_scaler, electron_density, altitude, latitude, longitude, local_time, f107, kp, dip):
    """Plot statistics about predictions across all data"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Process temporal features
    temporal_features = []
    for seconds in local_time:
        dt = convert_seconds_to_datetime(seconds)
        temporal_features.append([
            dt['month'],
            dt['doy'],
            dt['hour']
        ])
    
    temporal_features = np.array(temporal_features)
    
    # Prepare all input features
    X = np.column_stack([
        latitude,
        longitude,
        temporal_features,
        f107,
        kp,
        dip
    ])
    
    X_scaled = X_scaler.transform(X)
    X_tensor = torch.FloatTensor(X_scaled).to(device)
    
    # Get predictions for all data
    with torch.no_grad():
        y_pred_scaled = model(X_tensor)
        y_pred = y_scaler.inverse_transform(y_pred_scaled.cpu().numpy())
    
    # Load true parameters
    with h5py.File("./data/filtered/electron_density_profiles_2023_with_fits.h5", 'r') as f:
        true_params = f['fit_results/chapman_params'][:]
    
    # Calculate errors for each parameter
    param_names = ['Nmax', 'hmax', 'H', 'a1', 'Nm', 'a2', 'hm']
    
    # Handle division by zero and infinite values
    errors_percent = np.zeros_like(y_pred)
    for i in range(len(param_names)):
        # Only calculate percentage error where true_params is not zero
        mask = true_params[:, i] != 0
        errors_percent[mask, i] = np.abs(y_pred[mask, i] - true_params[mask, i]) / np.abs(true_params[mask, i]) * 100
        
        # For zero true values, use absolute error instead
        errors_percent[~mask, i] = np.abs(y_pred[~mask, i] - true_params[~mask, i])
    
    # Plot error distributions for each parameter
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for i, (name, error) in enumerate(zip(param_names, errors_percent.T)):
        ax = axes[i]
        # Filter out infinite values
        error = error[np.isfinite(error)]
        if len(error) > 0:  # Only plot if there are valid values
            ax.hist(error, bins=50, alpha=0.7)
            ax.set_title(f'{name} Error Distribution')
            ax.set_xlabel('Absolute Error (%)')
            ax.set_ylabel('Frequency')
            ax.grid(True)
    
    # Remove the last unused subplot
    axes[-1].remove()
    
    plt.tight_layout()
    plt.savefig('./data/fit_results/filter_negative/prediction_statistics_no_circular.png')
    plt.close()
    
    # Calculate and print overall statistics
    mean_errors = np.mean(errors_percent, axis=0)
    median_errors = np.median(errors_percent, axis=0)
    print("\nPrediction Error Statistics (No Circular Encoding Model):")
    print("-" * 50)
    for name, mean_err, median_err in zip(param_names, mean_errors, median_errors):
        print(f"{name}:")
        print(f"  Mean Error: {mean_err:.2e}")
        print(f"  Median Error: {median_err:.2e}")
        print()

def main():
    # Define model path
    model_path = './data/fit_results/filter_negative/best_chapman_mlp.pth'
    
    # Load data from filtered file
    electron_density, latitude, longitude, altitude, local_time, f107, kp, dip = load_data('./data/filtered/electron_density_profiles_2023_with_fits.h5')
    
    # Load model and scalers
    model, X_scaler, y_scaler = load_model_and_scalers(model_path)
    
    # Plot training history
    plot_training_history(model_path)
    
    # Plot predictions for random samples
    plot_predictions(model, X_scaler, y_scaler, electron_density, altitude, latitude, longitude, local_time, f107, kp, dip, n_samples=5)
    
    # Plot prediction statistics
    plot_prediction_statistics(model, X_scaler, y_scaler, electron_density, altitude, latitude, longitude, local_time, f107, kp, dip)
    
    print("Evaluation completed! All plots have been saved to data/fit_results/filter_negative/")

if __name__ == "__main__":
    main() 
import torch
import numpy as np
import matplotlib.pyplot as plt
import h5py
from model.chapman_mlp import ChapmanMLP, convert_seconds_to_datetime, circular_encode
from model.chapman_hybrid_v2 import ChapmanPredictor, ProfileCorrector, generate_chapman_profiles
from sklearn.model_selection import train_test_split

# --- Paths ---
MLP_MODEL_PATH = './data/fit_results/circular_encoding/best_chapman_mlp.pth'
MLP_X_SCALER_PATH = './data/fit_results/circular_encoding/X_scaler.npy'
MLP_Y_SCALER_PATH = './data/fit_results/circular_encoding/y_scaler.npy'
HYBRID_MODEL_DIR = './data/fit_results/hybrid_model'
HYBRID_X_SCALER_PATH = f'{HYBRID_MODEL_DIR}/X_scaler.npy'
HYBRID_Y_SCALER_PATH = f'{HYBRID_MODEL_DIR}/y_scaler.npy'
HYBRID_MODEL_PATH = f'{HYBRID_MODEL_DIR}/best_hybrid_model.pth'
DATA_PATH = './data/filtered/electron_density_profiles_2023_with_fits.h5'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Load Data ---
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
    # indices = np.arange(len(latitude))
    # _, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    return {
        'latitude': latitude,
        'longitude': longitude,
        'local_time': local_time,
        'f107': f107,
        'kp': kp,
        'dip': dip,
        'electron_density': electron_density,
        'altitude': altitude,
        'chapman_params': chapman_params
    }

# --- Load MLP Model ---
def load_mlp():
    model = ChapmanMLP(input_size=11)
    checkpoint = torch.load(MLP_MODEL_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    X_scaler = np.load(MLP_X_SCALER_PATH, allow_pickle=True).item()
    y_scaler = np.load(MLP_Y_SCALER_PATH, allow_pickle=True).item()
    return model, X_scaler, y_scaler

# --- Load Hybrid Model ---
def load_hybrid(altitude):
    model_stage1 = ChapmanPredictor(input_size=11, hidden_sizes=[128, 64])
    model_stage2 = ProfileCorrector(profile_size=len(altitude), feature_size=11, hidden_sizes=[128, 64])
    checkpoint = torch.load(HYBRID_MODEL_PATH, map_location=DEVICE)
    model_stage1.load_state_dict(checkpoint['stage1_state_dict'])
    model_stage2.load_state_dict(checkpoint['stage2_state_dict'])
    model_stage1.to(DEVICE)
    model_stage2.to(DEVICE)
    model_stage1.eval()
    model_stage2.eval()
    X_scaler = np.load(HYBRID_X_SCALER_PATH, allow_pickle=True).item()
    y_scaler = np.load(HYBRID_Y_SCALER_PATH, allow_pickle=True).item()
    return model_stage1, model_stage2, X_scaler, y_scaler

# --- Feature Engineering (shared) ---
def make_features(lat, lon, local_time, f107, kp, dip):
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
        lat,
        lon,
        temporal_features,
        f107,
        kp,
        dip
    ])
    return X

# --- Main Comparison ---
def main():
    # Load data
    data = load_test_data()
    lat, lon, local_time, f107, kp, dip = data['latitude'], data['longitude'], data['local_time'], data['f107'], data['kp'], data['dip']
    electron_density, altitude, chapman_params = data['electron_density'], data['altitude'], data['chapman_params']

    # --- MLP predictions ---
    mlp, mlp_X_scaler, mlp_y_scaler = load_mlp()
    X_mlp = make_features(lat, lon, local_time, f107, kp, dip)
    X_mlp_scaled = mlp_X_scaler.transform(X_mlp)
    with torch.no_grad():
        y_pred_scaled = mlp(torch.FloatTensor(X_mlp_scaled))
        y_pred = mlp_y_scaler.inverse_transform(y_pred_scaled.cpu().numpy())
    # Chapman profile from MLP (use generate_chapman_profiles)
    mlp_profiles = generate_chapman_profiles(
        torch.tensor(y_pred).float().to(DEVICE),
        torch.FloatTensor(altitude).to(DEVICE).unsqueeze(0).repeat(len(y_pred), 1)
    ).cpu().numpy()

    # --- Hybrid predictions ---
    model_stage1, model_stage2, hybrid_X_scaler, hybrid_y_scaler = load_hybrid(altitude)
    X_hybrid = make_features(lat, lon, local_time, f107, kp, dip)
    X_hybrid_scaled = hybrid_X_scaler.transform(X_hybrid)
    X_hybrid_tensor = torch.FloatTensor(X_hybrid_scaled).to(DEVICE)
    with torch.no_grad():
        chapman_params_pred_scaled = model_stage1(X_hybrid_tensor).cpu().numpy()
        chapman_params_pred = hybrid_y_scaler.inverse_transform(chapman_params_pred_scaled)
        chapman_profiles = generate_chapman_profiles(torch.tensor(chapman_params_pred).float().to(DEVICE), torch.FloatTensor(altitude).to(DEVICE).unsqueeze(0).repeat(len(X_hybrid_tensor), 1))
        corrections = model_stage2(X_hybrid_tensor, torch.tensor(chapman_params_pred).float().to(DEVICE))
        profile_scale = torch.mean(torch.abs(chapman_profiles), dim=1, keepdim=True)
        max_correction_scale = 0.1
        corrections = corrections * profile_scale * max_correction_scale
        hybrid_profiles = (chapman_profiles + corrections).cpu().numpy()
        chapman_profiles = chapman_profiles.cpu().numpy()

    # --- Error calculation ---
    rel_error_mlp = np.abs(mlp_profiles - electron_density) / (np.abs(electron_density) + 1e-8)
    rel_error_hybrid = np.abs(hybrid_profiles - electron_density) / (np.abs(electron_density) + 1e-8)
    mean_error_mlp = np.mean(rel_error_mlp, axis=1) * 100
    mean_error_hybrid = np.mean(rel_error_hybrid, axis=1) * 100

    print(f"MLP Mean Error: {np.mean(mean_error_mlp):.2f}% | Median: {np.median(mean_error_mlp):.2f}%")
    print(f"Hybrid Mean Error: {np.mean(mean_error_hybrid):.2f}% | Median: {np.median(mean_error_hybrid):.2f}%")

    # --- Plot comparison for 5 random samples ---
    n_samples = 5
    idxs = np.random.choice(len(electron_density), n_samples, replace=False)
    fig, axes = plt.subplots(1, n_samples, figsize=(4*n_samples, 6), sharey=True)
    for i, idx in enumerate(idxs):
        ax = axes[i]
        ax.scatter(electron_density[idx], altitude, label='Data', alpha=0.6, s=20)
        ax.semilogx(mlp_profiles[idx], altitude, 'b-', label='MLP')
        ax.semilogx(chapman_profiles[idx], altitude, 'r-', label='Chapman (Hybrid)')
        ax.semilogx(hybrid_profiles[idx], altitude, 'g--', label='Hybrid Model')
        dt = convert_seconds_to_datetime(local_time[idx])
        param_info = (f'F10.7: {f107[idx]:.1f}\nKp: {kp[idx]:.1f}\nDip: {dip[idx]:.1f}\n'
                      f'Month: {dt["month"]}\nDOY: {dt["doy"]}\nHour: {dt["hour"]}')
        ax.text(0.05, 0.95, param_info, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax.set_title(f'Sample {i+1}\nMLP Err: {mean_error_mlp[idx]:.1f}%\nHybrid Err: {mean_error_hybrid[idx]:.1f}%')
        ax.set_xlabel('Electron Density (cm⁻³)')
        if i == 0:
            ax.set_ylabel('Altitude (km)')
        ax.legend()
        ax.grid(True)
    plt.tight_layout()
    plt.savefig('./data/fit_results/model_comparison_samples.png')
    plt.show()

    # --- Error histograms ---
    plt.figure(figsize=(8, 5))
    plt.hist(mean_error_mlp, bins=50, alpha=0.6, label='MLP')
    plt.hist(mean_error_hybrid, bins=50, alpha=0.6, label='Hybrid')
    plt.xlabel('Mean Relative Error (%)')
    plt.ylabel('Frequency')
    plt.title('Error Distribution: MLP vs Hybrid')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('./data/fit_results/model_comparison_error_hist.png')
    plt.show()

if __name__ == '__main__':
    main() 
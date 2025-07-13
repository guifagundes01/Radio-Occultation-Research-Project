import torch
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from model.chapman_hybrid_v2 import ChapmanPredictor, ProfileCorrector, generate_chapman_profiles
import os
from transform.add_dip import calculate_dip
import datetime

MODEL_DIR = './data/fit_results/hybrid_model'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Reasonable defaults
DEFAULT_F107 = 140
DEFAULT_KP = 2
DEFAULT_DATE = '2023-06-19 21:30:00'  # UTC

# Helper: circular encoding
def circular_encode(value, max_value):
    sin_val = np.sin(2 * np.pi * value / max_value)
    cos_val = np.cos(2 * np.pi * value / max_value)
    return sin_val, cos_val

# Helper: convert datetime to features
def get_time_features(dt):
    month_sin, month_cos = circular_encode(dt.month, 12)
    doy_sin, doy_cos = circular_encode(dt.timetuple().tm_yday, 365)
    hour_sin, hour_cos = circular_encode(dt.hour, 24)
    return [month_sin, month_cos, doy_sin, doy_cos, hour_sin, hour_cos]

# Load scalers
def load_scalers():
    X_scaler = np.load(f'{MODEL_DIR}/X_scaler.npy', allow_pickle=True).item()
    y_scaler = np.load(f'{MODEL_DIR}/y_scaler.npy', allow_pickle=True).item()
    return X_scaler, y_scaler

# Load model
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

# Main worldmap function
def plot_worldmap_nmf2_hmf2():
    # 1. Grid of lat/lon (latitudes between -45 and +45, longitudes full range)
    lats = np.linspace(-45, 45, 45)
    lons = np.linspace(-180, 180, 90)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    n_points = lat_grid.size
    # Calculate dip for each grid point using IGRF
    alt = 300  # km
    date = datetime.datetime.strptime(DEFAULT_DATE, '%Y-%m-%d %H:%M:%S')
    dip = np.array([
        calculate_dip(lat, lon, alt, date)
        for lat, lon in zip(lat_grid.flatten(), lon_grid.flatten())
    ])
    # 2. Fixed time and geophysical params
    utc_datetime = datetime.datetime.strptime(DEFAULT_DATE, '%Y-%m-%d %H:%M:%S')
    f107 = DEFAULT_F107
    kp = DEFAULT_KP
    # 3. Altitude grid (use same as training)
    # Load altitude from data file
    import h5py
    with h5py.File('./data/filtered/electron_density_profiles_2023_with_fits.h5', 'r') as f:
        altitude = f['altitude'][:]
    # 4. Dip already calculated above
    # 5. Build local time features for each location
    def get_time_features(dt):
        month_sin, month_cos = circular_encode(dt.month, 12)
        doy_sin, doy_cos = circular_encode(dt.timetuple().tm_yday, 365)
        hour_sin, hour_cos = circular_encode(dt.hour, 24)
        return [month_sin, month_cos, doy_sin, doy_cos, hour_sin, hour_cos]
    # Compute local datetimes for each longitude
    local_datetimes = [
        utc_datetime + datetime.timedelta(hours=lon / 15.0)
        for lon in lon_grid.flatten()
    ]
    # Print the first 10 local times and their longitudes for checking
    print('First 10 local times and longitudes:')
    for i in range(50):
        print(f'Longitude: {lon_grid.flatten()[i]:7.2f}°, Local time: {local_datetimes[i]}')
    # Calculate dip for each grid point using IGRF and local time
    dip = np.array([
        calculate_dip(lat, lon, alt, local_dt)
        for lat, lon, local_dt in zip(lat_grid.flatten(), lon_grid.flatten(), local_datetimes)
    ])
    time_features = np.array([get_time_features(dt) for dt in local_datetimes])
    # 6. Build input features
    X = np.column_stack([
        lat_grid.flatten(),
        lon_grid.flatten(),
        time_features,
        np.full(n_points, f107),
        np.full(n_points, kp),
        dip
    ])
    # 7. Scale features
    X_scaler, y_scaler = load_scalers()
    X_scaled = X_scaler.transform(X)
    # 8. Load model
    model_stage1, model_stage2 = load_model(altitude)
    # 9. Predict Chapman params and profiles
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        chapman_params_scaled = model_stage1(X_tensor).cpu().numpy()
        chapman_params_pred = y_scaler.inverse_transform(chapman_params_scaled)
        chapman_profiles = generate_chapman_profiles(torch.tensor(chapman_params_pred).float().to(DEVICE), torch.tensor(altitude).float().to(DEVICE).unsqueeze(0).repeat(n_points,1))
        # Corrections
        chapman_params_pred_tensor = torch.tensor(chapman_params_pred).float().to(DEVICE)
        corrections = model_stage2(X_tensor, chapman_params_pred_tensor)
        profile_scale = torch.mean(torch.abs(chapman_profiles), dim=1, keepdim=True)
        max_correction_scale = 0.1
        corrections = corrections * profile_scale * max_correction_scale
        predicted_profiles = chapman_profiles + corrections
        profiles_np = predicted_profiles.cpu().numpy()
    # 10. Extract NmF2 and hmF2
    nmf2 = np.max(profiles_np, axis=1)
    hmf2 = altitude[np.argmax(profiles_np, axis=1)]
    nmf2_map = nmf2.reshape(lat_grid.shape)
    hmf2_map = hmf2.reshape(lat_grid.shape)
    # 11. Scatter plot: one point per profile (no binning)
    nmf2_log = np.log10(nmf2)
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(12, 6))
    sc = ax.scatter(
        lon_grid.flatten(), lat_grid.flatten(),
        c=nmf2_log, cmap='jet', s=40, edgecolor='k', transform=ccrs.PlateCarree(), vmin=5, vmax=6.5
    )
    # sc = ax.pcolor(
    #     lon_grid.flatten(), lat_grid.flatten(),
    #     c=nmf2_log, cmap='jet', s=40, edgecolor='k', transform=ccrs.PlateCarree(), vmin=5, vmax=6.5
    # )
    ax.coastlines()
    ax.set_global()
    ax.set_xticks(np.arange(-180, 181, 60), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(-90, 91, 30), crs=ccrs.PlateCarree())
    ax.set_xticklabels(np.arange(-180, 181, 60), fontsize=10)
    ax.set_yticklabels(np.arange(-90, 91, 30), fontsize=10)
    cb = plt.colorbar(sc, orientation='horizontal', pad=0.05, aspect=50)
    cb.set_label('log10 NmF2 [el/cm³]')
    plt.title('Predicted NmF2 (one point per profile, global)')
    plt.tight_layout()
    plt.savefig(f'{MODEL_DIR}/scatter_nmf2_global.png')
    plt.show()

    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(12, 6))
    sc = ax.scatter(
        lon_grid.flatten(), lat_grid.flatten(),
        c=hmf2, cmap='plasma', s=40, edgecolor='k', transform=ccrs.PlateCarree(), vmin=250, vmax=450
    )
    ax.coastlines()
    ax.set_global()
    ax.set_xticks(np.arange(-180, 181, 60), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(-90, 91, 30), crs=ccrs.PlateCarree())
    ax.set_xticklabels(np.arange(-180, 181, 60), fontsize=10)
    ax.set_yticklabels(np.arange(-90, 91, 30), fontsize=10)
    cb = plt.colorbar(sc, orientation='horizontal', pad=0.05, aspect=50)
    cb.set_label('hmF2 [km]')
    plt.title('Predicted hmF2 (one point per profile, global)')
    plt.tight_layout()
    plt.savefig(f'{MODEL_DIR}/scatter_hmf2_global.png')
    plt.show()

    # 12. Plot profiles for 5 random locations
    np.random.seed(42)
    idxs = np.random.choice(n_points, size=5, replace=False)
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    for idx in idxs:
        profile = profiles_np[idx]
        lat = lat_grid.flatten()[idx]
        lon = lon_grid.flatten()[idx]
        ax3.plot(profile, altitude, label=f'lat={lat:.1f}, lon={lon:.1f}')
    ax3.set_xlabel('Electron Density (m^-3)')
    ax3.set_ylabel('Altitude (km)')
    ax3.set_title('Predicted Electron Density Profiles\n(5 Random Locations)')
    ax3.legend()
    ax3.grid(True)
    plt.tight_layout()
    plt.savefig(f'{MODEL_DIR}/random_profiles.png')
    plt.show()

if __name__ == '__main__':
    plot_worldmap_nmf2_hmf2() 
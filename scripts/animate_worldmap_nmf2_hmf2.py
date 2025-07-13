import torch
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from model.chapman_hybrid_v2 import ChapmanPredictor, ProfileCorrector, generate_chapman_profiles
import os
from transform.add_dip import calculate_dip
import datetime
from matplotlib.animation import FuncAnimation
import imageio
import tempfile
import shutil

MODEL_DIR = './data/fit_results/hybrid_model'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Reasonable defaults
DEFAULT_F107 = 140
DEFAULT_KP = 2
BASE_DATE = '2023-06-19'  # Base date for the animation

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

# Function to generate predictions for a specific UTC time
def generate_predictions_for_time(utc_datetime, model_stage1, model_stage2, X_scaler, y_scaler, 
                                 lat_grid, lon_grid, altitude, f107, kp):
    """Generate NmF2 and hmF2 predictions for a specific UTC time."""
    n_points = lat_grid.size
    
    # Compute local datetimes for each longitude
    local_datetimes = [
        utc_datetime + datetime.timedelta(hours=lon / 15.0)
        for lon in lon_grid.flatten()
    ]
    
    # Calculate dip for each grid point using IGRF and local time
    alt = 300  # km
    dip = np.array([
        calculate_dip(lat, lon, alt, local_dt)
        for lat, lon, local_dt in zip(lat_grid.flatten(), lon_grid.flatten(), local_datetimes)
    ])
    
    # Build time features
    time_features = np.array([get_time_features(dt) for dt in local_datetimes])
    
    # Build input features
    X = np.column_stack([
        lat_grid.flatten(),
        lon_grid.flatten(),
        time_features,
        np.full(n_points, f107),
        np.full(n_points, kp),
        dip
    ])
    
    # Scale features
    X_scaled = X_scaler.transform(X)
    
    # Predict Chapman params and profiles
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        chapman_params_scaled = model_stage1(X_tensor).cpu().numpy()
        chapman_params_pred = y_scaler.inverse_transform(chapman_params_scaled)
        chapman_profiles = generate_chapman_profiles(
            torch.tensor(chapman_params_pred).float().to(DEVICE), 
            torch.tensor(altitude).float().to(DEVICE).unsqueeze(0).repeat(n_points, 1)
        )
        
        # Corrections
        chapman_params_pred_tensor = torch.tensor(chapman_params_pred).float().to(DEVICE)
        corrections = model_stage2(X_tensor, chapman_params_pred_tensor)
        profile_scale = torch.mean(torch.abs(chapman_profiles), dim=1, keepdim=True)
        max_correction_scale = 0.1
        corrections = corrections * profile_scale * max_correction_scale
        predicted_profiles = chapman_profiles + corrections
        profiles_np = predicted_profiles.cpu().numpy()
    
    # Extract NmF2 and hmF2
    nmf2 = np.max(profiles_np, axis=1)
    hmf2 = altitude[np.argmax(profiles_np, axis=1)]
    
    return nmf2, hmf2

# Function to create a single frame
def create_frame(utc_datetime, model_stage1, model_stage2, X_scaler, y_scaler, 
                lat_grid, lon_grid, altitude, f107, kp, param_type='nmf2'):
    """Create a single frame for the animation."""
    nmf2, hmf2 = generate_predictions_for_time(
        utc_datetime, model_stage1, model_stage2, X_scaler, y_scaler,
        lat_grid, lon_grid, altitude, f107, kp
    )
    
    # Create the plot
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(12, 6))
    
    if param_type == 'nmf2':
        values = np.log10(nmf2)
        cmap = 'jet'
        vmin, vmax = 5, 6.5
        label = 'log10 NmF2 [el/cm³]'
        title = f'Predicted NmF2 - UTC: {utc_datetime.strftime("%Y-%m-%d %H:%M")}'
    else:  # hmf2
        values = hmf2
        cmap = 'plasma'
        vmin, vmax = 250, 450
        label = 'hmF2 [km]'
        title = f'Predicted hmF2 - UTC: {utc_datetime.strftime("%Y-%m-%d %H:%M")}'
    
    sc = ax.scatter(
        lon_grid.flatten(), lat_grid.flatten(),
        c=values, cmap=cmap, s=40, edgecolor='k', 
        transform=ccrs.PlateCarree(), vmin=vmin, vmax=vmax
    )
    
    ax.coastlines()
    ax.set_global()
    ax.set_xticks(np.arange(-180, 181, 60), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(-90, 91, 30), crs=ccrs.PlateCarree())
    ax.set_xticklabels(np.arange(-180, 181, 60), fontsize=10)
    ax.set_yticklabels(np.arange(-90, 91, 30), fontsize=10)
    
    cb = plt.colorbar(sc, orientation='horizontal', pad=0.05, aspect=50)
    cb.set_label(label)
    plt.title(title)
    plt.tight_layout()
    
    return fig

# Main animation function
def create_worldmap_animation(param_type='nmf2', hours_step=1, frame_duration=2, output_path=None):
    """
    Create an animated GIF showing how the world map changes with UTC time.
    
    Args:
        param_type: 'nmf2' or 'hmf2'
        hours_step: Time step in hours between frames
        frame_duration: Duration of each frame in seconds
        output_path: Path to save the GIF (optional)
    """
    print(f"Creating animation for {param_type} with {hours_step}-hour steps...")
    
    # Setup
    lats = np.linspace(-45, 45, 45)
    lons = np.linspace(-180, 180, 90)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    
    # Load data and model
    import h5py
    with h5py.File('./data/filtered/electron_density_profiles_2023_with_fits.h5', 'r') as f:
        altitude = f['altitude'][:]
    
    X_scaler, y_scaler = load_scalers()
    model_stage1, model_stage2 = load_model(altitude)
    
    # Generate time sequence (24 hours)
    base_datetime = datetime.datetime.strptime(BASE_DATE, '%Y-%m-%d')
    time_sequence = [
        base_datetime + datetime.timedelta(hours=i * hours_step)
        for i in range(0, 24 // hours_step + 1)
    ]
    
    # Create temporary directory for frames
    temp_dir = tempfile.mkdtemp()
    frame_paths = []
    
    try:
        # Generate frames
        for i, utc_datetime in enumerate(time_sequence):
            print(f"Generating frame {i+1}/{len(time_sequence)}: {utc_datetime}")
            
            fig = create_frame(
                utc_datetime, model_stage1, model_stage2, X_scaler, y_scaler,
                lat_grid, lon_grid, altitude, DEFAULT_F107, DEFAULT_KP, param_type
            )
            
            frame_path = os.path.join(temp_dir, f'frame_{i:03d}.png')
            fig.savefig(frame_path, dpi=100, bbox_inches='tight')
            frame_paths.append(frame_path)
            plt.close(fig)
        
        # Create GIF
        if output_path is None:
            output_path = f'{MODEL_DIR}/animation_{param_type}_24h.gif'
        
        print(f"Creating GIF: {output_path}")
        with imageio.get_writer(output_path, mode='I', duration=frame_duration) as writer:
            for frame_path in frame_paths:
                image = imageio.imread(frame_path)
                writer.append_data(image)
        
        print(f"Animation saved to: {output_path}")
        
    finally:
        # Clean up temporary files
        shutil.rmtree(temp_dir)
    
    return output_path

# Function to create both NmF2 and hmF2 animations
def create_both_animations(hours_step=1, frame_duration=2):
    """Create animations for both NmF2 and hmF2."""
    print("Creating NmF2 animation...")
    nmf2_path = create_worldmap_animation('nmf2', hours_step, frame_duration)
    
    print("Creating hmF2 animation...")
    hmf2_path = create_worldmap_animation('hmf2', hours_step, frame_duration)
    
    return nmf2_path, hmf2_path

if __name__ == '__main__':
    # Create animations with 2-hour steps (12 frames total) and 1 second per frame
    nmf2_path, hmf2_path = create_both_animations(hours_step=2, frame_duration=10)
    print(f"\nAnimations created:")
    print(f"NmF2: {nmf2_path}")
    print(f"hmF2: {hmf2_path}")
    
    # Example: Create slower animation (3 seconds per frame)
    # nmf2_path, hmf2_path = create_both_animations(hours_step=2, frame_duration=3)
    
    # Example: Create faster animation (0.5 seconds per frame)
    # nmf2_path, hmf2_path = create_both_animations(hours_step=2, frame_duration=0.5) 
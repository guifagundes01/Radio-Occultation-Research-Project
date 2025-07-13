import torch
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from model.chapman_hybrid_v2 import ChapmanPredictor, ProfileCorrector, generate_chapman_profiles
import os
from transform.add_dip import calculate_dip
import datetime
from matplotlib.animation import FuncAnimation
import matplotlib.patches as mpatches

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

# Animation class
class WorldMapAnimator:
    def __init__(self, hours_step=1):
        self.hours_step = hours_step
        self.base_datetime = datetime.datetime.strptime(BASE_DATE, '%Y-%m-%d')
        self.current_frame = 0
        
        # Setup grid
        self.lats = np.linspace(-45, 45, 45)
        self.lons = np.linspace(-180, 180, 90)
        self.lon_grid, self.lat_grid = np.meshgrid(self.lons, self.lats)
        
        # Load data and model
        import h5py
        with h5py.File('./data/filtered/electron_density_profiles_2023_with_fits.h5', 'r') as f:
            self.altitude = f['altitude'][:]
        
        self.X_scaler, self.y_scaler = load_scalers()
        self.model_stage1, self.model_stage2 = load_model(self.altitude)
        
        # Calculate total frames
        self.total_frames = 24 // hours_step + 1
        
        # Setup figure
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, 
                                                      subplot_kw={'projection': ccrs.PlateCarree()}, 
                                                      figsize=(20, 8))
        
        # Initialize scatter plots
        self.sc1 = None
        self.sc2 = None
        
        # Setup axes
        for ax in [self.ax1, self.ax2]:
            ax.coastlines()
            ax.set_global()
            ax.set_xticks(np.arange(-180, 181, 60), crs=ccrs.PlateCarree())
            ax.set_yticks(np.arange(-90, 91, 30), crs=ccrs.PlateCarree())
            ax.set_xticklabels(np.arange(-180, 181, 60), fontsize=10)
            ax.set_yticklabels(np.arange(-90, 91, 30), fontsize=10)
        
        self.ax1.set_title('Predicted NmF2')
        self.ax2.set_title('Predicted hmF2')
        
        # Add colorbars
        self.cb1 = plt.colorbar(plt.cm.ScalarMappable(cmap='jet', 
                                                     norm=plt.Normalize(5, 6.5)), 
                               ax=self.ax1, orientation='horizontal', pad=0.05, aspect=30)
        self.cb1.set_label('log10 NmF2 [el/cm³]')
        
        self.cb2 = plt.colorbar(plt.cm.ScalarMappable(cmap='plasma', 
                                                     norm=plt.Normalize(250, 450)), 
                               ax=self.ax2, orientation='horizontal', pad=0.05, aspect=30)
        self.cb2.set_label('hmF2 [km]')
        
        plt.tight_layout()
    
    def init(self):
        """Initialize animation."""
        return []
    
    def animate(self, frame):
        """Update animation frame."""
        # Calculate current UTC time
        utc_datetime = self.base_datetime + datetime.timedelta(hours=frame * self.hours_step)
        
        # Generate predictions
        nmf2, hmf2 = generate_predictions_for_time(
            utc_datetime, self.model_stage1, self.model_stage2, 
            self.X_scaler, self.y_scaler, self.lat_grid, self.lon_grid, 
            self.altitude, DEFAULT_F107, DEFAULT_KP
        )
        
        # Clear previous scatter plots
        if self.sc1 is not None:
            self.sc1.remove()
        if self.sc2 is not None:
            self.sc2.remove()
        
        # Update NmF2 plot
        nmf2_log = np.log10(nmf2)
        self.sc1 = self.ax1.scatter(
            self.lon_grid.flatten(), self.lat_grid.flatten(),
            c=nmf2_log, cmap='jet', s=40, edgecolor='k', 
            transform=ccrs.PlateCarree(), vmin=5, vmax=6.5
        )
        
        # Update hmF2 plot
        self.sc2 = self.ax2.scatter(
            self.lon_grid.flatten(), self.lat_grid.flatten(),
            c=hmf2, cmap='plasma', s=40, edgecolor='k', 
            transform=ccrs.PlateCarree(), vmin=250, vmax=450
        )
        
        # Update titles with time
        time_str = utc_datetime.strftime("%Y-%m-%d %H:%M UTC")
        self.ax1.set_title(f'Predicted NmF2 - {time_str}')
        self.ax2.set_title(f'Predicted hmF2 - {time_str}')
        
        # Add progress indicator
        progress = (frame + 1) / self.total_frames * 100
        self.fig.suptitle(f'Animation Progress: {progress:.1f}% ({frame + 1}/{self.total_frames})', 
                         fontsize=14, y=0.95)
        
        return [self.sc1, self.sc2]
    
    def run_animation(self, interval=500, save_path=None):
        """Run the animation."""
        anim = FuncAnimation(self.fig, self.animate, init_func=self.init,
                           frames=self.total_frames, interval=interval, 
                           blit=False, repeat=True)
        
        if save_path:
            print(f"Saving animation to {save_path}...")
            anim.save(save_path, writer='pillow', fps=2)
        
        plt.show()
        return anim

def create_realtime_animation(hours_step=1, interval=500, save_path=None):
    """Create and run a real-time animation."""
    animator = WorldMapAnimator(hours_step=hours_step)
    return animator.run_animation(interval=interval, save_path=save_path)

if __name__ == '__main__':
    # Create real-time animation with 2-hour steps
    anim = create_realtime_animation(hours_step=2, interval=1000) 
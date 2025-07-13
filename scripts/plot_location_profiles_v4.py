import torch
import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
from model.chapman_hybrid_v4 import ChapmanPredictor, ProfileCorrector, generate_chapman_profiles
import datetime
from transform.add_dip import calculate_dip

# Configuration
MODEL_DIR = './data/fit_results/hybrid_model_v4'
DATA_PATH = './data/filtered/electron_density_profiles_2023_with_fits.h5'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Helper functions
def circular_encode(value, max_value):
    sin_val = np.sin(2 * np.pi * value / max_value)
    cos_val = np.cos(2 * np.pi * value / max_value)
    return sin_val, cos_val

def get_time_features(dt):
    month_sin, month_cos = circular_encode(dt.month, 12)
    doy_sin, doy_cos = circular_encode(dt.timetuple().tm_yday, 365)
    hour_sin, hour_cos = circular_encode(dt.hour, 24)
    return [month_sin, month_cos, doy_sin, doy_cos, hour_sin, hour_cos]

def load_model_and_scalers():
    """Load the hybrid V4 model and scalers"""
    # Load altitude from data file (constant for all profiles)
    with h5py.File(DATA_PATH, 'r') as f:
        altitude = f['altitude'][:]
    
    # Load scalers
    X_scaler = np.load(f'{MODEL_DIR}/X_scaler.npy', allow_pickle=True)
    y_scaler = np.load(f'{MODEL_DIR}/y_scaler.npy', allow_pickle=True)
    # If saved as dict, use .item()
    if hasattr(X_scaler, 'item'):
        X_scaler = X_scaler.item()
    if hasattr(y_scaler, 'item'):
        y_scaler = y_scaler.item()
    
    # Load model
    model_stage1 = ChapmanPredictor(input_size=11, hidden_sizes=[128, 64])
    model_stage2 = ProfileCorrector(profile_size=len(altitude), feature_size=11, hidden_sizes=[128, 64])
    checkpoint = torch.load(f'{MODEL_DIR}/finetune_best.pth', map_location=DEVICE)
    model_stage1.load_state_dict(checkpoint['stage1_state_dict'])
    model_stage2.load_state_dict(checkpoint['stage2_state_dict'])
    model_stage1.to(DEVICE)
    model_stage2.to(DEVICE)
    model_stage1.eval()
    model_stage2.eval()
    
    return model_stage1, model_stage2, X_scaler, y_scaler, altitude

def generate_diurnal_predictions(model_stage1, model_stage2, X_scaler, y_scaler, altitude, 
                               latitude, longitude, timezone_offset=0, base_date='2023-11-19'):
    """Generate predictions for a given location throughout the day"""
    # Create 24-hour time series (local time)
    local_hours = np.arange(24)
    nmf2_values = []
    hmf2_values = []
    profiles = []
    
    # Fixed geophysical parameters (use median values from data)
    f107 = 140  # Default solar flux
    kp = 1.5      # Default geomagnetic activity
    
    for local_hour in local_hours:
        # Create local datetime
        local_dt = datetime.datetime.strptime(base_date, '%Y-%m-%d') + datetime.timedelta(hours=int(local_hour))
        
        # Convert to UTC for model input
        utc_dt = local_dt - datetime.timedelta(hours=timezone_offset)
        
        # Calculate dip angle using local time
        dip = calculate_dip(latitude, longitude, 300, local_dt)
        
        # Prepare input features using local time
        time_features = get_time_features(utc_dt)
        X = np.array([[
            latitude,
            longitude,
            *time_features,
            f107,
            kp,
            dip
        ]])
        
        # Scale features
        X_scaled = X_scaler.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32, device=DEVICE)
        
        # Generate predictions
        with torch.no_grad():
            chapman_params_scaled = model_stage1(X_tensor).cpu().numpy()
            chapman_params_pred = y_scaler.inverse_transform(chapman_params_scaled)
            
            # Generate Chapman profile
            altitude_tensor = torch.tensor(altitude, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            chapman_profiles = generate_chapman_profiles(
                torch.tensor(chapman_params_pred).float().to(DEVICE), 
                altitude_tensor
            )
            
            # Apply corrections
            chapman_params_pred_tensor = torch.tensor(chapman_params_pred).float().to(DEVICE)
            corrections = model_stage2(X_tensor, chapman_params_pred_tensor)
            profile_scale = torch.mean(torch.abs(chapman_profiles), dim=1, keepdim=True)
            max_correction_scale = 0.1
            corrections = corrections * profile_scale * max_correction_scale
            predicted_profiles = chapman_profiles + corrections
            
            profile = predicted_profiles.cpu().numpy()[0]
            
            # Extract NmF2 and hmF2
            nmf2 = np.max(profile)
            hmf2 = altitude[np.argmax(profile)]
            
            nmf2_values.append(nmf2)
            hmf2_values.append(hmf2)
            profiles.append(profile)
    
    return np.array(nmf2_values), np.array(hmf2_values), np.array(profiles), local_hours

def plot_location_profiles(profiles, altitude, hours, location_name, save_dir='./data/fit_results/location_analysis_v4'):
    """Plot electron density profiles for different hours in the same figure"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Create a single figure for all profiles
    plt.figure(figsize=(12, 8))
    
    # Define colors for different hours
    colors = plt.cm.viridis(np.linspace(0, 1, len(profiles)))
    
    # Plot all profiles
    for i, (profile, hour, color) in enumerate(zip(profiles, hours, colors)):
        plt.semilogx(profile, altitude, color=color, linewidth=2, 
                    label=f'{hour:02d}:00 LT', alpha=0.8)
    
    plt.xlabel('Electron Density (cm⁻³)', fontsize=12)
    plt.ylabel('Altitude (km)', fontsize=12)
    plt.title(f'{location_name} - Electron Density Profiles Throughout the Day (V4 Model)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    # Add annotations for key times (highlight max and min NmF2)
    max_nmf2_idx = np.argmax([np.max(p) for p in profiles])
    min_nmf2_idx = np.argmin([np.max(p) for p in profiles])
    
    # Highlight maximum NmF2 profile
    max_profile = profiles[max_nmf2_idx]
    max_nmf2 = np.max(max_profile)
    max_hmf2 = altitude[np.argmax(max_profile)]
    plt.semilogx(max_profile, altitude, color='red', linewidth=3, 
                label=f'Max NmF2: {max_nmf2:.1e} cm⁻³ at {hours[max_nmf2_idx]:02d}:00 LT')
    
    # Highlight minimum NmF2 profile
    min_profile = profiles[min_nmf2_idx]
    min_nmf2 = np.max(min_profile)
    min_hmf2 = altitude[np.argmax(min_profile)]
    plt.semilogx(min_profile, altitude, color='blue', linewidth=3, 
                label=f'Min NmF2: {min_nmf2:.1e} cm⁻³ at {hours[min_nmf2_idx]:02d}:00 LT')
    
    # Add text box with summary
    summary_text = f'Diurnal Range:\nNmF2: {max_nmf2/min_nmf2:.1f}x\nhmF2: {max_hmf2-min_hmf2:.0f} km'
    plt.text(0.02, 0.98, summary_text, transform=plt.gca().transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/{location_name.lower().replace(" ", "_")}_profiles_diurnal_combined_v4.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Also create a version with just a few representative hours for clarity
    plt.figure(figsize=(10, 6))
    
    # Select representative hours
    selected_hours = [0, 6, 12, 18, 22]  # Midnight, dawn, noon, dusk, late night
    selected_indices = [0, 6, 12, 18, 22]  # Corresponding array indices
    colors_selected = ['black', 'orange', 'red', 'purple', 'blue']
    
    for hour, idx, color in zip(selected_hours, selected_indices, colors_selected):
        profile = profiles[idx]
        nmf2 = np.max(profile)
        hmf2 = altitude[np.argmax(profile)]
        plt.semilogx(profile, altitude, color=color, linewidth=2.5, 
                    label=f'{hour:02d}:00 LT (NmF2: {nmf2:.1e}, hmF2: {hmf2:.0f}km)')
    
    plt.xlabel('Electron Density (cm⁻³)', fontsize=12)
    plt.ylabel('Altitude (km)', fontsize=12)
    plt.title(f'{location_name} - Representative Electron Density Profiles (V4 Model)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/{location_name.lower().replace(" ", "_")}_profiles_representative_v4.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_diurnal_variation(nmf2_values, hmf2_values, hours, location_name, save_dir='./data/fit_results/location_analysis_v4'):
    """Plot diurnal variation of NmF2 and hmF2"""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot NmF2 variation
    ax1.plot(hours, nmf2_values, 'b-o', linewidth=2, markersize=6)
    ax1.set_xlabel('Local Time (hours)')
    ax1.set_ylabel('NmF2 (cm⁻³)')
    ax1.set_title(f'Diurnal Variation of NmF2 - {location_name} (V4 Model)')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(np.arange(0, 25, 2))
    
    # Add annotations for key times
    max_nmf2_idx = np.argmax(nmf2_values)
    min_nmf2_idx = np.argmin(nmf2_values)
    ax1.annotate(f'Max: {nmf2_values[max_nmf2_idx]:.1e} cm⁻³\nat {hours[max_nmf2_idx]:02d}:00 LT', 
                xy=(hours[max_nmf2_idx], nmf2_values[max_nmf2_idx]), 
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    ax1.annotate(f'Min: {nmf2_values[min_nmf2_idx]:.1e} cm⁻³\nat {hours[min_nmf2_idx]:02d}:00 LT', 
                xy=(hours[min_nmf2_idx], nmf2_values[min_nmf2_idx]), 
                xytext=(10, -10), textcoords='offset points',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # Plot hmF2 variation
    ax2.plot(hours, hmf2_values, 'r-o', linewidth=2, markersize=6)
    ax2.set_xlabel('Local Time (hours)')
    ax2.set_ylabel('hmF2 (km)')
    ax2.set_title(f'Diurnal Variation of hmF2 - {location_name} (V4 Model)')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(np.arange(0, 25, 2))
    
    # Add annotations for key times
    max_hmf2_idx = np.argmax(hmf2_values)
    min_hmf2_idx = np.argmin(hmf2_values)
    ax2.annotate(f'Max: {hmf2_values[max_hmf2_idx]:.0f} km\nat {hours[max_hmf2_idx]:02d}:00 LT', 
                xy=(hours[max_hmf2_idx], hmf2_values[max_hmf2_idx]), 
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    ax2.annotate(f'Min: {hmf2_values[min_hmf2_idx]:.0f} km\nat {hours[min_hmf2_idx]:02d}:00 LT', 
                xy=(hours[min_hmf2_idx], hmf2_values[min_hmf2_idx]), 
                xytext=(10, -10), textcoords='offset points',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/{location_name.lower().replace(" ", "_")}_diurnal_variation_v4.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print(f"\n=== {location_name} Diurnal Variation Summary (V4 Model) ===")
    print(f"NmF2 - Max: {nmf2_values[max_nmf2_idx]:.1e} cm⁻³ at {hours[max_nmf2_idx]:02d}:00 LT")
    print(f"NmF2 - Min: {nmf2_values[min_nmf2_idx]:.1e} cm⁻³ at {hours[min_nmf2_idx]:02d}:00 LT")
    print(f"NmF2 - Range: {nmf2_values[max_nmf2_idx]/nmf2_values[min_nmf2_idx]:.1f}x")
    print(f"hmF2 - Max: {hmf2_values[max_hmf2_idx]:.0f} km at {hours[max_hmf2_idx]:02d}:00 LT")
    print(f"hmF2 - Min: {hmf2_values[min_hmf2_idx]:.0f} km at {hours[min_hmf2_idx]:02d}:00 LT")
    print(f"hmF2 - Range: {hmf2_values[max_hmf2_idx] - hmf2_values[min_hmf2_idx]:.0f} km")

def compare_v3_v4_predictions(latitude, longitude, location_name, timezone_offset=0, base_date='2023-06-19'):
    """Compare predictions between V3 and V4 models for the same location"""
    print(f"Loading both V3 and V4 models for comparison...")
    
    # Import V3 models
    from model.chapman_hybrid_v3 import ChapmanPredictor as ChapmanPredictorV3, ProfileCorrector as ProfileCorrectorV3
    from model.chapman_function import generate_chapman_profiles
    
    MODEL_DIR_V3 = './data/fit_results/hybrid_model_v3'
    
    # Load V3 model and scalers
    with h5py.File(DATA_PATH, 'r') as f:
        altitude = f['altitude'][:]
    
    X_scaler_v3 = np.load(f'{MODEL_DIR_V3}/X_scaler.npy', allow_pickle=True).item()
    y_scaler_v3 = np.load(f'{MODEL_DIR_V3}/y_scaler.npy', allow_pickle=True).item()
    
    model_stage1_v3 = ChapmanPredictorV3(input_size=11, hidden_sizes=[128, 64])
    model_stage2_v3 = ProfileCorrectorV3(profile_size=len(altitude), feature_size=11, hidden_sizes=[128, 64])
    checkpoint_v3 = torch.load(f'{MODEL_DIR_V3}/best_hybrid_model_v3.pth', map_location=DEVICE)
    model_stage1_v3.load_state_dict(checkpoint_v3['stage1_state_dict'])
    model_stage2_v3.load_state_dict(checkpoint_v3['stage2_state_dict'])
    model_stage1_v3.to(DEVICE)
    model_stage2_v3.to(DEVICE)
    model_stage1_v3.eval()
    model_stage2_v3.eval()
    
    # Load V4 model and scalers
    model_stage1_v4, model_stage2_v4, X_scaler_v4, y_scaler_v4, _ = load_model_and_scalers()
    
    # Generate predictions for both models
    print(f"Generating predictions for both models...")
    nmf2_v3, hmf2_v3, profiles_v3, local_hours = generate_diurnal_predictions(
        model_stage1_v3, model_stage2_v3, X_scaler_v3, y_scaler_v3, altitude,
        latitude, longitude, timezone_offset, base_date
    )
    
    nmf2_v4, hmf2_v4, profiles_v4, _ = generate_diurnal_predictions(
        model_stage1_v4, model_stage2_v4, X_scaler_v4, y_scaler_v4, altitude,
        latitude, longitude, timezone_offset, base_date
    )
    
    # Create comparison plots
    save_dir = './data/fit_results/location_analysis_comparison_v3_v4'
    os.makedirs(save_dir, exist_ok=True)
    
    # Compare NmF2 predictions
    plt.figure(figsize=(12, 8))
    plt.plot(local_hours, nmf2_v3, 'b-o', linewidth=2, markersize=6, label='V3 Model')
    plt.plot(local_hours, nmf2_v4, 'r-s', linewidth=2, markersize=6, label='V4 Model')
    plt.xlabel('Local Time (hours)')
    plt.ylabel('NmF2 (cm⁻³)')
    plt.title(f'NmF2 Comparison: V3 vs V4 - {location_name}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xticks(np.arange(0, 25, 2))
    
    # Add statistics
    max_v3 = np.max(nmf2_v3)
    max_v4 = np.max(nmf2_v4)
    min_v3 = np.min(nmf2_v3)
    min_v4 = np.min(nmf2_v4)
    
    plt.text(0.02, 0.98, f'V3 Max: {max_v3:.1e}\nV4 Max: {max_v4:.1e}\nRatio: {max_v4/max_v3:.2f}', 
            transform=plt.gca().transAxes, verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/{location_name.lower().replace(" ", "_")}_nmf2_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Compare hmF2 predictions
    plt.figure(figsize=(12, 8))
    plt.plot(local_hours, hmf2_v3, 'b-o', linewidth=2, markersize=6, label='V3 Model')
    plt.plot(local_hours, hmf2_v4, 'r-s', linewidth=2, markersize=6, label='V4 Model')
    plt.xlabel('Local Time (hours)')
    plt.ylabel('hmF2 (km)')
    plt.title(f'hmF2 Comparison: V3 vs V4 - {location_name}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xticks(np.arange(0, 25, 2))
    
    # Add statistics
    max_hmf2_v3 = np.max(hmf2_v3)
    max_hmf2_v4 = np.max(hmf2_v4)
    min_hmf2_v3 = np.min(hmf2_v3)
    min_hmf2_v4 = np.min(hmf2_v4)
    
    plt.text(0.02, 0.98, f'V3 Range: {max_hmf2_v3-min_hmf2_v3:.0f} km\nV4 Range: {max_hmf2_v4-min_hmf2_v4:.0f} km\nDiff: {max_hmf2_v4-max_hmf2_v3:.0f} km', 
            transform=plt.gca().transAxes, verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/{location_name.lower().replace(" ", "_")}_hmf2_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Compare representative profiles
    plt.figure(figsize=(12, 8))
    
    # Select noon profile for comparison
    noon_idx = 12
    profile_v3 = profiles_v3[noon_idx]
    profile_v4 = profiles_v4[noon_idx]
    
    plt.semilogx(profile_v3, altitude, 'b-', linewidth=2.5, label=f'V3 Model (12:00 LT)')
    plt.semilogx(profile_v4, altitude, 'r-', linewidth=2.5, label=f'V4 Model (12:00 LT)')
    
    plt.xlabel('Electron Density (cm⁻³)', fontsize=12)
    plt.ylabel('Altitude (km)', fontsize=12)
    plt.title(f'{location_name} - Noon Profile Comparison: V3 vs V4', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add statistics
    nmf2_v3_noon = np.max(profile_v3)
    nmf2_v4_noon = np.max(profile_v4)
    hmf2_v3_noon = altitude[np.argmax(profile_v3)]
    hmf2_v4_noon = altitude[np.argmax(profile_v4)]
    
    plt.text(0.02, 0.98, f'V3: NmF2={nmf2_v3_noon:.1e}, hmF2={hmf2_v3_noon:.0f}km\nV4: NmF2={nmf2_v4_noon:.1e}, hmF2={hmf2_v4_noon:.0f}km\nRatio: {nmf2_v4_noon/nmf2_v3_noon:.2f}', 
            transform=plt.gca().transAxes, verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/{location_name.lower().replace(" ", "_")}_profile_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print comparison summary
    print(f"\n=== {location_name} V3 vs V4 Comparison Summary ===")
    print(f"NmF2 - V3 Max: {max_v3:.1e} cm⁻³, V4 Max: {max_v4:.1e} cm⁻³, Ratio: {max_v4/max_v3:.2f}")
    print(f"hmF2 - V3 Range: {max_hmf2_v3-min_hmf2_v3:.0f} km, V4 Range: {max_hmf2_v4-min_hmf2_v4:.0f} km")
    print(f"Noon Profile - V3 NmF2: {nmf2_v3_noon:.1e}, V4 NmF2: {nmf2_v4_noon:.1e}, Ratio: {nmf2_v4_noon/nmf2_v3_noon:.2f}")

def analyze_location_diurnal(latitude, longitude, location_name, timezone_offset=0, base_date='2023-06-19'):
    """Main function to run diurnal analysis for any location using V4 model"""
    print(f"Loading V4 model and altitude data for {location_name}...")
    model_stage1, model_stage2, X_scaler, y_scaler, altitude = load_model_and_scalers()
    
    print(f"Altitude range: {altitude.min():.0f} to {altitude.max():.0f} km")
    print(f"Number of altitude points: {len(altitude)}")
    print(f"Location: {location_name} ({latitude:.2f}°N, {longitude:.2f}°E)")
    print(f"Timezone offset: UTC{timezone_offset:+d}")
    
    # Generate diurnal predictions
    print(f"Generating diurnal predictions for {location_name} (Local Time)...")
    nmf2_values, hmf2_values, profiles, local_hours = generate_diurnal_predictions(
        model_stage1, model_stage2, X_scaler, y_scaler, altitude,
        latitude, longitude, timezone_offset, base_date
    )
    
    # Plot profiles for different hours
    print("Plotting electron density profiles...")
    plot_location_profiles(profiles, altitude, local_hours, location_name)
    
    # Plot diurnal variation
    print("Plotting diurnal variation...")
    plot_diurnal_variation(nmf2_values, hmf2_values, local_hours, location_name)
    
    print("Analysis complete! Check the output directory for plots.")
    
    return nmf2_values, hmf2_values, profiles, local_hours

def main():
    """Example usage with São Luís coordinates"""
    # São Luís coordinates (example)
    SAO_LUIS_LAT = -2.5
    SAO_LUIS_LON = -44.3
    SAO_LUIS_TIMEZONE = -3  # UTC-3
    
    # Run V4 analysis
    print("=== Running V4 Model Analysis ===")
    analyze_location_diurnal(
        latitude=SAO_LUIS_LAT,
        longitude=SAO_LUIS_LON,
        location_name="São Luís",
        timezone_offset=SAO_LUIS_TIMEZONE
    )   
    
    # Run comparison analysis
    print("\n=== Running V3 vs V4 Comparison ===")
    compare_v3_v4_predictions(
        latitude=SAO_LUIS_LAT,
        longitude=SAO_LUIS_LON,
        location_name="São Luís",
        timezone_offset=SAO_LUIS_TIMEZONE
    )

if __name__ == '__main__':
    main() 
import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from datetime import datetime, timedelta
import os

def convert_seconds_to_datetime(seconds):
    """Convert seconds to datetime dictionary"""
    local_time_date = pd.to_datetime(seconds, unit='s')
    return {
        'year': local_time_date.year,
        'month': local_time_date.month,
        'doy': local_time_date.dayofyear,
        'hour': local_time_date.hour,
        'minute': local_time_date.minute
    }

def analyze_data_coverage():
    """Analyze data coverage and distribution"""
    print("Loading data...")
    
    with h5py.File('./data/filtered/electron_density_profiles_2023_with_fits.h5', 'r') as f:
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
        
        # Filter for good fits
        latitude = latitude[is_good_fit]
        longitude = longitude[is_good_fit]
        local_time = local_time[is_good_fit]
        f107 = f107[is_good_fit]
        kp = kp[is_good_fit]
        dip = dip[is_good_fit]
        electron_density = electron_density[is_good_fit]
        chapman_params = chapman_params[is_good_fit]
    
    print(f"Total profiles after filtering: {len(electron_density)}")
    
    # Convert local time to datetime
    local_datetimes = pd.to_datetime(local_time, unit='s')
    
    # Extract temporal features
    hours = local_datetimes.hour
    months = local_datetimes.month
    doy = local_datetimes.dayofyear
    
    # Create output directory
    os.makedirs('./data/analysis/prereversal_analysis', exist_ok=True)
    
    # 1. Temporal Distribution Analysis
    print("\n1. Analyzing temporal distribution...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Hour distribution
    axes[0, 0].hist(hours, bins=24, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 0].set_xlabel('Local Hour')
    axes[0, 0].set_ylabel('Number of Profiles')
    axes[0, 0].set_title('Distribution of Profiles by Local Hour')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Highlight pre-reversal hours (17-20)
    prereversal_mask = (hours >= 17) & (hours <= 20)
    axes[0, 0].axvspan(17, 20, alpha=0.3, color='red', label='Pre-reversal period')
    axes[0, 0].legend()
    
    # Month distribution
    axes[0, 1].hist(months, bins=12, alpha=0.7, color='green', edgecolor='black')
    axes[0, 1].set_xlabel('Month')
    axes[0, 1].set_ylabel('Number of Profiles')
    axes[0, 1].set_title('Distribution of Profiles by Month')
    axes[0, 1].grid(True, alpha=0.3)
    
    # DOY distribution
    axes[1, 0].hist(doy, bins=50, alpha=0.7, color='orange', edgecolor='black')
    axes[1, 0].set_xlabel('Day of Year')
    axes[1, 0].set_ylabel('Number of Profiles')
    axes[1, 0].set_title('Distribution of Profiles by Day of Year')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Highlight equinox periods
    spring_equinox = 80  # March 21
    fall_equinox = 266   # September 23
    axes[1, 0].axvspan(spring_equinox-10, spring_equinox+10, alpha=0.3, color='red', label='Spring Equinox')
    axes[1, 0].axvspan(fall_equinox-10, fall_equinox+10, alpha=0.3, color='blue', label='Fall Equinox')
    axes[1, 0].legend()
    
    # Latitude distribution
    axes[1, 1].hist(latitude, bins=30, alpha=0.7, color='purple', edgecolor='black')
    axes[1, 1].set_xlabel('Latitude (°)')
    axes[1, 1].set_ylabel('Number of Profiles')
    axes[1, 1].set_title('Distribution of Profiles by Latitude')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Highlight equatorial region
    equatorial_mask = np.abs(latitude) < 30
    axes[1, 1].axvspan(-30, 30, alpha=0.3, color='red', label='Equatorial region (±30°)')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('./data/analysis/prereversal_analysis/temporal_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Pre-reversal Coverage Analysis
    print("\n2. Analyzing pre-reversal coverage...")
    
    # Define pre-reversal conditions
    prereversal_hours = (hours >= 17) & (hours <= 20)
    equinox_periods = ((doy >= 80) & (doy <= 100)) | ((doy >= 260) & (doy <= 280))
    equatorial_region = np.abs(latitude) < 30
    
    # Combined pre-reversal condition
    prereversal_condition = prereversal_hours & equinox_periods & equatorial_region
    
    print(f"Profiles in pre-reversal hours (17-20h): {np.sum(prereversal_hours)}")
    print(f"Profiles in equinox periods: {np.sum(equinox_periods)}")
    print(f"Profiles in equatorial region: {np.sum(equatorial_region)}")
    print(f"Profiles meeting all pre-reversal conditions: {np.sum(prereversal_condition)}")
    print(f"Percentage of total profiles in pre-reversal conditions: {np.sum(prereversal_condition)/len(electron_density)*100:.2f}%")
    
    # 3. Detailed Pre-reversal Analysis
    if np.sum(prereversal_condition) > 0:
        print("\n3. Analyzing pre-reversal profiles...")
        
        # Extract pre-reversal data
        prereversal_indices = np.where(prereversal_condition)[0]
        prereversal_profiles = electron_density[prereversal_indices]
        prereversal_lat = latitude[prereversal_indices]
        prereversal_lon = longitude[prereversal_indices]
        prereversal_hours_data = hours[prereversal_indices]
        prereversal_doy = doy[prereversal_indices]
        prereversal_chapman = chapman_params[prereversal_indices]
        
        # Calculate hmF2 for pre-reversal profiles
        prereversal_hmf2 = []
        prereversal_nmf2 = []
        
        for i, profile in enumerate(prereversal_profiles):
            nmf2 = np.max(profile)
            hmf2 = altitude[np.argmax(profile)]
            prereversal_nmf2.append(nmf2)
            prereversal_hmf2.append(hmf2)
        
        prereversal_hmf2 = np.array(prereversal_hmf2)
        prereversal_nmf2 = np.array(prereversal_nmf2)
        
        # Compare with non-pre-reversal profiles
        non_prereversal_condition = ~prereversal_condition & equatorial_region
        non_prereversal_indices = np.where(non_prereversal_condition)[0]
        
        if len(non_prereversal_indices) > 0:
            non_prereversal_profiles = electron_density[non_prereversal_indices]
            non_prereversal_hmf2 = []
            non_prereversal_nmf2 = []
            
            for profile in non_prereversal_profiles:
                nmf2 = np.max(profile)
                hmf2 = altitude[np.argmax(profile)]
                non_prereversal_nmf2.append(nmf2)
                non_prereversal_hmf2.append(hmf2)
            
            non_prereversal_hmf2 = np.array(non_prereversal_hmf2)
            non_prereversal_nmf2 = np.array(non_prereversal_nmf2)
            
            # Statistical comparison
            print(f"\nPre-reversal profiles: {len(prereversal_hmf2)}")
            print(f"Non-pre-reversal profiles: {len(non_prereversal_hmf2)}")
            print(f"Mean hmF2 - Pre-reversal: {np.mean(prereversal_hmf2):.1f} km")
            print(f"Mean hmF2 - Non-pre-reversal: {np.mean(non_prereversal_hmf2):.1f} km")
            print(f"hmF2 difference: {np.mean(prereversal_hmf2) - np.mean(non_prereversal_hmf2):.1f} km")
            
            # Plot comparison
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # hmF2 comparison
            axes[0, 0].hist(prereversal_hmf2, bins=20, alpha=0.7, color='red', label='Pre-reversal', density=True)
            axes[0, 0].hist(non_prereversal_hmf2, bins=20, alpha=0.7, color='blue', label='Non-pre-reversal', density=True)
            axes[0, 0].set_xlabel('hmF2 (km)')
            axes[0, 0].set_ylabel('Density')
            axes[0, 0].set_title('hmF2 Distribution Comparison')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # NmF2 comparison
            axes[0, 1].hist(np.log10(prereversal_nmf2), bins=20, alpha=0.7, color='red', label='Pre-reversal', density=True)
            axes[0, 1].hist(np.log10(non_prereversal_nmf2), bins=20, alpha=0.7, color='blue', label='Non-pre-reversal', density=True)
            axes[0, 1].set_xlabel('log10(NmF2) (cm⁻³)')
            axes[0, 1].set_ylabel('Density')
            axes[0, 1].set_title('NmF2 Distribution Comparison')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # hmF2 vs hour for pre-reversal
            axes[1, 0].scatter(prereversal_hours_data, prereversal_hmf2, alpha=0.6, color='red', s=20)
            axes[1, 0].set_xlabel('Local Hour')
            axes[1, 0].set_ylabel('hmF2 (km)')
            axes[1, 0].set_title('hmF2 vs Hour (Pre-reversal Profiles)')
            axes[1, 0].grid(True, alpha=0.3)
            
            # hmF2 vs DOY for pre-reversal
            axes[1, 1].scatter(prereversal_doy, prereversal_hmf2, alpha=0.6, color='red', s=20)
            axes[1, 1].set_xlabel('Day of Year')
            axes[1, 1].set_ylabel('hmF2 (km)')
            axes[1, 1].set_title('hmF2 vs DOY (Pre-reversal Profiles)')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('./data/analysis/prereversal_analysis/prereversal_comparison.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            # 4. Sample Profile Analysis
            print("\n4. Analyzing sample pre-reversal profiles...")
            
            # Select a few representative pre-reversal profiles
            sample_indices = prereversal_indices[:min(5, len(prereversal_indices))]
            
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            axes = axes.flatten()
            
            for i, idx in enumerate(sample_indices):
                if i >= len(axes):
                    break
                    
                profile = electron_density[idx]
                lat = latitude[idx]
                lon = longitude[idx]
                hour = hours[idx]
                doy_val = doy[idx]
                
                axes[i].semilogx(profile, altitude, 'r-', linewidth=2)
                axes[i].set_xlabel('Electron Density (cm⁻³)')
                axes[i].set_ylabel('Altitude (km)')
                axes[i].set_title(f'Profile {i+1}\nLat: {lat:.1f}°, Lon: {lon:.1f}°\nHour: {hour:02d}:00, DOY: {doy_val}')
                axes[i].grid(True, alpha=0.3)
                
                # Mark hmF2
                nmf2 = np.max(profile)
                hmf2 = altitude[np.argmax(profile)]
                axes[i].axhline(y=hmf2, color='blue', linestyle='--', alpha=0.7, label=f'hmF2: {hmf2:.0f} km')
                axes[i].legend()
            
            plt.tight_layout()
            plt.savefig('./data/analysis/prereversal_analysis/sample_prereversal_profiles.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    else:
        print("\n3. No profiles found meeting pre-reversal conditions!")
        print("This suggests the data may not have sufficient coverage for pre-reversal analysis.")
    
    # 5. Data Coverage Summary
    print("\n5. Data Coverage Summary:")
    print("=" * 50)
    
    # Hour coverage
    hour_coverage = np.bincount(hours, minlength=24)
    print(f"Hour coverage (17-20h): {hour_coverage[17:21]}")
    print(f"Total profiles in pre-reversal hours: {np.sum(hour_coverage[17:21])}")
    
    # Latitude coverage
    equatorial_coverage = np.sum(equatorial_region)
    print(f"Equatorial coverage (±30°): {equatorial_coverage} profiles")
    
    # Seasonal coverage
    spring_equinox_coverage = np.sum((doy >= 80) & (doy <= 100))
    fall_equinox_coverage = np.sum((doy >= 260) & (doy <= 280))
    print(f"Spring equinox coverage (DOY 80-100): {spring_equinox_coverage} profiles")
    print(f"Fall equinox coverage (DOY 260-280): {fall_equinox_coverage} profiles")
    
    # Combined coverage
    print(f"Combined pre-reversal coverage: {np.sum(prereversal_condition)} profiles")
    print(f"Coverage percentage: {np.sum(prereversal_condition)/len(electron_density)*100:.2f}%")
    
    return {
        'total_profiles': len(electron_density),
        'prereversal_profiles': np.sum(prereversal_condition),
        'coverage_percentage': np.sum(prereversal_condition)/len(electron_density)*100,
        'hour_coverage': hour_coverage,
        'equatorial_coverage': equatorial_coverage,
        'spring_equinox_coverage': spring_equinox_coverage,
        'fall_equinox_coverage': fall_equinox_coverage
    }

def analyze_radio_occultation_coverage():
    """Analyze radio occultation coverage patterns"""
    print("\n6. Analyzing radio occultation coverage patterns...")
    
    with h5py.File('./data/filtered/electron_density_profiles_2023_with_fits.h5', 'r') as f:
        latitude = f['latitude'][:]
        longitude = f['longitude'][:]
        local_time = f['local_time'][:]
        is_good_fit = f['fit_results/is_good_fit'][:]
        
        # Filter for good fits
        latitude = latitude[is_good_fit]
        longitude = longitude[is_good_fit]
        local_time = local_time[is_good_fit]
    
    local_datetimes = pd.to_datetime(local_time, unit='s')
    hours = local_datetimes.hour
    
    # Radio occultation coverage analysis
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Hourly coverage heatmap
    hour_lat_bins = np.linspace(-90, 90, 19)  # 10° latitude bins
    hour_bins = np.arange(0, 25, 1)
    
    H, xedges, yedges = np.histogram2d(hours, latitude, bins=[hour_bins, hour_lat_bins])
    
    im1 = axes[0, 0].imshow(H.T, aspect='auto', origin='lower', 
                            extent=[0, 24, -90, 90], cmap='viridis')
    axes[0, 0].set_xlabel('Local Hour')
    axes[0, 0].set_ylabel('Latitude (°)')
    axes[0, 0].set_title('Radio Occultation Coverage\n(Hour vs Latitude)')
    plt.colorbar(im1, ax=axes[0, 0], label='Number of Profiles')
    
    # Highlight pre-reversal period
    axes[0, 0].axvspan(17, 20, alpha=0.3, color='red', label='Pre-reversal')
    axes[0, 0].legend()
    
    # 2. Longitudinal coverage
    lon_bins = np.linspace(-180, 180, 37)  # 10° longitude bins
    lon_counts, _ = np.histogram(longitude, bins=lon_bins)
    lon_centers = (lon_bins[:-1] + lon_bins[1:]) / 2
    
    axes[0, 1].bar(lon_centers, lon_counts, width=10, alpha=0.7, color='blue')
    axes[0, 1].set_xlabel('Longitude (°)')
    axes[0, 1].set_ylabel('Number of Profiles')
    axes[0, 1].set_title('Radio Occultation Coverage by Longitude')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Temporal distribution by latitude band
    equatorial_mask = np.abs(latitude) < 30
    midlat_mask = (np.abs(latitude) >= 30) & (np.abs(latitude) < 60)
    highlat_mask = np.abs(latitude) >= 60
    
    axes[1, 0].hist(hours[equatorial_mask], bins=24, alpha=0.7, label='Equatorial (±30°)', density=True)
    axes[1, 0].hist(hours[midlat_mask], bins=24, alpha=0.7, label='Mid-latitude (30-60°)', density=True)
    axes[1, 0].hist(hours[highlat_mask], bins=24, alpha=0.7, label='High-latitude (>60°)', density=True)
    axes[1, 0].set_xlabel('Local Hour')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('Temporal Distribution by Latitude Band')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Highlight pre-reversal period
    axes[1, 0].axvspan(17, 20, alpha=0.3, color='red', label='Pre-reversal')
    
    # 4. Coverage gaps analysis
    # Identify hours with low coverage in equatorial region
    equatorial_hours = hours[equatorial_mask]
    hour_counts_equatorial = np.bincount(equatorial_hours, minlength=24)
    
    axes[1, 1].bar(range(24), hour_counts_equatorial, alpha=0.7, color='green')
    axes[1, 1].set_xlabel('Local Hour')
    axes[1, 1].set_ylabel('Number of Profiles')
    axes[1, 1].set_title('Equatorial Coverage by Hour')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Highlight pre-reversal period
    axes[1, 1].axvspan(17, 20, alpha=0.3, color='red', label='Pre-reversal')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('./data/analysis/prereversal_analysis/radio_occultation_coverage.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Coverage statistics
    print(f"\nRadio Occultation Coverage Statistics:")
    print(f"Equatorial profiles (±30°): {np.sum(equatorial_mask)}")
    print(f"Mid-latitude profiles (30-60°): {np.sum(midlat_mask)}")
    print(f"High-latitude profiles (>60°): {np.sum(highlat_mask)}")
    
    # Pre-reversal coverage in equatorial region
    prereversal_equatorial = (hours >= 17) & (hours <= 20) & equatorial_mask
    print(f"Pre-reversal equatorial profiles: {np.sum(prereversal_equatorial)}")
    print(f"Pre-reversal equatorial coverage: {np.sum(prereversal_equatorial)/np.sum(equatorial_mask)*100:.2f}%")

if __name__ == "__main__":
    print("Analyzing pre-reversal data coverage...")
    results = analyze_data_coverage()
    analyze_radio_occultation_coverage()
    
    print("\nAnalysis complete!")
    print(f"Results saved to: ./data/analysis/prereversal_analysis/")
    
    # Summary
    if results['coverage_percentage'] < 1.0:
        print(f"\n⚠️  WARNING: Very low pre-reversal coverage ({results['coverage_percentage']:.2f}%)")
        print("This suggests the radio occultation data may not capture the pre-reversal behavior.")
        print("Consider:")
        print("1. Using additional data sources (ionosondes, radar)")
        print("2. Implementing physics-based corrections")
        print("3. Using climatological models for pre-reversal behavior")
    else:
        print(f"\n✅ Good pre-reversal coverage found ({results['coverage_percentage']:.2f}%)")
        print("The data should be sufficient for modeling pre-reversal behavior.") 
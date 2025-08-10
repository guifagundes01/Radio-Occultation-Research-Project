import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from datetime import datetime
import os

def analyze_hmf2_prereversal():
    """Analyze pre-reversal behavior in the entire dataset (not just São Luís)"""
    
    # Load data
    print("Loading data...")
    with h5py.File('./data/filtered/electron_density_profiles_2023_with_fits.h5', 'r') as f:
        latitude = f['latitude'][:]
        longitude = f['longitude'][:]
        local_time = f['local_time'][:]
        electron_density = f['electron_density'][:]
        altitude = f['altitude'][:]
        is_good_fit = f['fit_results/is_good_fit'][:]
        
        # Filter for good fits
        latitude = latitude[is_good_fit]
        longitude = longitude[is_good_fit]
        local_time = local_time[is_good_fit]
        electron_density = electron_density[is_good_fit]
    
    # Convert local time to datetime
    local_datetimes = pd.to_datetime(local_time, unit='s')
    
    # Use the entire dataset
    all_lat = latitude
    all_lon = longitude
    all_datetimes = local_datetimes
    all_profiles = electron_density
    
    print(f"Found {len(all_profiles)} profiles in the dataset")
    
    if len(all_profiles) == 0:
        print("No profiles found in the dataset.")
        return
    
    # Calculate hmF2 for all profiles
    all_hmf2 = []
    all_nmf2 = []
    all_hours = []
    all_months = []
    all_doy = []
    
    for i, profile in enumerate(all_profiles):
        nmf2 = np.max(profile)
        hmf2 = altitude[np.argmax(profile)]
        hour = all_datetimes[i].hour
        month = all_datetimes[i].month
        doy = all_datetimes[i].dayofyear
        
        all_hmf2.append(hmf2)
        all_nmf2.append(nmf2)
        all_hours.append(hour)
        all_months.append(month)
        all_doy.append(doy)
    
    all_hmf2 = np.array(all_hmf2)
    all_nmf2 = np.array(all_nmf2)
    all_hours = np.array(all_hours)
    all_months = np.array(all_months)
    all_doy = np.array(all_doy)
    
    # Create output directory
    os.makedirs('./data/analysis/hmf2_analysis', exist_ok=True)
    
    # 1. Overall hmF2 vs Hour plot
    print("\n1. Creating overall hmF2 vs Hour plot...")
    
    plt.figure(figsize=(12, 8))
    plt.scatter(all_hours, all_hmf2, alpha=0.6, s=30)
    plt.xlabel('Hour')
    plt.ylabel('hmF2 (km)')
    plt.title('hmF2 vs Hour')
    # Highlight pre-reversal period
    plt.axvspan(17, 20, alpha=0.3, color='red', label='Pre-reversal')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('./data/analysis/hmf2_analysis/hmf2_vs_hour_overall.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Monthly plots
    print("\n2. Creating monthly hmF2 vs Hour plots...")
    
    # Get unique months
    unique_months = np.unique(all_months)
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    # Create subplots for each month
    n_months = len(unique_months)
    cols = 3
    rows = (n_months + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, month in enumerate(unique_months):
        row = i // cols
        col = i % cols
        
        # Filter data for this month
        month_mask = all_months == month
        month_hours = all_hours[month_mask]
        month_hmf2 = all_hmf2[month_mask]
        
        # Plot
        axes[row, col].scatter(month_hours, month_hmf2, alpha=0.7, s=40)
        axes[row, col].set_xlabel('Hour')
        axes[row, col].set_ylabel('hmF2 (km)')
        axes[row, col].set_title(f'{month_names[month-1]} ({len(month_hours)})')
        # Highlight pre-reversal period
        axes[row, col].axvspan(17, 20, alpha=0.3, color='red', label='Pre-reversal')
        
        # Calculate and display statistics
        if len(month_hours) > 0:
            prereversal_mask = (month_hours >= 17) & (month_hours <= 20)
            if np.sum(prereversal_mask) > 0:
                prereversal_hmf2 = month_hmf2[prereversal_mask]
                other_hmf2 = month_hmf2[~prereversal_mask]
                if len(other_hmf2) > 0:
                    hmf2_diff = np.mean(prereversal_hmf2) - np.mean(other_hmf2)
                    axes[row, col].text(0.05, 0.95, f'ΔhmF2: {hmf2_diff:.1f} km', 
                                      transform=axes[row, col].transAxes, 
                                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Hide empty subplots
    for i in range(n_months, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('./data/analysis/hmf2_analysis/hmf2_vs_hour_by_month.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Seasonal analysis
    print("\n3. Creating seasonal analysis...")
    
    # Define seasons
    spring_months = [3, 4, 5]  # March, April, May
    summer_months = [6, 7, 8]  # June, July, August
    fall_months = [9, 10, 11]  # September, October, November
    winter_months = [12, 1, 2]  # December, January, February
    
    seasons = {
        'Spring': spring_months,
        'Summer': summer_months,
        'Fall': fall_months,
        'Winter': winter_months
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, (season_name, season_months) in enumerate(seasons.items()):
        # Filter data for this season
        season_mask = np.isin(all_months, season_months)
        season_hours = all_hours[season_mask]
        season_hmf2 = all_hmf2[season_mask]
        
        if len(season_hours) > 0:
            # Plot
            axes[i].scatter(season_hours, season_hmf2, alpha=0.7, s=40)
            axes[i].set_xlabel('Hour')
            axes[i].set_ylabel('hmF2 (km)')
            axes[i].set_title(f'{season_name}')
            # Highlight pre-reversal period
            axes[i].axvspan(17, 20, alpha=0.3, color='red', label='Pre-reversal')
            
            # Calculate statistics
            prereversal_mask = (season_hours >= 17) & (season_hours <= 20)
            if np.sum(prereversal_mask) > 0:
                prereversal_hmf2 = season_hmf2[prereversal_mask]
                other_hmf2 = season_hmf2[~prereversal_mask]
                if len(other_hmf2) > 0:
                    hmf2_diff = np.mean(prereversal_hmf2) - np.mean(other_hmf2)
                    axes[i].text(0.05, 0.95, f'ΔhmF2: {hmf2_diff:.1f} km', 
                               transform=axes[i].transAxes, 
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('./data/analysis/hmf2_analysis/hmf2_vs_hour_by_season.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 4. Statistical analysis
    print("\n4. Performing statistical analysis...")
    
    # Calculate statistics for each month
    monthly_stats = []
    
    for month in unique_months:
        month_mask = all_months == month
        month_hours = all_hours[month_mask]
        month_hmf2 = all_hmf2[month_mask]
        
        if len(month_hours) > 0:
            # Pre-reversal statistics
            prereversal_mask = (month_hours >= 17) & (month_hours <= 20)
            other_mask = ~prereversal_mask
            
            prereversal_hmf2 = month_hmf2[prereversal_mask]
            other_hmf2 = month_hmf2[other_mask]
            
            stats = {
                'month': month,
                'month_name': month_names[month-1],
                'total_profiles': len(month_hours),
                'prereversal_profiles': len(prereversal_hmf2),
                'other_profiles': len(other_hmf2),
                'mean_hmf2_prereversal': np.mean(prereversal_hmf2) if len(prereversal_hmf2) > 0 else np.nan,
                'mean_hmf2_other': np.mean(other_hmf2) if len(other_hmf2) > 0 else np.nan,
                'hmf2_difference': np.mean(prereversal_hmf2) - np.mean(other_hmf2) if len(prereversal_hmf2) > 0 and len(other_hmf2) > 0 else np.nan
            }
            monthly_stats.append(stats)
    
    # Create summary table
    stats_df = pd.DataFrame(monthly_stats)
    print("\nMonthly Statistics:")
    print("=" * 80)
    print(f"{'Month':<8} {'Total':<6} {'Pre-rev':<7} {'Other':<6} {'hmF2 Pre':<8} {'hmF2 Other':<10} {'ΔhmF2':<6}")
    print("-" * 80)
    
    for _, row in stats_df.iterrows():
        print(f"{row['month_name']:<8} {row['total_profiles']:<6} {row['prereversal_profiles']:<7} "
              f"{row['other_profiles']:<6} {row['mean_hmf2_prereversal']:<8.1f} "
              f"{row['mean_hmf2_other']:<10.1f} {row['hmf2_difference']:<6.1f}")
    
    # Save statistics
    stats_df.to_csv('./data/analysis/hmf2_analysis/monthly_statistics.csv', index=False)
    
    # 5. Pre-reversal detection summary
    print("\n5. Pre-reversal behavior summary:")
    print("=" * 50)
    
    # Overall statistics
    prereversal_mask = (all_hours >= 17) & (all_hours <= 20)
    other_mask = ~prereversal_mask
    
    prereversal_hmf2 = all_hmf2[prereversal_mask]
    other_hmf2 = all_hmf2[other_mask]
    
    if len(prereversal_hmf2) > 0 and len(other_hmf2) > 0:
        overall_diff = np.mean(prereversal_hmf2) - np.mean(other_hmf2)
        print(f"Overall hmF2 difference (pre-reversal vs other): {overall_diff:.1f} km")
        
        if overall_diff > 10:
            print("✅ STRONG pre-reversal behavior detected!")
        elif overall_diff > 5:
            print("⚠️  MODERATE pre-reversal behavior detected")
        elif overall_diff > 0:
            print("🔍 WEAK pre-reversal behavior detected")
        else:
            print("❌ No pre-reversal behavior detected")
    else:
        print("❌ Insufficient data for pre-reversal analysis")
    
    print(f"\nData coverage:")
    print(f"Total profiles: {len(all_profiles)}")
    print(f"Pre-reversal profiles (17-20h): {len(prereversal_hmf2)}")
    print(f"Other profiles: {len(other_hmf2)}")
    
    return stats_df

def analyze_location_prereversal(
    center_lat=-2.5297, center_lon=-44.3028, lat_margin=5.0, lon_margin=5.0,
    output_dir_base='./data/analysis/location_analysis'):
    """Analyze pre-reversal behavior in a specific region (default: São Luís)"""
    
    print(f"Analyzing data near location ({center_lat:.2f}°, {center_lon:.2f}°)")
    print(f"Using margins: ±{lat_margin}° latitude, ±{lon_margin}° longitude")
    
    # Load data
    print("Loading data...")
    with h5py.File('./data/filtered/electron_density_profiles_2023_with_fits.h5', 'r') as f:
        latitude = f['latitude'][:]
        longitude = f['longitude'][:]
        local_time = f['local_time'][:]
        electron_density = f['electron_density'][:]
        altitude = f['altitude'][:]
        is_good_fit = f['fit_results/is_good_fit'][:]
        
        # Filter for good fits
        latitude = latitude[is_good_fit]
        longitude = longitude[is_good_fit]
        local_time = local_time[is_good_fit]
        electron_density = electron_density[is_good_fit]
    
    # Convert local time to datetime
    local_datetimes = pd.to_datetime(local_time, unit='s')
    
    # Filter data for the specified region
    lat_mask = (latitude >= center_lat - lat_margin) & (latitude <= center_lat + lat_margin)
    lon_mask = (longitude >= center_lon - lon_margin) & (longitude <= center_lon + lon_margin)
    region_mask = lat_mask & lon_mask
    
    region_lat = latitude[region_mask]
    region_lon = longitude[region_mask]
    region_datetimes = local_datetimes[region_mask]
    region_profiles = electron_density[region_mask]
    
    print(f"Found {len(region_profiles)} profiles in the selected region")
    
    if len(region_profiles) == 0:
        print("No profiles found in the selected region. Try increasing the margins.")
        return
    
    # Calculate hmF2 for all region profiles
    region_hmf2 = []
    region_nmf2 = []
    region_hours = []
    region_months = []
    region_doy = []
    
    for i, profile in enumerate(region_profiles):
        nmf2 = np.max(profile)
        hmf2 = altitude[np.argmax(profile)]
        hour = region_datetimes[i].hour
        month = region_datetimes[i].month
        doy = region_datetimes[i].dayofyear
        
        region_hmf2.append(hmf2)
        region_nmf2.append(nmf2)
        region_hours.append(hour)
        region_months.append(month)
        region_doy.append(doy)
    
    region_hmf2 = np.array(region_hmf2)
    region_nmf2 = np.array(region_nmf2)
    region_hours = np.array(region_hours)
    region_months = np.array(region_months)
    region_doy = np.array(region_doy)
    
    # Create output directory
    output_dir = os.path.join(output_dir_base, f"lat_{center_lat:.2f}_lon_{center_lon:.2f}_latm_{lat_margin}_lonm_{lon_margin}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Overall hmF2 vs Hour plot
    print("\n1. Creating overall hmF2 vs Hour plot...")
    
    plt.figure(figsize=(12, 8))
    plt.scatter(region_hours, region_hmf2, alpha=0.6, s=30)
    plt.xlabel('Hour')
    plt.ylabel('hmF2 (km)')
    plt.title('hmF2 vs Hour')
    # Highlight pre-reversal period
    plt.axvspan(17, 20, alpha=0.3, color='red', label='Pre-reversal')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'hmf2_vs_hour_overall.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Monthly plots
    print("\n2. Creating monthly hmF2 vs Hour plots...")
    
    # Get unique months
    unique_months = np.unique(region_months)
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    # Create subplots for each month
    n_months = len(unique_months)
    cols = 3
    rows = (n_months + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, month in enumerate(unique_months):
        row = i // cols
        col = i % cols
        
        # Filter data for this month
        month_mask = region_months == month
        month_hours = region_hours[month_mask]
        month_hmf2 = region_hmf2[month_mask]
        
        # Plot
        axes[row, col].scatter(month_hours, month_hmf2, alpha=0.7, s=40)
        axes[row, col].set_xlabel('Hour')
        axes[row, col].set_ylabel('hmF2 (km)')
        axes[row, col].set_title(f'{month_names[month-1]} ({len(month_hours)})')
        # Highlight pre-reversal period
        axes[row, col].axvspan(17, 20, alpha=0.3, color='red', label='Pre-reversal')
        
        # Calculate and display statistics
        if len(month_hours) > 0:
            prereversal_mask = (month_hours >= 17) & (month_hours <= 20)
            if np.sum(prereversal_mask) > 0:
                prereversal_hmf2 = month_hmf2[prereversal_mask]
                other_hmf2 = month_hmf2[~prereversal_mask]
                if len(other_hmf2) > 0:
                    hmf2_diff = np.mean(prereversal_hmf2) - np.mean(other_hmf2)
                    axes[row, col].text(0.05, 0.95, f'ΔhmF2: {hmf2_diff:.1f} km', 
                                      transform=axes[row, col].transAxes, 
                                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Hide empty subplots
    for i in range(n_months, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'hmf2_vs_hour_by_month.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Seasonal analysis
    print("\n3. Creating seasonal analysis...")
    
    # Define seasons
    spring_months = [3, 4, 5]  # March, April, May
    summer_months = [6, 7, 8]  # June, July, August
    fall_months = [9, 10, 11]  # September, October, November
    winter_months = [12, 1, 2]  # December, January, February
    
    seasons = {
        'Spring': spring_months,
        'Summer': summer_months,
        'Fall': fall_months,
        'Winter': winter_months
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, (season_name, season_months) in enumerate(seasons.items()):
        # Filter data for this season
        season_mask = np.isin(region_months, season_months)
        season_hours = region_hours[season_mask]
        season_hmf2 = region_hmf2[season_mask]
        
        if len(season_hours) > 0:
            # Plot
            axes[i].scatter(season_hours, season_hmf2, alpha=0.7, s=40)
            axes[i].set_xlabel('Hour')
            axes[i].set_ylabel('hmF2 (km)')
            axes[i].set_title(f'{season_name}')
            # Highlight pre-reversal period
            axes[i].axvspan(17, 20, alpha=0.3, color='red', label='Pre-reversal')
            
            # Calculate statistics
            prereversal_mask = (season_hours >= 17) & (season_hours <= 20)
            if np.sum(prereversal_mask) > 0:
                prereversal_hmf2 = season_hmf2[prereversal_mask]
                other_hmf2 = season_hmf2[~prereversal_mask]
                if len(other_hmf2) > 0:
                    hmf2_diff = np.mean(prereversal_hmf2) - np.mean(other_hmf2)
                    axes[i].text(0.05, 0.95, f'ΔhmF2: {hmf2_diff:.1f} km', 
                               transform=axes[i].transAxes, 
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'hmf2_vs_hour_by_season.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 4. Statistical analysis
    print("\n4. Performing statistical analysis...")
    
    # Calculate statistics for each month
    monthly_stats = []
    
    for month in unique_months:
        month_mask = region_months == month
        month_hours = region_hours[month_mask]
        month_hmf2 = region_hmf2[month_mask]
        
        if len(month_hours) > 0:
            # Pre-reversal statistics
            prereversal_mask = (month_hours >= 17) & (month_hours <= 20)
            other_mask = ~prereversal_mask
            
            prereversal_hmf2 = month_hmf2[prereversal_mask]
            other_hmf2 = month_hmf2[other_mask]
            
            stats = {
                'month': month,
                'month_name': month_names[month-1],
                'total_profiles': len(month_hours),
                'prereversal_profiles': len(prereversal_hmf2),
                'other_profiles': len(other_hmf2),
                'mean_hmf2_prereversal': np.mean(prereversal_hmf2) if len(prereversal_hmf2) > 0 else np.nan,
                'mean_hmf2_other': np.mean(other_hmf2) if len(other_hmf2) > 0 else np.nan,
                'hmf2_difference': np.mean(prereversal_hmf2) - np.mean(other_hmf2) if len(prereversal_hmf2) > 0 and len(other_hmf2) > 0 else np.nan
            }
            monthly_stats.append(stats)
    
    # Create summary table
    stats_df = pd.DataFrame(monthly_stats)
    print("\nMonthly Statistics:")
    print("=" * 80)
    print(f"{'Month':<8} {'Total':<6} {'Pre-rev':<7} {'Other':<6} {'hmF2 Pre':<8} {'hmF2 Other':<10} {'ΔhmF2':<6}")
    print("-" * 80)
    
    for _, row in stats_df.iterrows():
        print(f"{row['month_name']:<8} {row['total_profiles']:<6} {row['prereversal_profiles']:<7} "
              f"{row['other_profiles']:<6} {row['mean_hmf2_prereversal']:<8.1f} "
              f"{row['mean_hmf2_other']:<10.1f} {row['hmf2_difference']:<6.1f}")
    
    # Save statistics
    stats_df.to_csv(os.path.join(output_dir, 'monthly_statistics.csv'), index=False)
    
    # 5. Pre-reversal detection summary
    print("\n5. Pre-reversal behavior summary:")
    print("=" * 50)
    
    # Overall statistics
    prereversal_mask = (region_hours >= 17) & (region_hours <= 20)
    other_mask = ~prereversal_mask
    
    prereversal_hmf2 = region_hmf2[prereversal_mask]
    other_hmf2 = region_hmf2[other_mask]
    
    if len(prereversal_hmf2) > 0 and len(other_hmf2) > 0:
        overall_diff = np.mean(prereversal_hmf2) - np.mean(other_hmf2)
        print(f"Overall hmF2 difference (pre-reversal vs other): {overall_diff:.1f} km")
        
        if overall_diff > 10:
            print("✅ STRONG pre-reversal behavior detected!")
        elif overall_diff > 5:
            print("⚠️  MODERATE pre-reversal behavior detected")
        elif overall_diff > 0:
            print("🔍 WEAK pre-reversal behavior detected")
        else:
            print("❌ No pre-reversal behavior detected")
    else:
        print("❌ Insufficient data for pre-reversal analysis")
    
    print(f"\nData coverage:")
    print(f"Total profiles: {len(region_profiles)}")
    print(f"Pre-reversal profiles (17-20h): {len(prereversal_hmf2)}")
    print(f"Other profiles: {len(other_hmf2)}")
    
    return stats_df

if __name__ == "__main__":
    print("Analyzing pre-reversal behavior in a specific region (default: São Luís)...")
    stats = analyze_location_prereversal()
    print(f"\nAnalysis complete! Results saved to: ./data/analysis/location_analysis/") 
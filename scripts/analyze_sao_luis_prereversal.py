import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from datetime import datetime
import os

def analyze_sao_luis_prereversal():
    """Analyze pre-reversal behavior in São Luís data"""
    
    # São Luís coordinates
    SAO_LUIS_LAT = -2.5297  # São Luís latitude
    SAO_LUIS_LON = -44.3028  # São Luís longitude
    
    # Define margin for data selection (in degrees)
    LAT_MARGIN = 5.0  # ±5° latitude
    LON_MARGIN = 5.0  # ±5° longitude
    
    print(f"Analyzing data near São Luís ({SAO_LUIS_LAT:.2f}°, {SAO_LUIS_LON:.2f}°)")
    print(f"Using margins: ±{LAT_MARGIN}° latitude, ±{LON_MARGIN}° longitude")
    
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
    
    # Filter data for São Luís region
    lat_mask = (latitude >= SAO_LUIS_LAT - LAT_MARGIN) & (latitude <= SAO_LUIS_LAT + LAT_MARGIN)
    lon_mask = (longitude >= SAO_LUIS_LON - LON_MARGIN) & (longitude <= SAO_LUIS_LON + LON_MARGIN)
    region_mask = lat_mask & lon_mask
    
    # Extract São Luís region data
    sao_luis_lat = latitude[region_mask]
    sao_luis_lon = longitude[region_mask]
    sao_luis_datetimes = local_datetimes[region_mask]
    sao_luis_profiles = electron_density[region_mask]
    
    print(f"Found {len(sao_luis_profiles)} profiles in São Luís region")
    
    if len(sao_luis_profiles) == 0:
        print("No profiles found in São Luís region. Try increasing the margins.")
        return
    
    # Calculate hmF2 for all São Luís profiles
    sao_luis_hmf2 = []
    sao_luis_nmf2 = []
    sao_luis_hours = []
    sao_luis_months = []
    sao_luis_doy = []
    
    for i, profile in enumerate(sao_luis_profiles):
        nmf2 = np.max(profile)
        hmf2 = altitude[np.argmax(profile)]
        hour = sao_luis_datetimes[i].hour
        month = sao_luis_datetimes[i].month
        doy = sao_luis_datetimes[i].dayofyear
        
        sao_luis_hmf2.append(hmf2)
        sao_luis_nmf2.append(nmf2)
        sao_luis_hours.append(hour)
        sao_luis_months.append(month)
        sao_luis_doy.append(doy)
    
    sao_luis_hmf2 = np.array(sao_luis_hmf2)
    sao_luis_nmf2 = np.array(sao_luis_nmf2)
    sao_luis_hours = np.array(sao_luis_hours)
    sao_luis_months = np.array(sao_luis_months)
    sao_luis_doy = np.array(sao_luis_doy)
    
    # Create output directory
    os.makedirs('./data/analysis/sao_luis_analysis', exist_ok=True)
    
    # 1. Overall hmF2 vs Hour plot
    print("\n1. Creating overall hmF2 vs Hour plot...")
    
    plt.figure(figsize=(12, 8))
    plt.scatter(sao_luis_hours, sao_luis_hmf2, alpha=0.6, s=30, c=sao_luis_months, cmap='viridis')
    plt.colorbar(label='Month')
    plt.xlabel('Local Hour')
    plt.ylabel('hmF2 (km)')
    plt.title(f'São Luís - hmF2 vs Local Hour\n({len(sao_luis_profiles)} profiles, ±{LAT_MARGIN}° lat, ±{LON_MARGIN}° lon)')
    plt.grid(True, alpha=0.3)
    
    # Highlight pre-reversal period
    plt.axvspan(17, 20, alpha=0.3, color='red', label='Pre-reversal period (17-20h)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('./data/analysis/sao_luis_analysis/hmf2_vs_hour_overall.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Monthly plots
    print("\n2. Creating monthly hmF2 vs Hour plots...")
    
    # Get unique months
    unique_months = np.unique(sao_luis_months)
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
        month_mask = sao_luis_months == month
        month_hours = sao_luis_hours[month_mask]
        month_hmf2 = sao_luis_hmf2[month_mask]
        month_nmf2 = sao_luis_nmf2[month_mask]
        
        # Plot
        scatter = axes[row, col].scatter(month_hours, month_hmf2, 
                                       c=month_nmf2, cmap='viridis', 
                                       alpha=0.7, s=40)
        axes[row, col].set_xlabel('Local Hour')
        axes[row, col].set_ylabel('hmF2 (km)')
        axes[row, col].set_title(f'{month_names[month-1]} ({len(month_hours)} profiles)')
        axes[row, col].grid(True, alpha=0.3)
        axes[row, col].set_xlim(0, 23)
        
        # Highlight pre-reversal period
        axes[row, col].axvspan(17, 20, alpha=0.3, color='red', label='Pre-reversal')
        
        # Add colorbar for NmF2
        cbar = plt.colorbar(scatter, ax=axes[row, col])
        cbar.set_label('NmF2 (cm⁻³)')
        
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
    plt.savefig('./data/analysis/sao_luis_analysis/hmf2_vs_hour_by_month.png', dpi=300, bbox_inches='tight')
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
        season_mask = np.isin(sao_luis_months, season_months)
        season_hours = sao_luis_hours[season_mask]
        season_hmf2 = sao_luis_hmf2[season_mask]
        season_nmf2 = sao_luis_nmf2[season_mask]
        
        if len(season_hours) > 0:
            # Plot
            scatter = axes[i].scatter(season_hours, season_hmf2, 
                                    c=season_nmf2, cmap='viridis', 
                                    alpha=0.7, s=40)
            axes[i].set_xlabel('Local Hour')
            axes[i].set_ylabel('hmF2 (km)')
            axes[i].set_title(f'{season_name} ({len(season_hours)} profiles)')
            axes[i].grid(True, alpha=0.3)
            axes[i].set_xlim(0, 23)
            
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
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=axes[i])
            cbar.set_label('NmF2 (cm⁻³)')
    
    plt.tight_layout()
    plt.savefig('./data/analysis/sao_luis_analysis/hmf2_vs_hour_by_season.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 4. Statistical analysis
    print("\n4. Performing statistical analysis...")
    
    # Calculate statistics for each month
    monthly_stats = []
    
    for month in unique_months:
        month_mask = sao_luis_months == month
        month_hours = sao_luis_hours[month_mask]
        month_hmf2 = sao_luis_hmf2[month_mask]
        
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
    stats_df.to_csv('./data/analysis/sao_luis_analysis/monthly_statistics.csv', index=False)
    
    # 5. Pre-reversal detection summary
    print("\n5. Pre-reversal behavior summary:")
    print("=" * 50)
    
    # Overall statistics
    prereversal_mask = (sao_luis_hours >= 17) & (sao_luis_hours <= 20)
    other_mask = ~prereversal_mask
    
    prereversal_hmf2 = sao_luis_hmf2[prereversal_mask]
    other_hmf2 = sao_luis_hmf2[other_mask]
    
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
    print(f"Total profiles: {len(sao_luis_profiles)}")
    print(f"Pre-reversal profiles (17-20h): {len(prereversal_hmf2)}")
    print(f"Other profiles: {len(other_hmf2)}")
    
    return stats_df

if __name__ == "__main__":
    print("Analyzing pre-reversal behavior in São Luís data...")
    stats = analyze_sao_luis_prereversal()
    print(f"\nAnalysis complete! Results saved to: ./data/analysis/sao_luis_analysis/") 
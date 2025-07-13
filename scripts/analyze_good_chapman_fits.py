import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from os import path
from scipy import stats
import h5py

def chapmanF2F1(x, Nmax, hmax, H, a1, Nm, a2, hm):
    """Chapman function for F2 and F1 layers"""
    z = (x - hmax) / H
    zf1 = (x - hm) / H
    F2 = Nmax * np.exp(1 - z - np.exp(-z))
    F1 = Nm * np.exp(1 - zf1 - np.exp(-zf1))
    return F2 + a1 * F1

def load_data():
    """Load the filtered data with good fits"""
    file_path = "./data/filtered/electron_density_profiles_2023_with_fits.h5"
    with h5py.File(file_path, 'r') as f:
        # Get base features
        latitude = f['latitude'][:]
        longitude = f['longitude'][:]
        local_time = f['local_time'][:]
        f107 = f['f107'][:]
        kp = f['kp'][:]
        dip = f['dip'][:]
        electron_density = f['electron_density'][:]
        altitude = f['altitude'][:]  # This is a single array of height points
        
        # Get Chapman parameters
        chapman_params = f['fit_results/chapman_params'][:]
        
        # Only use good fits
        is_good_fit = f['fit_results/is_good_fit'][:]
        latitude = latitude[is_good_fit]
        longitude = longitude[is_good_fit]
        local_time = local_time[is_good_fit]
        f107 = f107[is_good_fit]
        kp = kp[is_good_fit]
        dip = dip[is_good_fit]
        electron_density = electron_density[is_good_fit]
        chapman_params = chapman_params[is_good_fit]
        
        # Calculate MSE for each fit
        mse = np.zeros(len(chapman_params))
        for i in range(len(chapman_params)):
            y_fit = chapmanF2F1(altitude, *chapman_params[i])
            mse[i] = np.mean((electron_density[i] - y_fit) ** 2)
    
    return (chapman_params, electron_density, altitude, 
            latitude, longitude, local_time, f107, kp, dip, mse)

def analyze_good_fit_parameters(chapman_params):
    """Analyze parameters of good fits"""
    param_names = ['Nmax', 'hmax', 'H', 'a1', 'Nm', 'a2', 'hm']
    
    print("\nGood Fit Parameters Statistics:")
    print("-" * 40)
    for i, name in enumerate(param_names):
        values = chapman_params[:, i]
        print(f"{name}:")
        print(f"  Mean: {np.mean(values):.2e}")
        print(f"  Std:  {np.std(values):.2e}")
        print(f"  Min:  {np.min(values):.2e}")
        print(f"  Max:  {np.max(values):.2e}")
        print()

def plot_good_fit_parameters(chapman_params):
    """Plot parameter distributions for good fits"""
    param_names = ['Nmax', 'hmax', 'H', 'a1', 'Nm', 'a2', 'hm']
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('Good Chapman Fit Parameter Distributions')
    
    for i, (ax, name) in enumerate(zip(axes.flat, param_names)):
        if i < len(param_names):
            sns.histplot(chapman_params[:, i], ax=ax)
            ax.set_title(name)
    
    plt.tight_layout()
    plt.savefig('./data/fit_results/good_chapman_params_dist.png')
    plt.show()
    plt.close()

def plot_good_fit_correlations(chapman_params):
    """Plot correlation matrix for good fit parameters"""
    corr_matrix = np.corrcoef(chapman_params.T)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix,
                xticklabels=['Nmax', 'hmax', 'H', 'a1', 'Nm', 'a2', 'hm'],
                yticklabels=['Nmax', 'hmax', 'H', 'a1', 'Nm', 'a2', 'hm'],
                annot=True, cmap='coolwarm')
    plt.title('Good Chapman Fit Parameters Correlation Matrix')
    plt.tight_layout()
    plt.savefig('./data/fit_results/good_chapman_correlations.png')
    plt.show()
    plt.close()

def plot_representative_good_fits(chapman_params, electron_density, altitude, n_samples=10):
    """Plot representative good fits"""
    # Select random samples from good fits
    n_profiles = len(chapman_params)
    selected_indices = np.random.choice(n_profiles, n_samples, replace=False)
    
    # Calculate number of images needed (10 fits per image)
    n_images = (n_samples + 9) // 10  # Ceiling division
    
    for img_idx in range(n_images):
        start_idx = img_idx * 10
        end_idx = min((img_idx + 1) * 10, n_samples)
        current_indices = selected_indices[start_idx:end_idx]
        
        fig, axes = plt.subplots(len(current_indices), 1, figsize=(12, 4*len(current_indices)))
        fig.suptitle(f'Representative Good Chapman Fits (Set {img_idx + 1})')
        
        for i, idx in enumerate(current_indices):
            ax = axes[i]
            # Plot original data
            ax.scatter(electron_density[idx], altitude, label='Data', alpha=0.6, s=20)
            
            # Plot Chapman fit
            params = chapman_params[idx]
            y_chapman = chapmanF2F1(altitude, *params)
            ax.plot(y_chapman, altitude, 'r-', label='Chapman fit')
            
            ax.set_title(f'Profile {idx}')
            ax.set_xlabel('Electron Density (cm⁻³)')
            ax.set_ylabel('Altitude (km)')
            ax.legend()
            ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'./data/fit_results/representative_good_fits_set_{img_idx + 1}.png')
        plt.show()
        plt.close()

def plot_parameter_vs_features(chapman_params, latitude, longitude, local_time, f107, kp, dip):
    """Plot Chapman parameters vs various features"""
    param_names = ['Nmax', 'hmax', 'H', 'a1', 'Nm', 'a2', 'hm']
    feature_names = ['Latitude', 'Longitude', 'Local Time', 'F10.7', 'Kp', 'Dip']
    features = [latitude, longitude, local_time, f107, kp, dip]
    
    for i, param_name in enumerate(param_names):
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'{param_name} vs Features')
        
        for j, (ax, feature_name, feature) in enumerate(zip(axes.flat, feature_names, features)):
            ax.scatter(feature, chapman_params[:, i], alpha=0.5)
            ax.set_xlabel(feature_name)
            ax.set_ylabel(param_name)
            ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'./data/fit_results/{param_name}_vs_features.png')
        plt.show()
        plt.close()

def plot_best_worst_fits(chapman_params, electron_density, altitude, mse, n_samples=5):
    """Plot the n best and n worst fits based on MSE in a 2x5 grid."""
    # Get indices of best and worst fits
    best_indices = np.argsort(mse)[:n_samples]
    worst_indices = np.argsort(mse)[-n_samples:][::-1]  # Reverse to get worst first

    fig, axes = plt.subplots(2, n_samples, figsize=(4*n_samples, 8), sharey=True)
    fig.suptitle('Best (Top) and Worst (Bottom) Chapman Fits (MSE)')

    # Plot best fits (top row)
    for i, idx in enumerate(best_indices):
        ax = axes[0, i]
        ax.scatter(electron_density[idx], altitude, label='Data', alpha=0.6, s=20)
        params = chapman_params[idx]
        y_chapman = chapmanF2F1(altitude, *params)
        ax.plot(y_chapman, altitude, 'r-', label='Chapman fit')
        ax.set_title(f'Best {i+1}\nMSE: {mse[idx]:.2e}')
        if i == 0:
            ax.set_ylabel('Altitude (km)')
            ax.legend()
        ax.grid(True)

    # Plot worst fits (bottom row)
    for i, idx in enumerate(worst_indices):
        ax = axes[1, i]
        ax.scatter(electron_density[idx], altitude, label='Data', alpha=0.6, s=20)
        params = chapman_params[idx]
        y_chapman = chapmanF2F1(altitude, *params)
        ax.plot(y_chapman, altitude, 'r-', label='Chapman fit')
        ax.set_title(f'Worst {i+1}\nMSE: {mse[idx]:.2e}')
        if i == 0:
            ax.set_ylabel('Altitude (km)')
            ax.legend()
        ax.grid(True)

    for ax in axes[1, :]:
        ax.set_xlabel('Electron Density (cm⁻³)')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('./data/fit_results/best_worst_chapman_fits_grid.png')
    plt.show()
    plt.close()

def main():
    # Create analysis directory if it doesn't exist
    if not path.exists('./data/fit_results'):
        print("Error: Fit results directory not found!")
        exit(1)
    
    # Load data
    (chapman_params, electron_density, altitude,
     latitude, longitude, local_time, f107, kp, dip, mse) = load_data()
    
    print(f"\nAnalyzing {len(chapman_params)} good fits")
    
    # Run analyses
    analyze_good_fit_parameters(chapman_params)
    
    # Create visualizations
    plot_good_fit_parameters(chapman_params)
    plot_good_fit_correlations(chapman_params)
    plot_representative_good_fits(chapman_params, electron_density, altitude)
    plot_best_worst_fits(chapman_params, electron_density, altitude, mse)
    # plot_parameter_vs_features(chapman_params, latitude, longitude, local_time, f107, kp, dip)
    
    print("\nAnalysis complete! Plots have been saved in the data/fit_results directory.")

if __name__ == "__main__":
    main() 
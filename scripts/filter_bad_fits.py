import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.stats import zscore

def chapmanF2F1(x, Nmax, hmax, H, a1, Nm, a2, hm):
    """Chapman function for F2 and F1 layers"""
    z = (x - hmax) / H
    zf1 = (x - hm) / H
    F2 = Nmax * np.exp(1 - z - np.exp(-z))
    F1 = Nm * np.exp(1 - zf1 - np.exp(-zf1))
    return F2 + a1 * F1

def calculate_fit_errors(electron_density, altitude, chapman_params):
    """Calculate MSE between fitted curves and original data"""
    errors = []
    for i in range(len(electron_density)):
        # Get fitted curve
        params = chapman_params[i]
        fitted_curve = chapmanF2F1(altitude, *params)
        
        # Calculate MSE
        mse = np.mean((fitted_curve - electron_density[i])**2)
        errors.append(mse)
    
    return np.array(errors)

def plot_error_distribution(errors):
    """Plot the distribution of fitting errors"""
    plt.figure(figsize=(10, 6))
    plt.hist(np.log10(errors), bins=50)
    plt.xlabel('Log10(MSE)')
    plt.ylabel('Count')
    plt.title('Distribution of Fitting Errors')
    plt.grid(True)
    plt.savefig('./data/fit_results/error_distribution.png')
    plt.close()

def plot_worst_fits(electron_density, altitude, chapman_params, errors, n_worst=5):
    """Plot the n worst fits"""
    worst_indices = np.argsort(errors)[-n_worst:]
    
    fig, axes = plt.subplots(n_worst, 1, figsize=(12, 4*n_worst))
    if n_worst == 1:
        axes = [axes]
    
    for i, idx in enumerate(worst_indices):
        ax = axes[i]
        # Plot original data
        ax.scatter(electron_density[idx], altitude, label='Data', alpha=0.6, s=20)
        
        # Plot fitted curve
        params = chapman_params[idx]
        fitted_curve = chapmanF2F1(altitude, *params)
        ax.plot(fitted_curve, altitude, 'r-', label='Fitted curve')
        
        ax.set_title(f'Profile {idx} - MSE: {errors[idx]:.2e}')
        ax.set_xlabel('Electron Density (cm⁻³)')
        ax.set_ylabel('Altitude (km)')
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('./data/fit_results/worst_fits.png')
    plt.close()

def main():
    # Load data
    with h5py.File("./data/transformed/electron_density_profiles_2023.h5", 'r') as f:
        electron_density = f['electron_density'][:]
        altitude = f['altitude'][:]
        chapman_params = f['chapman_params'][:]
    
    # Calculate errors
    errors = calculate_fit_errors(electron_density, altitude, chapman_params)
    
    # Plot error distribution
    plot_error_distribution(errors)
    
    # Plot worst fits
    plot_worst_fits(electron_density, altitude, chapman_params, errors, n_worst=5)
    
    # Identify outliers using z-score
    z_scores = zscore(np.log10(errors))
    outlier_threshold = 2.0  # Profiles with z-score > 2 are considered outliers
    outlier_indices = np.where(np.abs(z_scores) > outlier_threshold)[0]
    
    print(f"\nFound {len(outlier_indices)} profiles with poor fits (z-score > {outlier_threshold})")
    print("\nWorst 5 fits (indices):")
    worst_indices = np.argsort(errors)[-5:]
    for idx in worst_indices:
        print(f"Profile {idx}: MSE = {errors[idx]:.2e}")
    
    # Save filtered data
    good_indices = np.setdiff1d(np.arange(len(electron_density)), outlier_indices)
    
    # Save filtered data to a new file
    with h5py.File("./data/transformed/filtered_electron_density_profiles_2023.h5", 'w') as f:
        f.create_dataset('electron_density', data=electron_density[good_indices])
        f.create_dataset('altitude', data=altitude)
        f.create_dataset('chapman_params', data=chapman_params[good_indices])
    
    print(f"\nSaved {len(good_indices)} good profiles to filtered_electron_density_profiles_2023.h5")

if __name__ == "__main__":
    main() 
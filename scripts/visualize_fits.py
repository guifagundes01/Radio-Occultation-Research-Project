import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from analyze.functions import *
import os

def load_fit_results(file_path):
    """Load the fit results from the HDF5 file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")
        
    with h5py.File(file_path, 'r') as f:
        # Validate required datasets exist
        required_datasets = ['electron_density', 'altitude', 'fit_results']
        for dataset in required_datasets:
            if dataset not in f:
                raise KeyError(f"Required dataset '{dataset}' not found in file")
        
        electron_density = f['electron_density'][:]
        altitude = f['altitude'][:]
        fit_results = f['fit_results']
        
        # Validate fit results group
        required_fit_results = ['chapman_params', 'fit_errors', 'is_good_fit']
        for result in required_fit_results:
            if result not in fit_results:
                raise KeyError(f"Required fit result '{result}' not found in file")
        
        chapman_params = fit_results['chapman_params'][:]
        fit_errors = fit_results['fit_errors'][:]
        is_good_fit = fit_results['is_good_fit'][:]
        mse_threshold = fit_results.attrs['mse_threshold']
        
        # Validate data shapes
        if not (len(electron_density) == len(chapman_params) == len(fit_errors) == len(is_good_fit)):
            raise ValueError("Inconsistent array lengths in data")
            
    return electron_density, altitude, chapman_params, fit_errors, is_good_fit, mse_threshold

def plot_fit_quality_distribution(fit_errors, is_good_fit, mse_threshold):
    """Plot the distribution of fit errors."""
    plt.figure(figsize=(10, 6))
    # Filter out zero errors and invalid values
    valid_errors = fit_errors[(fit_errors > 0) & ~np.isnan(fit_errors)]
    if len(valid_errors) == 0:
        print("Warning: No valid errors to plot")
        return
        
    sns.histplot(np.log10(valid_errors), bins=50)
    plt.axvline(np.log10(mse_threshold), color='r', linestyle='--', 
                label=f'MSE Threshold: {mse_threshold:.2e}')
    plt.xlabel('Log10(MSE)')
    plt.ylabel('Count')
    plt.title('Distribution of Fit Errors')
    plt.legend()
    plt.savefig('figures/fit_error_distribution.png')
    plt.close()

def plot_parameter_distributions(chapman_params, is_good_fit):
    """Plot distributions of Chapman parameters for good and bad fits."""
    param_names = ['Nmax', 'hmax', 'H', 'a1', 'Nm', 'a2', 'hm']  # Correct parameter names
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()
    
    for i, (ax, name) in enumerate(zip(axes, param_names)):
        if i < len(param_names):
            # Filter out invalid values
            good_params = chapman_params[is_good_fit, i]
            bad_params = chapman_params[~is_good_fit, i]
            good_params = good_params[~np.isnan(good_params)]
            bad_params = bad_params[~np.isnan(bad_params)]
            
            if len(good_params) > 0:
                sns.histplot(data=good_params, label='Good Fits', alpha=0.5, ax=ax)
            if len(bad_params) > 0:
                sns.histplot(data=bad_params, label='Bad Fits', alpha=0.5, ax=ax)
                
            ax.set_xlabel(name)
            ax.set_ylabel('Count')
            ax.legend()
    
    plt.tight_layout()
    plt.savefig('figures/parameter_distributions.png')
    plt.close()

def plot_example_fits(electron_density, altitude, chapman_params, is_good_fit, n_examples=5):
    """Plot example good and bad fits."""
    # Get indices of good and bad fits
    good_indices = np.where(is_good_fit)[0]
    bad_indices = np.where(~is_good_fit)[0]
    
    if len(good_indices) == 0 or len(bad_indices) == 0:
        print("Warning: Not enough good/bad fits to plot examples")
        return
    
    # Select random examples
    n_good = min(n_examples, len(good_indices))
    n_bad = min(n_examples, len(bad_indices))
    good_examples = np.random.choice(good_indices, n_good, replace=False)
    bad_examples = np.random.choice(bad_indices, n_bad, replace=False)
    
    fig, axes = plt.subplots(2, max(n_good, n_bad), figsize=(20, 8))
    if n_good == 1 and n_bad == 1:
        axes = axes.reshape(2, 1)
    
    # Plot good fits
    for i, idx in enumerate(good_examples):
        ax = axes[0, i]
        ydata = electron_density[idx]
        yfit = chapmanF2F1(altitude, *chapman_params[idx])
        
        ax.plot(altitude, ydata, 'k.', label='Data', alpha=0.5)
        ax.plot(altitude, yfit, 'r-', label='Fit')
        ax.set_title(f'Good Fit {i+1}')
        ax.set_xlabel('Altitude (km)')
        ax.set_ylabel('Electron Density (cm⁻³)')
        ax.legend()
    
    # Plot bad fits
    for i, idx in enumerate(bad_examples):
        ax = axes[1, i]
        ydata = electron_density[idx]
        yfit = chapmanF2F1(altitude, *chapman_params[idx])
        
        ax.plot(altitude, ydata, 'k.', label='Data', alpha=0.5)
        ax.plot(altitude, yfit, 'r-', label='Fit')
        ax.set_title(f'Bad Fit {i+1}')
        ax.set_xlabel('Altitude (km)')
        ax.set_ylabel('Electron Density (cm⁻³)')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('figures/example_fits.png')
    plt.close()

def main():
    try:
        # Create figures directory if it doesn't exist
        if not os.path.exists('figures'):
            os.makedirs('figures')
        
        # Load the fit results
        input_file = "./data/transformed/electron_density_profiles_2023_with_fits.h5"
        electron_density, altitude, chapman_params, fit_errors, is_good_fit, mse_threshold = load_fit_results(input_file)
        
        # Generate visualizations
        plot_fit_quality_distribution(fit_errors, is_good_fit, mse_threshold)
        plot_parameter_distributions(chapman_params, is_good_fit)
        plot_example_fits(electron_density, altitude, chapman_params, is_good_fit)
        
        # Print summary statistics
        print(f"Total number of profiles: {len(is_good_fit)}")
        print(f"Number of good fits: {np.sum(is_good_fit)}")
        print(f"Number of bad fits: {np.sum(~is_good_fit)}")
        print(f"Good fit percentage: {100 * np.mean(is_good_fit):.2f}%")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main() 
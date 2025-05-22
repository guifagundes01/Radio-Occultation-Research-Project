import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from os import path
from scipy import stats

def epstein(x, Nmax, hmax, H, s):
    """Epstein function for electron density profile"""
    z = (x - hmax) / H
    return Nmax * np.exp(1 - z - np.exp(-z)) * (1 + s * np.tanh(z))

def chapmanF2F1(x, Nmax, hmax, H, a1, Nm, a2, hm):
    """Chapman function for F2 and F1 layers"""
    z = (x - hmax) / H
    zf1 = (x - hm) / H
    F2 = Nmax * np.exp(1 - z - np.exp(-z))
    F1 = Nm * np.exp(1 - zf1 - np.exp(-zf1))
    return F2 + a1 * F1

# Load the fit results
fit_results = np.load('./data/fit_results/fit_results_2023.npz')
chapman_params = fit_results['chapman']
epstein_params = fit_results['epstein']
chapman_errors = fit_results['chapman_errors']
epstein_errors = fit_results['epstein_errors']

# Load original data
file_path = "./data/transformed/electron_density_profiles_2023.h5"
import h5py
with h5py.File(file_path, 'r') as f:
    electron_density = f['electron_density'][:]
    altitude = f['altitude'][:]

def analyze_chapman_parameters():
    """Analyze Chapman model parameters"""
    # Chapman parameters: [Nmax, hmax, H, a1, Nm, a2, hm]
    param_names = ['Nmax', 'hmax', 'H', 'a1', 'Nm', 'a2', 'hm']
    
    print("\nChapman Model Parameters Statistics:")
    print("-" * 40)
    for i, name in enumerate(param_names):
        values = chapman_params[:, i]
        print(f"{name}:")
        print(f"  Mean: {np.mean(values):.2e}")
        print(f"  Std:  {np.std(values):.2e}")
        print(f"  Min:  {np.min(values):.2e}")
        print(f"  Max:  {np.max(values):.2e}")
        print()

def analyze_epstein_parameters():
    """Analyze Epstein model parameters"""
    # Epstein parameters: [Nmax, hmax, H, s]
    param_names = ['Nmax', 'hmax', 'H', 's']
    
    print("\nEpstein Model Parameters Statistics:")
    print("-" * 40)
    for i, name in enumerate(param_names):
        values = epstein_params[:, i]
        print(f"{name}:")
        print(f"  Mean: {np.mean(values):.2e}")
        print(f"  Std:  {np.std(values):.2e}")
        print(f"  Min:  {np.min(values):.2e}")
        print(f"  Max:  {np.max(values):.2e}")
        print()

def compare_model_errors():
    """Compare the errors between Chapman and Epstein models"""
    print("\nModel Error Comparison:")
    print("-" * 40)
    print(f"Chapman Mean MSE:  {np.mean(chapman_errors):.2e}")
    print(f"Epstein Mean MSE:  {np.mean(epstein_errors):.2e}")
    
    # Perform t-test to compare errors
    t_stat, p_value = stats.ttest_ind(chapman_errors, epstein_errors)
    print(f"\nt-test p-value: {p_value:.4f}")
    if p_value < 0.05:
        better_model = "Chapman" if np.mean(chapman_errors) < np.mean(epstein_errors) else "Epstein"
        print(f"The difference is statistically significant. {better_model} performs better.")
    else:
        print("The difference is not statistically significant.")

def plot_error_comparison():
    """Create violin plot comparing model errors"""
    plt.figure(figsize=(10, 6))
    data = [chapman_errors, epstein_errors]
    sns.violinplot(data=data)
    plt.xticks([0, 1], ['Chapman', 'Epstein'])
    plt.ylabel('Mean Squared Error')
    plt.title('Distribution of Fitting Errors')
    plt.yscale('log')
    plt.savefig('./data/fit_results/error_comparison.png')
    plt.close()

def plot_parameter_distributions():
    """Plot parameter distributions for both models"""
    # Chapman parameters
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('Chapman Model Parameter Distributions')
    param_names = ['Nmax', 'hmax', 'H', 'a1', 'Nm', 'a2', 'hm']
    for i, (ax, name) in enumerate(zip(axes.flat, param_names)):
        if i < len(param_names):
            sns.histplot(chapman_params[:, i], ax=ax)
            ax.set_title(name)
    plt.tight_layout()
    plt.savefig('./data/fit_results/chapman_params_dist.png')
    plt.close()

    # Epstein parameters
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Epstein Model Parameter Distributions')
    param_names = ['Nmax', 'hmax', 'H', 's']
    for i, (ax, name) in enumerate(zip(axes.flat, param_names)):
        if i < len(param_names):
            sns.histplot(epstein_params[:, i], ax=ax)
            ax.set_title(name)
    plt.tight_layout()
    plt.savefig('./data/fit_results/epstein_params_dist.png')
    plt.close()

def plot_correlation_matrices():
    """Plot correlation matrices for model parameters"""
    # Chapman correlations
    chapman_corr = np.corrcoef(chapman_params.T)
    plt.figure(figsize=(10, 8))
    sns.heatmap(chapman_corr, 
                xticklabels=['Nmax', 'hmax', 'H', 'a1', 'Nm', 'a2', 'hm'],
                yticklabels=['Nmax', 'hmax', 'H', 'a1', 'Nm', 'a2', 'hm'],
                annot=True, cmap='coolwarm')
    plt.title('Chapman Parameters Correlation Matrix')
    plt.tight_layout()
    plt.savefig('./data/fit_results/chapman_correlations.png')
    plt.close()

    # Epstein correlations
    epstein_corr = np.corrcoef(epstein_params.T)
    plt.figure(figsize=(8, 6))
    sns.heatmap(epstein_corr,
                xticklabels=['Nmax', 'hmax', 'H', 's'],
                yticklabels=['Nmax', 'hmax', 'H', 's'],
                annot=True, cmap='coolwarm')
    plt.title('Epstein Parameters Correlation Matrix')
    plt.tight_layout()
    plt.savefig('./data/fit_results/epstein_correlations.png')
    plt.close()

def plot_worst_fits(n_worst=5):
    """Plot the n worst fits for both models"""
    # Get indices of worst fits
    chapman_worst_idx = np.argsort(chapman_errors)[-n_worst:]
    epstein_worst_idx = np.argsort(epstein_errors)[-n_worst:]
    
    # Create subplots for Chapman worst fits
    fig, axes = plt.subplots(n_worst, 1, figsize=(12, 4*n_worst))
    fig.suptitle('Worst Chapman Fits')
    
    for i, idx in enumerate(chapman_worst_idx):
        ax = axes[i]
        # Plot original data
        ax.scatter(electron_density[idx], altitude, label='Data', alpha=0.6, s=20)
        
        # Plot Chapman fit
        params = chapman_params[idx]
        y_chapman = chapmanF2F1(altitude, *params)
        ax.plot(y_chapman, altitude, 'r-', label='Chapman fit')
        
        # Plot Epstein fit for comparison
        params = epstein_params[idx]
        y_epstein = epstein(altitude, *params)
        ax.plot(y_epstein, altitude, 'g--', label='Epstein fit')
        
        ax.set_title(f'Profile {idx} - MSE: {chapman_errors[idx]:.2e}')
        ax.set_xlabel('Electron Density (m⁻³)')
        ax.set_ylabel('Altitude (km)')
        ax.legend()
        ax.grid(True)
        # ax.set_xscale('log')  # Use log scale for electron density
    
    plt.tight_layout()
    plt.savefig('./data/fit_results/worst_chapman_fits.png')
    plt.close()
    
    # Create subplots for Epstein worst fits
    fig, axes = plt.subplots(n_worst, 1, figsize=(12, 4*n_worst))
    fig.suptitle('Worst Epstein Fits')
    
    for i, idx in enumerate(epstein_worst_idx):
        ax = axes[i]
        # Plot original data
        ax.scatter(electron_density[idx], altitude, label='Data', alpha=0.6, s=20)
        
        # Plot Epstein fit
        params = epstein_params[idx]
        y_epstein = epstein(altitude, *params)
        ax.plot(y_epstein, altitude, 'g-', label='Epstein fit')
        
        # Plot Chapman fit for comparison
        params = chapman_params[idx]
        y_chapman = chapmanF2F1(altitude, *params)
        ax.plot(y_chapman, altitude, 'r--', label='Chapman fit')
        
        ax.set_title(f'Profile {idx} - MSE: {epstein_errors[idx]:.2e}')
        ax.set_xlabel('Electron Density (m⁻³)')
        ax.set_ylabel('Altitude (km)')
        ax.legend()
        ax.grid(True)
        # ax.set_xscale('log')  # Use log scale for electron density
    
    plt.tight_layout()
    plt.savefig('./data/fit_results/worst_epstein_fits.png')
    plt.close()

if __name__ == "__main__":
    # Create analysis directory if it doesn't exist
    if not path.exists('./data/fit_results'):
        print("Error: Fit results directory not found!")
        exit(1)

    # Run analyses
    # analyze_chapman_parameters()
    # analyze_epstein_parameters()
    # compare_model_errors()
    
    # # Create visualizations
    # plot_error_comparison()
    # plot_parameter_distributions()
    # plot_correlation_matrices()
    plot_worst_fits(n_worst=5)  # Plot 5 worst fits for each model
    
    print("\nAnalysis complete! Plots have been saved in the data/fit_results directory.") 
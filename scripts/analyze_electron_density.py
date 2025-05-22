import h5py
import numpy as np
import matplotlib.pyplot as plt
import glob
import pandas as pd

def analyze_electron_density(file_path):
    with h5py.File(file_path, "r") as f:
        electron_density = f["electron_density"][:]
        
        # Count profiles with negative values
        negative_profiles = np.any(electron_density < 0, axis=1)
        num_negative_profiles = np.sum(negative_profiles)
        total_profiles = electron_density.shape[0]
        
        # Get all negative values
        negative_values = electron_density[electron_density < 0]
        
        # Calculate statistics
        stats = {
            'total_profiles': total_profiles,
            'profiles_with_negative_values': num_negative_profiles,
            'percentage_negative_profiles': (num_negative_profiles / total_profiles) * 100,
            'min_value': np.min(negative_values) if len(negative_values) > 0 else 0,
            'max_negative': np.max(negative_values) if len(negative_values) > 0 else 0,
            'mean_negative': np.mean(negative_values) if len(negative_values) > 0 else 0,
            'std_negative': np.std(negative_values) if len(negative_values) > 0 else 0
        }
        
        # Create histogram of negative values
        plt.figure(figsize=(10, 6))
        plt.hist(negative_values, bins=50, alpha=0.7)
        plt.title('Distribution of Negative Electron Density Values')
        plt.xlabel('Electron Density')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        plt.savefig('negative_electron_density_distribution.png')
        plt.close()
        
        return stats

def main():
    # Get all HDF5 files in the data directory
    files = glob.glob("./data/transformed/*")
    
    all_stats = []
    for file in files:
        print(f"\nAnalyzing file: {file}")
        stats = analyze_electron_density(file)
        all_stats.append(stats)
        
        # Print statistics for each file
        print("\nStatistics:")
        print(f"Total profiles: {stats['total_profiles']}")
        print(f"Profiles with negative values: {stats['profiles_with_negative_values']}")
        print(f"Percentage of profiles with negative values: {stats['percentage_negative_profiles']:.2f}%")
        print(f"Minimum value: {stats['min_value']:.2e}")
        print(f"Maximum negative value: {stats['max_negative']:.2e}")
        print(f"Mean of negative values: {stats['mean_negative']:.2e}")
        print(f"Standard deviation of negative values: {stats['std_negative']:.2e}")
    
    # Create summary DataFrame
    df = pd.DataFrame(all_stats)
    df.to_csv('./data/analysis/electron_density_analysis_summary.csv', index=False)
    print("\nAnalysis complete. Results saved to 'electron_density_analysis_summary.csv'")

if __name__ == "__main__":
    main() 
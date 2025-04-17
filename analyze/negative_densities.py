import h5py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    # Open the HDF5 file
    with h5py.File("./data/original/electron_density_profiles_2023.h5", "r") as f:
        # Access datasets    
        electron_density_2023 = f["electron_density"][:]
        latitude = f["latitude"][:]
        longitude = f["longitude"][:]
        altitude = f["altitude"][:]
        local_time = f["local_time"][:]
        f107 = f["F10.7"][:]
        kp = f["Kp"][:]

        # Access metadata
        description = f.attrs["description"]
        num_profiles = f.attrs["num_profiles"]

    print(f"Loaded {num_profiles} profiles: {description}")
    local_time_date = pd.to_datetime(local_time, unit='s')

    negative_mask = electron_density_2023 < 0
    percent_negative_per_profile = np.sum(negative_mask, axis=1) / electron_density_2023.shape[1] * 100
    total_negative_percentage = np.sum(negative_mask) / np.prod(electron_density_2023.shape) * 100

    print(f"Total percentage of negative values: {total_negative_percentage:.2f}%")
    plt.figure(figsize=(10, 5))
    plt.hist(percent_negative_per_profile, bins=50, color='blue', edgecolor='black')
    plt.xlabel('Percentage of Negative Values per Profile')
    plt.ylabel('Frequency')
    plt.title('Distribution of Negative Values per Profile')
    plt.show()

    # Filter out profiles with less than 1% negative values
    filtered_percent_negative = percent_negative_per_profile[percent_negative_per_profile >= 1]

    plt.figure(figsize=(10, 5))
    plt.hist(filtered_percent_negative, bins=50, color='green', edgecolor='black')
    plt.xlabel('Percentage of Negative Values per Profile')
    plt.ylabel('Frequency')
    plt.title('Distribution of Negative Values per Profile (>= 1%)')
    plt.show()

    # Plot altitude-wise analysis
    negative_percentage_per_altitude_2023 = np.sum(negative_mask, axis=0) / electron_density_2023.shape[0] * 100

    plt.figure(figsize=(10, 5))
    plt.plot(altitude, negative_percentage_per_altitude_2023, color='red', marker='.')
    plt.xlabel('Altitude Index')
    plt.ylabel('Percentage of Negative Values')
    plt.title('Negative Values Distribution by Altitude')
    plt.grid(True)
    plt.show()

    negative_percentage_per_altitude_2023_2 = np.sum(negative_mask, axis=0) / filtered_percent_negative.shape[0] * 100

    plt.figure(figsize=(10, 5))
    plt.plot(altitude, negative_percentage_per_altitude_2023_2, color='red', marker='.')
    plt.xlabel('Altitude Index')
    plt.ylabel('Percentage of Negative Values')
    plt.title('Negative Values Distribution by Altitude from profiles with >= 1% negative values')
    plt.grid(True)
    plt.show()


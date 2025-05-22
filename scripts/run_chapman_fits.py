from analyze.fits import *
from analyze.functions import *
from tqdm import tqdm
from os import path, mkdir, getcwd
import numpy as np
import h5py
from scipy.optimize import curve_fit
import time
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


if __name__ == "__main__":

    input_file_path = "./data/transformed/electron_density_profiles_2023.h5"
    output_file_path = "./data/transformed/electron_density_profiles_2023_with_fits.h5"
    
    # Load data from input file
    electron_density, latitude, longitude, altitude, local_time, f107, kp, dip = load_data(input_file_path)

    counter = 0
    good_fits_counter = 0

    # all profiles
    n_profiles = electron_density.shape[0]
    
    # Threshold for good fit (mean squared error)
    MSE_THRESHOLD = 1e11 
    
    # Initialize arrays for all profiles with zeros
    all_chapman_params = np.zeros((n_profiles, 7))  # 7 parameters for Chapman
    all_fit_errors = np.zeros(n_profiles)
    is_good_fit = np.zeros(n_profiles, dtype=bool)

    for idx in tqdm(range(n_profiles)):
        ydata = electron_density[idx, :]
        xdata = altitude

        if np.all(np.isnan(ydata)) or np.all(ydata == 0):
            continue

        try:
            # Initial guesses
            peak_idx = np.argmax(ydata)
            Nmax = ydata[peak_idx]
            hmax = altitude[peak_idx]

            p0_chapman = [Nmax, hmax, 50, 1, 1e4, 1, 180]

            bounds_lower = [1e2, 250, 10, 0.1, 1e1, 0.1, 10]
            bounds_upper = [1e7, 450, 250, 5.0, 5e5, 5.0, 250]

            if not all(lower <= value <= upper for value, lower, upper in zip(p0_chapman, bounds_lower, bounds_upper)):
                continue
            
            counter += 1
            # Fit Chapman model
            popt_chapman, _ = curve_fit(chapmanF2F1, xdata, ydata, bounds=(bounds_lower, bounds_upper), p0=p0_chapman, maxfev=1000)

            # Compute mean squared error
            y_chapman = chapmanF2F1(xdata, *popt_chapman)
            mse_chapman = np.mean((ydata - y_chapman) ** 2)

            # Store parameters and error for all profiles
            all_chapman_params[idx] = popt_chapman
            all_fit_errors[idx] = mse_chapman
            
            # Mark if it's a good fit
            if mse_chapman < MSE_THRESHOLD:
                good_fits_counter += 1
                is_good_fit[idx] = True

        except RuntimeError as e:
            print(f"Fit failed for profile {idx}: {e}")
            continue

    print(f"Total profiles fit: {counter}")
    print(f"Good fits (below threshold): {good_fits_counter}")

    # Create a new file with all the data
    with h5py.File(output_file_path, 'w') as f:
        # Copy original data
        f.create_dataset('electron_density', data=electron_density)
        f.create_dataset('latitude', data=latitude)
        f.create_dataset('longitude', data=longitude)
        f.create_dataset('altitude', data=altitude)
        f.create_dataset('local_time', data=local_time)
        f.create_dataset('f107', data=f107)
        f.create_dataset('kp', data=kp)
        f.create_dataset('dip', data=dip)
        
        # Add fit results
        fit_group = f.create_group('fit_results')
        fit_group.create_dataset('chapman_params', data=all_chapman_params)
        fit_group.create_dataset('fit_errors', data=all_fit_errors)
        fit_group.create_dataset('is_good_fit', data=is_good_fit)
        fit_group.attrs['mse_threshold'] = MSE_THRESHOLD

    print(f"Results saved to {output_file_path}")   

from analyze.fits import *
from analyze.functions import *
from tqdm import tqdm
from os import path, mkdir, getcwd
import numpy as np
import h5py
from scipy.optimize import curve_fit


if __name__ == "__main__":

    file_path = "./data/transformed/electron_density_profiles_2023.h5"
    electron_density, latitude, longitude, altitude, local_time, f107, kp, dip = load_data(file_path)

    counter = 0

    # all profiles
    n_profiles = electron_density.shape[0]
    fit_results = {
        "chapman": [],
        "epstein": [],
    }

    fit_errors = {
        "chapman": [],
        "epstein": [],
    }

    for idx in tqdm(range(n_profiles)):
    # for idx in tqdm(range(100)):
        ydata = electron_density[idx, :]
        xdata = altitude

        if np.all(np.isnan(ydata)) or np.all(ydata == 0):
            continue

        try:
            # Initial guesses
            peak_idx = np.argmax(ydata)
            Nmax = ydata[peak_idx]
            hmax = altitude[peak_idx]

            p0_epstein = [Nmax, hmax, 80, 0.2]
            p0_chapman = [Nmax, hmax, 50, 1, 1e4, 1, 180]

            bounds_lower = [1e2, 250, 10, 0.1, 1e3, 0.1, 10]
            bounds_upper = [1e7, 450, 250, 5.0, 5e5, 5.0, 250]

            if not all(lower <= value <= upper for value, lower, upper in zip(p0_chapman, bounds_lower, bounds_upper)):
                continue
            
            counter += 1
            # Fit models
            popt_epstein, _ = curve_fit(epstein, xdata, ydata, p0=p0_epstein, maxfev=10000)
            popt_chapman, _ = curve_fit(chapmanF2F1, xdata, ydata, bounds=(bounds_lower, bounds_upper), p0=p0_chapman, maxfev=10000)

            # Store params
            fit_results["chapman"].append(popt_chapman)
            fit_results["epstein"].append(popt_epstein)

            # Compute mean squared error for each
            y_chapman = chapmanF2F1(xdata, *popt_chapman)
            y_epstein = epstein(xdata, *popt_epstein)

            mse_chapman = np.mean((ydata - y_chapman) ** 2)
            mse_epstein = np.mean((ydata - y_epstein) ** 2)

            fit_errors["chapman"].append(mse_chapman)
            fit_errors["epstein"].append(mse_epstein)

        except RuntimeError as e:
            print(f"Fit failed for profile {idx}: {e}")
            # Skip profile if any fit fails
            continue

    print(f"Total profiles fit: {counter}")

    # Save fit results
    fit_results_dir = "./data/fit_results"
    if not path.exists(fit_results_dir): mkdir(fit_results_dir)
    fit_results_path = path.join(fit_results_dir, "fit_results_2023.npz")
    
    np.savez(
        fit_results_path,
        chapman=fit_results["chapman"],
        epstein=fit_results["epstein"],
        chapman_errors=fit_errors["chapman"],
        epstein_errors=fit_errors["epstein"]
    )

    print(f"Fit results saved to {fit_results_path}")



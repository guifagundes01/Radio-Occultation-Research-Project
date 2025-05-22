import h5py
import numpy as np
import os

def filter_profiles(input_file):
    print(f"Processing file: {input_file}")
    
    with h5py.File(input_file, "r") as f:
        # Read all datasets
        electron_density = f["electron_density"][:]
        latitude = f["latitude"][:]
        longitude = f["longitude"][:]
        altitude = f["altitude"][:]
        local_time = f["local_time"][:]
        kp = f["kp"][:]
        f10_7 = f["f107"][:]
        dip = f["dip"][:]
        fit_results = f["fit_results"]
        
        # Create mask for positive electron density and Kp < 3
        positive_density_mask = np.all(electron_density >= 0, axis=1)
        kp_mask = kp < 3
        
        # Combine masks
        valid_profiles_mask = positive_density_mask & kp_mask
        
        # Count statistics
        total_profiles = len(valid_profiles_mask)
        valid_profiles = np.sum(valid_profiles_mask)
        print(f"Total profiles: {total_profiles}")
        print(f"Valid profiles (positive density and Kp < 3): {valid_profiles}")
        print(f"Percentage kept: {(valid_profiles/total_profiles)*100:.2f}%")
        
        # Filter all datasets
        filtered_data = {
            "electron_density": electron_density[valid_profiles_mask],
            "latitude": latitude[valid_profiles_mask],
            "longitude": longitude[valid_profiles_mask],
            "altitude": altitude,
            "local_time": local_time[valid_profiles_mask],
            "kp": kp[valid_profiles_mask],
            "f107": f10_7[valid_profiles_mask],
            "dip": dip[valid_profiles_mask]
        }
        
        # Save filtered data
        output_file = input_file.replace('transformed', 'filtered')
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with h5py.File(output_file, "w") as f_out:
            # Save main datasets
            for key, data in filtered_data.items():
                f_out.create_dataset(key, data=data, compression="gzip", compression_opts=9)
            
            # Create fit_results group and save its datasets
            fit_group = f_out.create_group("fit_results")
            for key in fit_results.keys():
                fit_group.create_dataset(key, data=fit_results[key][valid_profiles_mask], 
                                       compression="gzip", compression_opts=9)
            
            # Copy any additional attributes from the original file
            for key, value in f.attrs.items():
                f_out.attrs[key] = value
            
            # Add new attributes about filtering
            f_out.attrs["original_profiles"] = total_profiles
            f_out.attrs["filtered_profiles"] = valid_profiles
            f_out.attrs["filtering_criteria"] = "Positive electron density and Kp < 3"
        
        print(f"Saved filtered data to: {output_file}\n")

def main():
    # Process only the specific file with fits
    input_file = "./data/transformed/electron_density_profiles_2023_with_fits.h5"
    
    if not os.path.exists(input_file):
        print(f"File not found: {input_file}")
        return
    
    filter_profiles(input_file)
    print("Filtering complete!")

if __name__ == "__main__":
    main() 
import h5py


# Open the HDF5 file
with h5py.File("electron_density_profiles.h5", "r") as f:
    # Access datasets
    print(f.items)
    electron_density = f["electron_density"][:]
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
import numpy as np
import matplotlib.pyplot as plt
import h5py

def load_data(file_path):
    f = h5py.File(file_path, "r")
    electron_density = f["electron_density"][:]
    latitude = f["latitude"][:]
    longitude = f["longitude"][:]
    altitude = f["altitude"][:]
    local_time = f["local_time"][:]
    f107 = f["f107"][:]
    kp = f["kp"][:]
    dip = f["dip"][:]
    return electron_density, latitude, longitude, altitude, local_time, f107, kp, dip

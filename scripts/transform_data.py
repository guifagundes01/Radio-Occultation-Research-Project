import h5py
import pandas as pd
import glob
from transform.add_dip import dip_array
from transform.add_f10_7 import f10_7_array
from transform.add_Kp import kp_array


if __name__ == "__main__":

    globs = glob.glob("./data/original/*")
    for file in globs:
        print(file)
        with h5py.File(file, "r") as f:
            electron_density = f["electron_density"][:]
            latitude = f["latitude"][:]
            longitude = f["longitude"][:]
            altitude = f["altitude"][:]
            local_time = f["local_time"][:]
            description = f.attrs["description"]
            num_profiles = f.attrs["num_profiles"]
            a = 'data/original/123'
        local_time_date = pd.to_datetime(local_time, unit='s')

        transformed_file = file.replace('original', 'transformed')

        dip_array_ = dip_array(latitude, longitude, altitude, local_time_date, num_profiles)
        print('success dip')
        f10_7_array_ = f10_7_array(local_time_date, longitude)
        print('success f10.7')
        kp_array_ = kp_array(local_time_date, longitude)
        print('success kp')

        with h5py.File(transformed_file, "w") as f:
            f.create_dataset("electron_density", data=electron_density, compression="gzip", compression_opts=9)
            f.create_dataset("latitude", data=latitude, compression="gzip", compression_opts=9)
            f.create_dataset("longitude", data=longitude, compression="gzip", compression_opts=9)
            f.create_dataset("altitude", data=altitude, compression="gzip", compression_opts=9)
            f.create_dataset("local_time", data=local_time, compression="gzip", compression_opts=9)
            f.create_dataset("dip", data=dip_array_, compression="gzip", compression_opts=9)
            f.create_dataset("f10_7", data=f10_7_array_, compression="gzip", compression_opts=9)
            f.create_dataset("kp", data=kp_array_, compression="gzip", compression_opts=9)
        print(f'saved {transformed_file}')


import pyIGRF
import datetime
import math
import numpy as np

def year_fraction(date):
    start = datetime.date(date.year, 1, 1).toordinal()
    year_length = datetime.date(date.year+1, 1, 1).toordinal() - start
    return date.year + float(date.toordinal() - start) / year_length

def calculate_dip(lat, lon, alt, local_time):
    year_date = year_fraction(local_time)
    D, I, H, X, Y, Z, F = pyIGRF.igrf_value(lat, lon, alt, year_date)
    occ_dip = math.atan(math.tan(math.radians(I)) / 2) * (180 / math.pi)
    return occ_dip

def dip_array(latitude, longitude, altitude, local_time_date, num_profiles):
    dip_values = []
    for i in range(num_profiles):
        lat = latitude[i]
        lon = longitude[i]
        
        # Use the altitude of 300m 
        alt = altitude[150]

        date = local_time_date[i]

        # Calculate DIP and append
        dip = calculate_dip(lat, lon, alt, date)
        dip_values.append(dip)
    return np.array(dip_values)

import numpy as np
import pandas as pd
import datetime
import json

# Kp data from https://kp.gfz-potsdam.de/en/

def utc_time(row):
    utc_time = row['aux'] - datetime.timedelta(hours=row['lon']/15)
    return utc_time

def data_to_df(local_time_date, longitude):
    model_df = pd.DataFrame(local_time_date, columns=['aux'])
    model_df['lon'] = longitude
    model_df['utc_aux'] = model_df.apply(lambda row: utc_time(row), axis=1)
    model_df['date'] = model_df.apply(lambda x: x['utc_aux'].strftime("%d/%m/%Y"), axis=1)
    return model_df

def kp_array(local_time_date, longitude, file_kp='./data/kp/kp_2019_2024.json'):
    with open(file_kp, 'r') as file:
        data = json.load(file)
    data.pop("meta")
    data.pop("status")
    kp = pd.DataFrame.from_dict(data)
    kp['year'] = pd.to_datetime(kp['datetime']).dt.year
    kp['DOY'] =  pd.to_datetime(kp['datetime']).dt.day_of_year
    kp['utc_hour'] = pd.to_datetime(kp['datetime']).dt.hour
    kp = kp[kp['utc_hour'] == 12]
    kp['date'] = kp.apply(lambda row: (datetime.datetime(int(row['year']), 1, 1) + datetime.timedelta(days=row['DOY'] - 1)).strftime('%d/%m/%Y'), axis=1)
    model_df = data_to_df(local_time_date, longitude)
    model_df = model_df.merge(kp[['date', 'Kp']], on='date', how='left')    
    kp_array = np.array(model_df['Kp'])
    return kp_array


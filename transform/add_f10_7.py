import numpy as np
import pandas as pd
import datetime

def utc_time(row):
    utc_time = row['aux'] - datetime.timedelta(hours=row['lon']/15)
    return utc_time

def data_to_df(local_time_date, longitude):
    model_df = pd.DataFrame(local_time_date, columns=['aux'])
    model_df['lon'] = longitude
    model_df['utc_aux'] = model_df.apply(lambda row: utc_time(row), axis=1)
    model_df['date'] = model_df.apply(lambda x: x['utc_aux'].strftime("%d/%m/%Y"), axis=1)
    return model_df

def f10_7_array(local_time_date, longitude, f10_7_file='./data/f10_7/f10_7_2019_2024.csv'):
    f10_df = pd.read_csv(f10_7_file)
    model_df = data_to_df(local_time_date, longitude)
    model_df = model_df.merge(f10_df[['date', 'f10_7']], on='date', how='left')
    f10_7_array = np.array(model_df['f10_7'])
    return f10_7_array


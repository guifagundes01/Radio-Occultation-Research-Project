import numpy as np
import pandas as pd

def data_to_df(local_time_date):
    model_df = pd.DataFrame(local_time_date, columns=['aux'])
    model_df['date'] = model_df.apply(lambda x: x['aux'].strftime("%d/%m/%Y"), axis=1)
    return model_df

def f10_7_array(local_time_date, f10_7_file='../data/f10_7/f10_7_2019_2024.csv'):
    f10_df = pd.read_csv(f10_7_file)
    model_df = data_to_df(local_time_date)
    model_df = model_df.merge(f10_df[['date', 'f10_7']], on='date', how='left')
    f10_7_array = np.array(model_df['f10_7'])
    return f10_7_array


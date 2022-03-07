import pandas as pd
import matplotlib.pyplot as plt

fileloc = 'C://Users//marku//Desktop//krafthack_files//'

df1 = pd.read_parquet(fileloc+'input_dataset-1.parquet', engine='pyarrow')
df2 = pd.read_parquet(fileloc+'input_dataset-2.parquet', engine='pyarrow')

df_target = pd.read_parquet(fileloc+'prediction_input.parquet', engine='pyarrow')

target_columns1 = set(df1.columns).intersection(df_target.columns)
target_columns2 = set(df2.columns).intersection(df_target.columns)


t1 = pd.date_range(df1.index[0],df1.index[-1], freq='1s')
t2 = pd.date_range(df2.index[0],df2.index[-1], freq='1s') 
tt = pd.date_range(df_target.index[0],df_target.index[-1], freq='1s') 


print("fin")
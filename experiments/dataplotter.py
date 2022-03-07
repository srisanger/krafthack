import pandas as pd
import matplotlib.pyplot as plt

fileloc = 'C://Users//marku//Desktop//krafthack_files//'


df2 = pd.read_parquet(fileloc+'input_dataset-2.parquet', engine='pyarrow')
df1 = pd.read_parquet(fileloc+'input_dataset-1.parquet', engine='pyarrow')

print("fin")
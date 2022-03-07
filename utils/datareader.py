import pandas as pd
import numpy as np

def df_from_folder(folder, filenames=['input_dataset-1.parquet', 'input_dataset-2.parquet', 'prediction_input.parquet']):
    """
    read data into pandas
    """
    assert len(filenames) == 3
    # read dataframes:
    df1 = pd.read_parquet(folder+'//'+filenames[0], engine='pyarrow')
    df2 = pd.read_parquet(folder+'//'+filenames[1], engine='pyarrow')
    dfT = pd.read_parquet(folder+'//'+filenames[2], engine='pyarrow')
    # combine input dataframes:
    df = pd.concat((df1,df2))
    # change operational modes to integers:
    df['mode'] = (df['mode'] == 'operation')
    dfT['mode'] = (dfT['mode'] == 'operation')
    # add nan values:
    df = df.reindex(pd.date_range(df.index[0],df.index[-1], freq='1s'), fill_value=np.nan)
    dfT= dfT.reindex(pd.date_range(dfT.index[0],dfT.index[-1], freq='1s'), fill_value=np.nan)
    # change operational modes to integers:
    return df, dfT

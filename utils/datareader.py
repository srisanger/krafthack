import pandas as pd
import numpy as np

def df_from_folder(folder, filenames=['input_dataset-1.parquet', 'input_dataset-2.parquet', 'prediction_input.parquet']):
    """
    read data into pandas and returns the training and target set
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
    # return training and target set:
    return df, dfT

def normalize_input(training_set, target_set):
    """
    normalizes the data set
    """
    mins = training_set.min()
    maxs = training_set.max()
    df = (training_set - mins) / (maxs - mins) - 0.5
    dfT = (target_set - mins[target_set.columns]) / (maxs[target_set.columns] - mins[target_set.columns]) - 0.5
    # return output and scaler:
    scaler = (mins, maxs)
    return df, dfT, scaler

def rescale_output(df, scaler):
    """
    rescale the given data set by the given scalers
    """
    mins = scaler[0][df.columns]
    maxs = scaler[1][df.columns]
    df = (df + 0.5) * (maxs - mins) + mins
    return df
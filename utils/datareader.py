import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def df_from_folder(folder, filenames=['input_dataset-1.parquet', 'input_dataset-2.parquet', 'prediction_input.parquet'], use_firstdataset:bool=False):
    """
    read data into pandas and returns the training and target set
    """
    assert len(filenames) == 3
    # read dataframes:
    if use_firstdataset:
        df1 = pd.read_parquet(folder+'//'+filenames[0], engine='pyarrow')
        df2 = pd.read_parquet(folder+'//'+filenames[1], engine='pyarrow')
        # combine input dataframes:
        df = pd.concat((df1,df2))
    else:
        df = pd.read_parquet(folder+'//'+filenames[1], engine='pyarrow')
    dfT = pd.read_parquet(folder+'//'+filenames[2], engine='pyarrow')
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

def get_seconds_from_start(df):
    """
    Outouts a numpy array of seconds since first timestamp in pandas dataframe.
    """
    return np.array(list(map(lambda x: (x - df.index[0]).total_seconds(), df.index))).reshape(-1,1)

def detrend_series(timestamps, series):
    """
    Argument timestamps is seconds from first measurement.
    Outputs coefficient and intercept of input series and a series where linear trend is removed.
    NB: timestamps must be Nx1 array. Use function reshape(-1,1) to transform 1xN to Nx1.
    """
    model = LinearRegression().fit(timestamps,series)
    return model.coef_, model.intercept_, series - model.coef_*timestamps
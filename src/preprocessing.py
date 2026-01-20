import pandas as pd
import numpy as np

def clean_time_series(df):
    """
    Cleans the time series data.
    """
    # Convert timestamp
    time_col = [c for c in df.columns if 'timestamp' in c.lower()][0]
    df[time_col] = pd.to_datetime(df[time_col])
    
    # Sort and Set index
    df = df.sort_values(by=time_col).set_index(time_col)
    
    # Filter for year 2015-2020 if needed, or keep all. 
    # User mentioned 2020 data links, but dataset might cover more.
    
    # Handle missing values (Interpolate linear for load/generation)
    df = df.interpolate(method='time', limit_direction='both')
    
    return df

def clean_weather_data(df):
    """
    Cleans the weather data.
    """
    time_col = [c for c in df.columns if 'timestamp' in c.lower()][0]
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(by=time_col).set_index(time_col)
    
    # Fill missing weather data (Forward fill is often good for weather)
    df = df.fillna(method='ffill').fillna(method='bfill')
    return df

def detect_outliers_rolling(df, column, window=24, threshold=3):
    """
    Detects outliers using rolling median and std.
    """
    rolling_median = df[column].rolling(window=window, center=True).median()
    rolling_std = df[column].rolling(window=window, center=True).std()
    
    lower_bound = rolling_median - (threshold * rolling_std)
    upper_bound = rolling_median + (threshold * rolling_std)
    
    outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
    return outliers, lower_bound, upper_bound

def remove_outliers(df, column, window=24, threshold=3):
    """
    Replaces outliers with rolling median.
    """
    outliers, lower, upper = detect_outliers_rolling(df, column, window, threshold)
    count = outliers.sum()
    if count > 0:
        print(f"Detected {count} outliers in {column}. Replacing with rolling median.")
        rolling_median = df[column].rolling(window=window, center=True).median()
        df.loc[outliers, column] = rolling_median[outliers]
        # Fill any edges if rolling median is NaN (though center=True helps)
        df.loc[outliers & df[column].isna(), column] = df[column].mean() 
    return df

def preprocess_pipeline(df_ts, df_weather):
    """
    Full preprocessing pipeline.
    """
    print("Preprocessing Time Series...")
    df_ts = clean_time_series(df_ts)
    
    print("Preprocessing Weather...")
    df_weather = clean_weather_data(df_weather)
    
    # Merge
    print("Merging datasets...")
    # Resample weather to match TS if needed, or just merge index
    # Weather is likely hourly, TS hourly.
    merged = df_ts.join(df_weather, how='inner')
    
    return merged

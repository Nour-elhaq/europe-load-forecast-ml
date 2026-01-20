import pandas as pd
import numpy as np

def create_time_features(df):
    """
    Creates time-based features from the index.
    """
    df['hour'] = df.index.hour
    df['day'] = df.index.day
    df['month'] = df.index.month
    df['weekday'] = df.index.weekday
    df['weekend'] = (df['weekday'] >= 5).astype(int)
    
    # Cyclical encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    return df

def create_lag_rollup_features(df, target_col='load_actual'):
    """
    Creates lag and rolling window features.
    NOTE: Using 'load_actual' as placeholder, adjust based on actual column name in OPSD data.
    """
    if target_col not in df.columns:
        # Try to find a load column
        cols = [c for c in df.columns if 'load' in c.lower()]
        if cols:
            target_col = cols[0]
        else:
            print("No load column found for lagging.")
            return df

    # Lags
    lags = [1, 2, 24, 48, 168]
    for lag in lags:
        df[f'lag_{lag}'] = df[target_col].shift(lag)
    
    # Rolling stats
    windows = [3, 6, 24, 168]
    for w in windows:
        df[f'rolling_mean_{w}h'] = df[target_col].rolling(window=w).mean()
        df[f'rolling_std_{w}h'] = df[target_col].rolling(window=w).std()
        
    # Differencing
    df['diff_24h'] = df[target_col] - df[target_col].shift(24)
    
    return df

def calculate_carbon_proxy(df_conv, df_ren):
    """
    Calculates Carbon Proxy Score based on capacity.
    """
    # Simply sum capacities. 
    # Conventionals: 'electrical_capacity' (check columns, sometimes 'capacity')
    # Renewables: 'electrical_capacity' or 'installed_capacity'
    
    fossil_capacity = 0
    renewable_capacity = 0
    
    if 'capacity' in df_conv.columns: # guessing column name based on typical OPSD
        fossil_capacity = df_conv['capacity'].sum() # or filter by fuel type
    elif 'electrical_capacity' in df_conv.columns:
        fossil_capacity = df_conv['electrical_capacity'].sum()
        
    if 'electrical_capacity' in df_ren.columns:
        renewable_capacity = df_ren['electrical_capacity'].sum()
        
    # Create a score (static for the whole region/dataset for now, 
    # unless we join by country and broadcast to time series)
    
    # Carbon Proxy Score: Higher renewable -> Lower carbon score theoretically? 
    # User formula: carbon_score = fossil / (renewable + 1)
    # Scaling to 0-100?
    
    score = fossil_capacity / (renewable_capacity + 1)
    
    return score

def add_carbon_features(df, carbon_score):
    """
    Adds the static carbon score to the dataframe (same value for all rows).
    """
    df['carbon_proxy_score'] = carbon_score
    return df

def feature_engineering_pipeline(df, df_conv, df_ren):
    """
    Master pipeline for features.
    """
    print("Creating time features...")
    df = create_time_features(df)
    
    print("Creating lag/rolling features...")
    df = create_lag_rollup_features(df)
    
    print("Calculating Carbon Proxy...")
    score = calculate_carbon_proxy(df_conv, df_ren)
    print(f"Carbon Proxy Score: {score:.4f}")
    df = add_carbon_features(df, score)
    
    # Drop NaNs created by lags
    df = df.dropna()
    
    return df

import pandas as pd
import os

def load_time_series(data_dir, filename='time_series_60min_singleindex.csv'):
    """
    Loads the time series data.
    """
    # Find the correct subdirectory for time series
    ts_dir = [d for d in os.listdir(data_dir) if 'opsd-time_series' in d]
    if not ts_dir:
        raise FileNotFoundError("Time series directory not found.")
    
    path = os.path.join(data_dir, ts_dir[0], filename)
    print(f"Loading time series from {path}...")
    df = pd.read_csv(path)
    return df

def load_weather_data(data_dir, filename='weather_data.csv'):
    """
    Loads the weather data.
    """
    w_dir = [d for d in os.listdir(data_dir) if 'opsd-weather_data' in d]
    if not w_dir:
         raise FileNotFoundError("Weather data directory not found.")
    
    path = os.path.join(data_dir, w_dir[0], filename)
    print(f"Loading weather data from {path}...")
    df = pd.read_csv(path)
    return df

def load_conventional_power_plants(data_dir, filename='conventional_power_plants_EU.csv'):
    """
    Loads conventional power plants data.
    """
    cpp_dir = [d for d in os.listdir(data_dir) if 'opsd-conventional_power_plants' in d]
    if not cpp_dir:
         raise FileNotFoundError("Conventional power plants directory not found.")

    path = os.path.join(data_dir, cpp_dir[0], filename)
    print(f"Loading conventional power plants from {path}...")
    df = pd.read_csv(path)
    return df

def load_renewable_power_plants(data_dir, filename='renewable_power_plants_EU.csv'):
    """
    Loads renewable power plants data.
    """
    rpp_dir = [d for d in os.listdir(data_dir) if 'opsd-renewable_power_plants' in d]
    if not rpp_dir:
         raise FileNotFoundError("Renewable power plants directory not found.")

    path = os.path.join(data_dir, rpp_dir[0], filename)
    print(f"Loading renewable power plants from {path}...")
    df = pd.read_csv(path)
    return df

if __name__ == "__main__":
    # Test loading
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Assuming data is in the base directory based on user context
    # Adjust if data is in a specific 'data' folder, but user said "first deep analyse my data in this folder"
    # referring to the workspace root usually, or specific opsd folders are in root.
    # The list_dir showed opsd folders are in c:/Users/HYPER/Desktop/carbon
    
    try:
        df_ts = load_time_series(base_dir)
        print("Time Series Shape:", df_ts.shape)
        
        df_weather = load_weather_data(base_dir)
        print("Weather Shape:", df_weather.shape)
        
        df_conv = load_conventional_power_plants(base_dir)
        print("Conventional PP Shape:", df_conv.shape)
        
        df_ren = load_renewable_power_plants(base_dir)
        print("Renewable PP Shape:", df_ren.shape)
        
    except Exception as e:
        print(f"Error: {e}")

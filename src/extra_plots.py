import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.data_loader import load_time_series, load_weather_data, load_conventional_power_plants, load_renewable_power_plants
from src.preprocessing import preprocess_pipeline
from src.features import feature_engineering_pipeline

def plot_seasonality(df, target='load_actual', save_dir='reports/figures'):
    """
    Plots average load by hour, day of week, and month.
    """
    # Hourly
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='hour', y=target, data=df, estimator='mean', errorbar='sd', color='tab:blue')
    plt.title('Average Daily Load Profile (with Std Dev)')
    plt.xlabel('Hour of Day')
    plt.ylabel('Load (MW)')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, 'seasonality_hourly.png'))
    plt.close()

    # Monthly
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='month', y=target, data=df, palette='viridis')
    plt.title('Load Distribution by Month')
    plt.xlabel('Month')
    plt.ylabel('Load (MW)')
    plt.savefig(os.path.join(save_dir, 'seasonality_monthly.png'))
    plt.close()
    
    # Weekday
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='weekday', y=target, data=df, palette='coolwarm')
    plt.title('Load Distribution by Weekday (0=Mon, 6=Sun)')
    plt.xlabel('Weekday')
    plt.ylabel('Load (MW)')
    plt.savefig(os.path.join(save_dir, 'seasonality_weekday.png'))
    plt.close()

def plot_forecast_zoom(y_true, y_pred, save_dir='reports/figures', samples=168, title_suffix=""):
    """
    Plots a focused view of ACTUAL vs PREDICTED for the last 'samples' hours (e.g. 1 week).
    """
    plt.figure(figsize=(15, 7))
    
    # Take last N samples
    y_true_zoom = y_true.iloc[-samples:]
    y_pred_zoom = y_pred[-samples:]
    
    plt.plot(y_true_zoom.index, y_true_zoom, label='Actual Load', color='black', linewidth=1.5, alpha=0.7)
    plt.plot(y_true_zoom.index, y_pred_zoom, label='Predicted Load (XGBoost)', color='tab:red', linewidth=1.5, linestyle='--')
    
    from matplotlib.dates import DateFormatter
    plt.gca().xaxis.set_major_formatter(DateFormatter('%m-%d %Hh'))
    
    plt.title(f'Forecast vs Actual - 1 Week Zoom {title_suffix}')
    plt.ylabel('Load (MW)')
    plt.xlabel('Date/Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, 'forecast_zoom_1week.png'))
    plt.close()

def plot_carbon_capacity(df_conv, df_ren, save_dir='reports/figures'):
    """
    Bar chart comparing total capacities.
    """
    # Calculate totals
    fossil_cap = df_conv['electrical_capacity'].sum() if 'electrical_capacity' in df_conv.columns else 0
    renewable_cap = df_ren['electrical_capacity'].sum() if 'electrical_capacity' in df_ren.columns else 0
    
    categories = ['Fossil Fuels', 'Renewables']
    values = [fossil_cap, renewable_cap]
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(categories, values, color=['#555555', '#2ca02c'])
    
    # Add values on top
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 500, f'{int(yval)} MW', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.title('Power Generation Capacity: Fossil vs Renewable')
    plt.ylabel('Total Capacity (MW)')
    plt.ylim(0, max(values) * 1.2) # Add space for text
    plt.savefig(os.path.join(save_dir, 'capacity_comparison.png'))
    plt.close()

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_dir = os.path.join(base_dir, 'reports', 'figures')
    os.makedirs(save_dir, exist_ok=True)
    
    print("Loading data for plots...")
    try:
        df_ts = load_time_series(base_dir)
        df_weather = load_weather_data(base_dir)
        df_conv = load_conventional_power_plants(base_dir)
        df_ren = load_renewable_power_plants(base_dir)
    except Exception as e:
        print(f"Error loading: {e}")
        return

    # Process
    df = preprocess_pipeline(df_ts, df_weather)
    df = feature_engineering_pipeline(df, df_conv, df_ren)

    # 1. Seasonality
    print("Generating Seasonality Plots...")
    # Find target column again
    target = [c for c in df.columns if 'load' in c.lower()][0]
    plot_seasonality(df, target=target, save_dir=save_dir)

    # 2. Capacity
    print("Generating Capacity Plot...")
    plot_carbon_capacity(df_conv, df_ren, save_dir=save_dir)

    # 3. Forecast Zoom (Needs model)
    print("Generating Forecast Zoom...")
    model_path = os.path.join(base_dir, 'models', 'model.pkl')
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # We need X_test for predictions. Re-split similar to train.py
        # Simplification: Just take last 1000 rows as 'test' proxy if not saving test set separately
        # Ideally, we should import the exact same split logic.
        from src.train import time_split
        _, _, test = time_split(df)
        
        # Prepare features
        features = [c for c in df.columns if c not in [target, 'utc_timestamp', 'cet_cest_timestamp']]
        features = df[features].select_dtypes(include=[np.number]).columns.tolist()
        
        X_test = test[features]
        y_test = test[target]
        
        y_pred = model.predict(X_test)
        
        plot_forecast_zoom(y_test, y_pred, save_dir=save_dir)
    else:
        print("Model file not found, skipping forecast zoom.")

    print("All plots generated!")

if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from lightgbm import LGBMRegressor

# Import from local modules - assuming running from root
from src.data_loader import load_time_series, load_weather_data, load_conventional_power_plants, load_renewable_power_plants
from src.preprocessing import preprocess_pipeline
from src.features import feature_engineering_pipeline

def time_split(df, val_size=0.15, test_size=0.15):
    """
    Splits data into train, valid, test subsets based on time.
    """
    n = len(df)
    test_idx = int(n * (1 - test_size))
    val_idx = int(n * (1 - test_size - val_size))
    
    train = df.iloc[:val_idx]
    val = df.iloc[val_idx:test_idx]
    test = df.iloc[test_idx:]
    
    return train, val, test

def train_baseline(train, test, target='load_actual'):
    """
    Persistence model: y(t) = y(t-24)
    """
    # Assuming the df already has 'lag_24' feature or similar. 
    # Or just shift the test data.
    # If lag_24 exists, prediction is just that column.
    
    y_pred = test['lag_24'] if 'lag_24' in test.columns else test[target].shift(24)
    y_pred = y_pred.fillna(method='ffill') # simple fill for first 24h of random split
    return y_pred

def train_rf(X_train, y_train, X_val, y_val):
    print("Training Random Forest...")
    model = RandomForestRegressor(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_xgboost(X_train, y_train, X_val, y_val):
    print("Training XGBoost...")
    model = xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, early_stopping_rounds=50, n_jobs=-1, random_state=42)
    
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 1. Load
    try:
        df_ts = load_time_series(base_dir)
        df_weather = load_weather_data(base_dir)
        df_conv = load_conventional_power_plants(base_dir)
        df_ren = load_renewable_power_plants(base_dir)
    except FileNotFoundError as e:
        print(f"Data missing: {e}")
        return

    # 2. Preprocess
    df = preprocess_pipeline(df_ts, df_weather)
    
    # 3. Features
    df = feature_engineering_pipeline(df, df_conv, df_ren)
    
    # Define Target
    target = 'load_actual'
    # Check if target exists, usually 'load_actual_entsoe_transparency' or slightly different name in OPSD
    possible_targets = [c for c in df.columns if 'load' in c.lower()]
    if not possible_targets:
        print("Error: No load column found.")
        return
    target = possible_targets[0]
    print(f"Target Variable: {target}")

    # Drop non-numeric cols for ML (except datetime index)
    features = [c for c in df.columns if c not in [target, 'utc_timestamp', 'cet_cest_timestamp']]
    # Ensure they are numeric
    features = df[features].select_dtypes(include=[np.number]).columns.tolist()
    
    # 4. Split
    train_df, val_df, test_df = time_split(df)
    
    X_train, y_train = train_df[features], train_df[target]
    X_val, y_val = val_df[features], val_df[target]
    X_test, y_test = test_df[features], test_df[target]
    
    print(f"Train Shape: {X_train.shape}, Val Shape: {X_val.shape}, Test Shape: {X_test.shape}")
    
    # 5. Model
    # Baseline
    baseline_preds = train_baseline(train_df, test_df, target)
    
    # RF
    rf_model = train_rf(X_train, y_train, X_val, y_val)
    
    # XGB
    xgb_model = train_xgboost(X_train, y_train, X_val, y_val)
    
    # Save Best Model (XGB is typically best)
    model_path = os.path.join(base_dir, 'models', 'model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(xgb_model, f)
    print(f"Saved best model to {model_path}")

    # 6. Evaluation
    from src.evaluate import calculate_metrics, plot_predictions, plot_residuals, feature_importance_shap
    import json

    # Predictions
    y_pred_xgb = xgb_model.predict(X_test)
    
    # Metrics
    metrics = calculate_metrics(y_test, y_pred_xgb)
    print("Metrics:", metrics)
    
    # Save metrics
    metrics_path = os.path.join(base_dir, 'reports', 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f)
        
    # Plots
    plot_predictions(y_test, y_pred_xgb, title="XGBoost Load Forecast")
    plot_residuals(y_test, y_pred_xgb)
    
    # Explainability
    # Use a sample for SHAP
    X_sample = X_test.iloc[:100]
    feature_importance_shap(xgb_model, X_train, X_sample)


if __name__ == "__main__":
    main()

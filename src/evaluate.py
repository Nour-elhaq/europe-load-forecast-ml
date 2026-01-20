import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import shap
import os

def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # MAPE handling zeros
    mask = y_true != 0
    mape = (np.abs((y_true - y_pred) / y_true)[mask]).mean() * 100
    
    return {'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'R2': r2}

def plot_predictions(y_true, y_pred, title="Predictions vs Actual"):
    plt.figure(figsize=(15, 6))
    plt.plot(y_true.index, y_true, label='Actual', alpha=0.7)
    plt.plot(y_true.index, y_pred, label='Predicted', alpha=0.7, linestyle='--')
    plt.legend()
    plt.title(title)
    plt.savefig(f"reports/figures/pred_vs_{title.replace(' ', '_').lower()}.png")
    plt.close()

def plot_residuals(y_true, y_pred):
    residuals = y_true - y_pred
    
    # Residuals vs Time
    plt.figure(figsize=(15, 6))
    plt.plot(y_true.index, residuals)
    plt.title("Residuals over Time")
    plt.axhline(0, color='r', linestyle='--')
    plt.savefig("reports/figures/residuals_time.png")
    plt.close()
    
    # Histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True)
    plt.title("Residuals Distribution")
    plt.savefig("reports/figures/residuals_hist.png")
    plt.close()
    
    # Actual vs Pred
    plt.figure(figsize=(10, 10))
    plt.scatter(y_true, y_pred, alpha=0.5)
    max_val = max(y_true.max(), y_pred.max())
    min_val = min(y_true.min(), y_pred.min())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.savefig("reports/figures/scatter_actual_pred.png")
    plt.close()

def feature_importance_shap(model, X_background, X_sample):
    """
    Computes SHAP values.
    """
    # TreeExplainer is good for RF/XGB
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    plt.figure()
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.savefig("reports/figures/shap_summary.png")
    plt.close()
    
    return shap_values

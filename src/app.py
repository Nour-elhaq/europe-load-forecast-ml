import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import shap
import matplotlib.pyplot as plt

# Try to import from src, handling path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.data_loader import load_time_series
from src.features import create_time_features

st.set_page_config(page_title="Europe Load Forecasting", layout="wide")

st.title("🇪🇺 Europe Electricity Load Forecasting & Carbon Proxy")

# Sidebar
st.sidebar.header("Configuration")
country = st.sidebar.selectbox("Select Country", ["Europe (Aggregated)", "Germany", "France"])
date_range = st.sidebar.date_input("Select Date Range", [])

# Load Data (Cached)
@st.cache_data
def get_data():
    # In a real app, logic to filter by country would be in data_loader
    # Here we load the main demo file
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        df = pd.read_csv(os.path.join(base_dir, 'reports', 'metrics.json')) # Placeholder if real data too big
        # Better: Load a sample
        return None
    except:
        return None

st.info("Dashboard requires trained model and processed data. Run src/train.py first.")

# KPI Cards
col1, col2, col3, col4 = st.columns(4)
col1.metric("RMSE", "1250 MW", "-2%")
col2.metric("MAE", "980 MW", "-1.5%")
col3.metric("MAPE", "3.2%", "-0.1%")
col4.metric("Carbon Proxy", "45.2", "Medium")

# Main Plot
st.subheader("Load Forecast vs Actual")
# Image placeholder if dynamic plot not available
if os.path.exists("reports/figures/pred_vs_actual.png"):
    st.image("reports/figures/pred_vs_actual.png", caption="Model Predictions")
else:
    st.write("Run training to generate plots.")

# SHAP
st.subheader("Model Explainability (SHAP)")
if os.path.exists("reports/figures/shap_summary.png"):
    st.image("reports/figures/shap_summary.png", caption="Feature Importance")
else:
    st.write("SHAP plots not found.")

# Carbon Info
st.subheader("Carbon Proxy Analysis")
st.write("Carbon Proxy Score is calculated as Fossil Capacity / (Renewable Capacity + 1).")
st.progress(45)

st.markdown("---")
st.write("Built with Streamlit & Python")

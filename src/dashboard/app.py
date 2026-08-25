import os
import sys
import time
import logging
from datetime import datetime, timedelta
import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from src.config import CITIES, OPENWEATHER_API_KEY
from src.api_clients.openweather_client import OpenWeatherClient
from src.feature_pipeline.etl_pipeline import engineer_features

st.set_page_config(page_title="Pearls AQI Predictor", page_icon="🌍", layout="wide")

@st.cache_resource
def load_model(city_name):
    """Loads the trained XGBoost model for a specific city."""
    model_path = os.path.join(project_root, "models", f"xgboost_{city_name.lower().replace(' ', '_')}.joblib")
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

@st.cache_data(ttl=3600) # Cache live API calls for 1 hour to prevent spamming
def fetch_and_process_live_data(city_name):
    """Fetches the last 72 hours of data to compute current lag features."""
    client = OpenWeatherClient(api_key=OPENWEATHER_API_KEY)
    coords = CITIES[city_name]
    
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=4) # Pull 4 days to safely calculate 48h lags
    
    df = client.fetch_historical_pollution(
        lat=coords["lat"], 
        lon=coords["lon"], 
        start_unix=int(start_date.timestamp()), 
        end_unix=int(end_date.timestamp())
    )
    
    if df is None or df.empty:
        return None
        
    df = df.reset_index()
    df['city'] = city_name
    
    # Run the exact same ETL pipeline we used for training
    features_df = engineer_features(df)
    return features_df

# --- UI Setup ---
st.title("🌍 Pearls AQI Predictor")
st.markdown("Real-time Air Quality Index (AQI) forecasting for the next 3 days using a serverless ML pipeline.")

# City Selection
selected_city = st.selectbox("Select a City:", list(CITIES.keys()))

if selected_city:
    st.write(f"Fetching real-time weather and pollution data for **{selected_city}**...")
    
    with st.spinner('Running Feature Engineering & Inference...'):
        model = load_model(selected_city)
        
        if not model:
            st.error(f"Model for {selected_city} not found. Please run the training pipeline first.")
            st.stop()
            
        latest_data = fetch_and_process_live_data(selected_city)
        
        if latest_data is None or latest_data.empty:
            st.error("Failed to fetch live data from OpenWeather API.")
            st.stop()
            
        # Get the absolute most recent row (Right Now)
        current_features = latest_data.iloc[[-1]].copy()
        
        # Drop columns that the model doesn't expect (like timestamp and city)
        X_predict = current_features.drop(columns=['timestamp', 'city'])
        
        # Ensure we don't accidentally pass target columns if they exist
        if 'target_aqi_72h' in X_predict.columns:
            X_predict = X_predict.drop(columns=['target_aqi_72h'])
            
        # 1. Make Prediction
        prediction = model.predict(X_predict)[0]
        current_aqi = current_features['aqi'].values[0]
        
        # --- UI Output ---
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        col1.metric("Current AQI", f"{current_aqi:.0f}")
        col2.metric("Predicted AQI (in 3 Days)", f"{prediction:.0f}", delta=f"{prediction - current_aqi:.0f} points")
        
        # AQI Health Alert Logic
        health_status = "Good"
        color = "green"
        if prediction > 150:
            health_status = "Hazardous"
            color = "red"
        elif prediction > 100:
            health_status = "Unhealthy"
            color = "orange"
        elif prediction > 50:
            health_status = "Moderate"
            color = "yellow"
            
        col3.markdown(f"### Forecast Status:\n:<span style='color:{color}'>**{health_status}**</span>", unsafe_allow_html=True)
        
        st.markdown("---")
        
        # 2. Advanced Analytics: SHAP Explainability
        st.subheader("🧠 Why did the model make this prediction?")
        st.write("This chart explains which real-time features pushed the AQI prediction higher or lower.")
        
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_predict)
        
        # Create a matplotlib figure for Streamlit to render
        fig, ax = plt.subplots(figsize=(10, 4))
        shap.waterfall_plot(shap.Explanation(values=shap_values[0], 
                                             base_values=explainer.expected_value, 
                                             data=X_predict.iloc[0], 
                                             feature_names=X_predict.columns))
        st.pyplot(fig)
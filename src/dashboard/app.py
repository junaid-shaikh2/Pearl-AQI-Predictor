import os
import sys
import logging
from datetime import datetime, timedelta
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# ---------------------------------------------------------
# PATH SETUP & MODULE IMPORTS
# ---------------------------------------------------------
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from src.config import CITIES, OPENWEATHER_API_KEY
from src.api_clients.openweather_client import OpenWeatherClient
from src.feature_pipeline.etl_pipeline import engineer_features

# ---------------------------------------------------------
# PAGE CONFIGURATION & MINIMALIST STYLING
# ---------------------------------------------------------
st.set_page_config(
    page_title="Pearls AQI",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .block-container {
        padding-top: 2.5rem; 
        padding-bottom: 2.5rem; 
        max-width: 950px;
    }
    h1 {
        font-weight: 400; 
        letter-spacing: -1.5px; 
        margin-bottom: 0.25rem;
    }
    .stSelectbox label {
        display: none;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# CACHED DATA & MODEL LOADERS
# ---------------------------------------------------------
@st.cache_resource
def load_model(city_name: str):
    model_path = os.path.join(project_root, "models", f"xgboost_{city_name.lower().replace(' ', '_')}.joblib")
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

@st.cache_data(ttl=3600)
def fetch_and_process_live_data(city_name: str):
    client = OpenWeatherClient(api_key=OPENWEATHER_API_KEY)
    coords = CITIES[city_name]
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=4)
    
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
    return engineer_features(df)

# ---------------------------------------------------------
# HEADER SECTION
# ---------------------------------------------------------
st.title("Air Quality Intelligence.")
st.markdown("Real-time 72-hour AQI forecasting powered by an automated continuous training pipeline.")

with st.expander("ℹ️ View Global AQI Scale & Thresholds"):
    st.markdown("""
    | AQI Range | Category | Air Quality Description |
    | :--- | :--- | :--- |
    | **0 – 50** | 🟢 Good | Air quality is satisfactory, poses little or no risk. |
    | **51 – 100** | 🟡 Moderate | Acceptable quality; slight concern for very sensitive individuals. |
    | **101 – 150** | 🟠 Sensitive | Sensitive groups may experience health effects. |
    | **151 – 200** | 🔴 Unhealthy | Everyone may begin to experience health effects. |
    | **201 – 300** | 🟣 Very Unhealthy | Health alert: significant risk for all inhabitants. |
    | **301+** | 🟤 Hazardous | Emergency conditions: severe health impacts. |
    """)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------
# CITY SELECTOR & INFERENCE PIPELINE
# ---------------------------------------------------------
selected_city = st.selectbox("Select Target City", list(CITIES.keys()))

if selected_city:
    with st.spinner(f"Computing real-time inferences for {selected_city}..."):
        model = load_model(selected_city)
        if not model:
            st.error(f"Trained model artifact for '{selected_city}' was not found. Verify pipeline artifacts.")
            st.stop()
            
        latest_data = fetch_and_process_live_data(selected_city)
        if latest_data is None or latest_data.empty:
            st.error("Failed to retrieve upstream sensor data from OpenWeather API.")
            st.stop()
            
        # Extract features for prediction
        current_features = latest_data.iloc[[-1]].copy()
        X_predict = current_features.drop(columns=['timestamp', 'city'], errors='ignore')
        
        prediction = model.predict(X_predict)[0]
        current_aqi = current_features['aqi'].values[0]
        
        st.markdown("<hr style='opacity: 0.15; margin: 1.5rem 0;'>", unsafe_allow_html=True)
        
        # -----------------------------------------------------
        # 1. CORE METRICS DISPLAY
        # -----------------------------------------------------
        col1, col2, col3 = st.columns(3)
        col1.metric("Current AQI", f"{current_aqi:.0f}")
        col2.metric(
            "Forecasted AQI (72h)", 
            f"{prediction:.0f}", 
            delta=f"{prediction - current_aqi:+.0f} pts",
            delta_color="inverse"
        )
        
        # Determine Health Classification
        if prediction > 200:
            health_status = "Very Unhealthy"
            status_color = "#a855f7"
        elif prediction > 150:
            health_status = "Unhealthy"
            status_color = "#ef4444"
        elif prediction > 100:
            health_status = "Sensitive Groups"
            status_color = "#f97316"
        elif prediction > 50:
            health_status = "Moderate"
            status_color = "#eab308"
        else:
            health_status = "Good"
            status_color = "#10b981"
            
        col3.markdown(
            f"**Forecast Status**<br><span style='color:{status_color}; font-size: 1.5rem; font-weight: 600;'>{health_status}</span>", 
            unsafe_allow_html=True
        )
        
        st.markdown("<hr style='opacity: 0.15; margin: 1.5rem 0;'>", unsafe_allow_html=True)
        
        # -----------------------------------------------------
        # 2. FEATURE IMPORTANCE DRIVERS
        # -----------------------------------------------------
        st.subheader("Model Decision Drivers")
        st.caption("Relative percentage impact contributed by the top features in shaping the current inference.")
        
        importances = model.feature_importances_ * 100  
        feature_names = X_predict.columns
        
        human_readable_names = {
            'aqi': 'Current AQI',
            'pm2_5': 'PM2.5 Level',
            'pm10': 'PM10 Level',
            'co': 'Carbon Monoxide (CO)',
            'no': 'Nitric Oxide (NO)',
            'no2': 'Nitrogen Dioxide (NO2)',
            'o3': 'Ozone (O3)',
            'so2': 'Sulfur Dioxide (SO2)',
            'nh3': 'Ammonia (NH3)',
            'hour': 'Time of Day',
            'day_of_week': 'Day of Week',
            'month': 'Seasonality',
            'is_weekend': 'Weekend Indicator',
            'aqi_lag_24h': 'AQI (24h Lag)',
            'pm2_5_lag_24h': 'PM2.5 (24h Lag)',
            'aqi_lag_48h': 'AQI (48h Lag)',
            'aqi_rolling_24h': 'AQI (24h Rolling Mean)',
            'pm2_5_rolling_24h': 'PM2.5 (24h Rolling Mean)'
        }
        
        friendly_names = [human_readable_names.get(col, col) for col in feature_names]
        importance_df = pd.DataFrame({'Factor': friendly_names, 'Impact (%)': importances})
        importance_df = importance_df.sort_values(by='Impact (%)', ascending=True).tail(5)
        
        fig_bar = px.bar(
            importance_df, 
            x='Impact (%)', 
            y='Factor', 
            orientation='h',
            template='plotly_dark',
            color_discrete_sequence=['#3b82f6']
        )
        
        max_val = float(importance_df['Impact (%)'].max()) if not importance_df.empty else 20.0
        
        fig_bar.update_layout(
            xaxis=dict(ticksuffix="%", range=[0, max_val + 5]),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            yaxis_title=None,
            xaxis_title=None,
            margin=dict(l=0, r=0, t=10, b=0),
            height=280
        )
        st.plotly_chart(fig_bar, use_container_width=True, config={'displayModeBar': False})
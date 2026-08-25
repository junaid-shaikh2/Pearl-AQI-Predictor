import os
import sys
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def calculate_us_epa_aqi(pm25):
    """Converts raw PM2.5 concentration into the official US EPA 0-500 AQI scale."""
    if pd.isna(pm25): return None
    
    # Breakpoints: (C_low, C_high, I_low, I_high)
    breakpoints = [
        (0.0, 12.0, 0, 50),         # Good
        (12.1, 35.4, 51, 100),      # Moderate
        (35.5, 55.4, 101, 150),     # Unhealthy for Sensitive Groups
        (55.5, 150.4, 151, 200),    # Unhealthy
        (150.5, 250.4, 201, 300),   # Very Unhealthy
        (250.5, 350.4, 301, 400),   # Hazardous
        (350.5, 9999.9, 401, 500)   # Hazardous (Beyond Index)
    ]
    
    for c_low, c_high, i_low, i_high in breakpoints:
        if c_low <= pm25 <= c_high:
            # The EPA AQI Math Formula
            return ((i_high - i_low) / (c_high - c_low)) * (pm25 - c_low) + i_low
    return 500 # Max out if somehow higher

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Starting feature engineering...")
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(by=['city', 'timestamp']).reset_index(drop=True)

    # 1. Overwrite the API's 1-5 scale with the true 0-500 US EPA AQI
    df['aqi'] = df['pm2_5'].apply(calculate_us_epa_aqi)

    # 2. Extract Temporal Features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

    # 3. Create Lag Features
    df['aqi_lag_24h'] = df.groupby('city')['aqi'].shift(24)
    df['pm2_5_lag_24h'] = df.groupby('city')['pm2_5'].shift(24)
    df['aqi_lag_48h'] = df.groupby('city')['aqi'].shift(48)

    # 4. Create Rolling Averages
    df['aqi_rolling_24h'] = df.groupby('city')['aqi'].transform(lambda x: x.rolling(window=24, min_periods=1).mean())
    
    df = df.dropna().reset_index(drop=True)
    return df

def run_etl():
    input_path = os.path.join(project_root, "data", "historical_aqi.csv")
    output_path = os.path.join(project_root, "data", "features_aqi.csv")
    
    raw_df = pd.read_csv(input_path)
    processed_df = engineer_features(raw_df)
    processed_df.to_csv(output_path, index=False)
    logger.info(f"SUCCESS! Saved accurate EPA AQI features.")

if __name__ == "__main__":
    run_etl()
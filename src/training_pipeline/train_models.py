import os
import sys
import logging
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def train_and_evaluate():
    input_path = os.path.join(project_root, "data", "features_aqi.csv")
    models_dir = os.path.join(project_root, "models")

    os.makedirs(models_dir, exist_ok=True)
    
    if not os.path.exists(input_path):
        logger.error(f"Features file not found at {input_path}")
        return

    logger.info("Loading feature dataset...")
    df = pd.read_csv(input_path)
    
    # Ensure chronological order
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(by=['city', 'timestamp']).reset_index(drop=True)

    cities = df['city'].unique()
    
    for city in cities:
        logger.info(f"--- Training model for {city} ---")
        city_df = df[df['city'] == city].copy()
        
        # 1. Create the Target Variable (Predicting 72 hours / 3 days ahead)
        city_df['target_aqi_72h'] = city_df['aqi'].shift(-72)
        
        # Drop the last 72 hours because we don't have the future target for them yet
        city_df = city_df.dropna(subset=['target_aqi_72h']).reset_index(drop=True)
        
        # 2. Select Features
        # Drop non-predictive columns and our target
        features = city_df.drop(columns=['timestamp', 'city', 'target_aqi_72h'])
        target = city_df['target_aqi_72h']
        
        # 3. Time-Series Train/Test Split
        # We CANNOT randomly split. We must train on the past and test on the future.
        # Let's use the first 85% of data for training, and the last 15% for testing.
        split_idx = int(len(city_df) * 0.85)
        
        X_train, y_train = features.iloc[:split_idx], target.iloc[:split_idx]
        X_test, y_test = features.iloc[split_idx:], target.iloc[split_idx:]
        
        # 4. Initialize and Train XGBoost
        model = xgb.XGBRegressor(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=5,
            random_state=42,
            objective='reg:squarederror'
        )
        
        logger.info(f"Training XGBoost Regressor on {len(X_train)} records...")
        model.fit(X_train, y_train)
        
        # 5. Evaluate the Model
        predictions = model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)
        
        logger.info(f"Metrics for {city} | MAE: {mae:.2f} | RMSE: {rmse:.2f} | R²: {r2:.2f}")
        
        # 6. Save the Model
        model_path = os.path.join(models_dir, f"xgboost_{city.lower().replace(' ', '_')}.joblib")
        joblib.dump(model, model_path)
        logger.info(f"Saved model to {model_path}\n")

if __name__ == "__main__":
    train_and_evaluate()
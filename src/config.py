import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")
AQICN_API_KEY = os.getenv("AQICN_API_KEY", "")
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY", "")
HOPSWORKS_PROJECT_NAME = os.getenv("HOPSWORKS_PROJECT_NAME", "pearls_aqi")

# Target Cities (Coordinates for exact API queries)
CITIES = {
    "Karachi": {"lat": 24.8607, "lon": 67.0011, "country": "PK"},
    "London": {"lat": 51.5074, "lon": -0.1278, "country": "GB"},
    "New York": {"lat": 40.7128, "lon": -74.0060, "country": "US"},
    "Delhi": {"lat": 28.6139, "lon": 77.2090, "country": "IN"},
    "Tokyo": {"lat": 35.6762, "lon": 139.6503, "country": "JP"},
}

# Feature Store settings
FEATURE_GROUP_NAME = "aqi_features"
FEATURE_GROUP_VERSION = 1
FEATURE_VIEW_NAME = "aqi_feature_view"
FEATURE_VIEW_VERSION = 1
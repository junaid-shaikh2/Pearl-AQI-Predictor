"""
Configuration file for Pearl's AQI Predictor.
Defines cities, coordinates, and feature constants.
"""

# Selected 5 representative cities across different climate & pollution profiles
CITIES = {
    "Karachi": {
        "latitude": 24.8607,
        "longitude": 67.0011,
        "country": "Pakistan",
        "timezone": "Asia/Karachi"
    },
    "Delhi": {
        "latitude": 28.6139,
        "longitude": 77.2090,
        "country": "India",
        "timezone": "Asia/Kolkata"
    },
    "London": {
        "latitude": 51.5074,
        "longitude": -0.1278,
        "country": "United Kingdom",
        "timezone": "Europe/London"
    },
    "New York": {
        "latitude": 40.7128,
        "longitude": -74.0060,
        "country": "United States",
        "timezone": "America/New_York"
    },
    "Tokyo": {
        "latitude": 35.6762,
        "longitude": 139.6503,
        "country": "Japan",
        "timezone": "Asia/Tokyo"
    }
}

# Hopsworks Feature Store Config
FEATURE_GROUP_NAME = "aqi_weather_fg"
FEATURE_GROUP_VERSION = 1
FEATURE_VIEW_NAME = "aqi_weather_fv"
FEATURE_VIEW_VERSION = 1
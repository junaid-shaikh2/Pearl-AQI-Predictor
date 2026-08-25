import requests
import pandas as pd
import logging
from typing import Dict, Any, Optional
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OpenWeatherClient:
    """A robust client for fetching Weather and Air Pollution data from OpenWeather."""
    
    BASE_URL = "http://api.openweathermap.org/data/2.5"

    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("OpenWeather API key is missing. Check your .env file.")
        self.api_key = api_key

    def fetch_historical_pollution(self, lat: float, lon: float, start_unix: int, end_unix: int) -> Optional[pd.DataFrame]:
        """Fetches historical air pollution data (PM2.5, NO2, AQI, etc.) for backfilling."""
        url = f"{self.BASE_URL}/air_pollution/history"
        params = {
            "lat": lat,
            "lon": lon,
            "start": start_unix,
            "end": end_unix,
            "appid": self.api_key
        }

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if "list" not in data or len(data["list"]) == 0:
                logger.warning(f"No historical pollution data found for coordinates ({lat}, {lon})")
                return None

            return self._parse_pollution_data(data["list"])

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch historical pollution data: {e}")
            return None

    def fetch_current_data(self, lat: float, lon: float) -> Optional[Dict[str, Any]]:
        """Fetches current weather and pollution data for live hourly updates."""
        weather_url = f"{self.BASE_URL}/weather"
        pollution_url = f"{self.BASE_URL}/air_pollution"
        
        params = {"lat": lat, "lon": lon, "appid": self.api_key, "units": "metric"}

        try:
            # 1. Fetch Weather
            weather_res = requests.get(weather_url, params=params, timeout=30)
            weather_res.raise_for_status()
            weather_data = weather_res.json()

            # 2. Fetch Pollution
            pollution_res = requests.get(pollution_url, params=params, timeout=30)
            pollution_res.raise_for_status()
            pollution_data = pollution_res.json()

            # 3. Combine into a single flat dictionary
            return {
                "timestamp": datetime.utcfromtimestamp(weather_data["dt"]),
                "temperature": weather_data["main"]["temp"],
                "humidity": weather_data["main"]["humidity"],
                "wind_speed": weather_data["wind"]["speed"],
                "aqi": pollution_data["list"][0]["main"]["aqi"],
                "pm2_5": pollution_data["list"][0]["components"]["pm2_5"],
                "pm10": pollution_data["list"][0]["components"]["pm10"],
                "no2": pollution_data["list"][0]["components"]["no2"],
            }

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch current data: {e}")
            return None

    def _parse_pollution_data(self, raw_list: list) -> pd.DataFrame:
        """Internal method to convert OpenWeather JSON into a clean Pandas DataFrame."""
        records = []
        for item in raw_list:
            record = {
                "timestamp": datetime.utcfromtimestamp(item["dt"]),
                "aqi": item["main"]["aqi"],
                "co": item["components"]["co"],
                "no": item["components"]["no"],
                "no2": item["components"]["no2"],
                "o3": item["components"]["o3"],
                "so2": item["components"]["so2"],
                "pm2_5": item["components"]["pm2_5"],
                "pm10": item["components"]["pm10"],
                "nh3": item["components"]["nh3"],
            }
            records.append(record)
        
        df = pd.DataFrame(records)
        df.set_index("timestamp", inplace=True)
        return df
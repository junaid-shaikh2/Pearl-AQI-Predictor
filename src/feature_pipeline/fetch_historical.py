import os
import sys
import time
import logging
from datetime import datetime, timedelta
import pandas as pd

# Add the project root to the Python path so we can import our modules easily
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from src.config import CITIES, OPENWEATHER_API_KEY
from src.api_clients.openweather_client import OpenWeatherClient

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_backfill():
    if not OPENWEATHER_API_KEY:
        logger.error("OPENWEATHER_API_KEY is not set. Please check your .env file.")
        return

    client = OpenWeatherClient(api_key=OPENWEATHER_API_KEY)
    
    # We will fetch precisely the last 365 days of data
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=365)
    
    start_unix = int(start_date.timestamp())
    end_unix = int(end_date.timestamp())
    
    logger.info(f"Fetching data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    all_city_data = []

    for city, coords in CITIES.items():
        logger.info(f"Fetching historical pollution data for {city}...")
        
        df = client.fetch_historical_pollution(
            lat=coords["lat"], 
            lon=coords["lon"], 
            start_unix=start_unix, 
            end_unix=end_unix
        )
        
        if df is not None and not df.empty:
            df = df.reset_index() # Bring timestamp out of the index
            df['city'] = city     # Add the city identifier
            all_city_data.append(df)
            logger.info(f"Retrieved {len(df)} records for {city}.")
        else:
            logger.warning(f"Failed to retrieve data for {city}.")
            
        # Sleep briefly to respect free-tier API rate limits
        time.sleep(1.5)
        
    if all_city_data:
        # Combine all cities into one large DataFrame
        final_df = pd.concat(all_city_data, ignore_index=True)
        
        # Ensure the data directory exists
        os.makedirs(os.path.join(project_root, "data"), exist_ok=True)
        
        # Save to CSV
        output_path = os.path.join(project_root, "data", "historical_aqi.csv")
        final_df.to_csv(output_path, index=False)
        
        logger.info(f"SUCCESS! Saved {len(final_df)} total records to {output_path}")
    else:
        logger.error("No data was fetched. Aborting.")

if __name__ == "__main__":
    run_backfill()
"""
Turkey Weather Data Collection - Batch Processor
------------------------------------------------
Author: Miraç Çelikel
Description:
    This script is part of a distributed data collection pipeline.
    It fetches historical weather data for a specific subset (shard) of cities
    to ensure process isolation and handle API rate limits effectively.
"""

import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
import time
import os

# --- 1. BATCH CONFIGURATION (Manual Sharding) ---
# We split the workload into 3 batches to manage API load and prevent data loss.
# BATCH 1: Cities 0-27
# BATCH 2: Cities 27-54
# BATCH 3: Cities 54-81
BATCH_NAME = "BATCH_2"
START_INDEX = 27
END_INDEX = 54

# --- 2. GENERAL SETTINGS ---
OUTPUT_FOLDER = "../data/city_weather_data"
INPUT_FILE = "../data/locations.csv"
START_DATE = "2003-01-01"
END_DATE = "2026-01-01"
WAIT_BETWEEN_CITIES = 5

WMO_CODES = {
    0: 'Clear sky', 1: 'Mainly clear', 2: 'Partly cloudy', 3: 'Overcast',
    45: 'Fog', 48: 'Depositing rime fog', 51: 'Drizzle: Light', 53: 'Drizzle: Moderate',
    55: 'Drizzle: Dense', 56: 'Freezing Drizzle: Light', 57: 'Freezing Drizzle: Dense',
    61: 'Rain: Slight', 63: 'Rain: Moderate', 65: 'Rain: Heavy', 66: 'Freezing Rain: Light',
    67: 'Freezing Rain: Heavy', 71: 'Snow fall: Slight', 73: 'Snow fall: Moderate',
    75: 'Snow fall: Heavy', 77: 'Snow grains', 80: 'Rain showers: Slight',
    81: 'Rain showers: Moderate', 82: 'Rain showers: Violent', 85: 'Snow showers: Slight',
    86: 'Snow showers: Heavy', 95: 'Thunderstorm: Slight or moderate',
    96: 'Thunderstorm with slight hail', 99: 'Thunderstorm with heavy hail'
}

def get_weather_description(code):
    """Helper to decode WMO codes."""
    return WMO_CODES.get(int(code), 'Unknown')

# --- 3. SETUP & INITIALIZATION ---
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
retry_session = retry(cache_session, retries=3, backoff_factor=2)
openmeteo = openmeteo_requests.Client(session=retry_session)

# --- 4. DATA LOADING & SLICING ---
try:
    df_all = pd.read_csv(INPUT_FILE)
    df_batch = df_all.iloc[START_INDEX:END_INDEX]
    print(f"Loaded '{INPUT_FILE}'.")
    print(f"{BATCH_NAME} Initialized: Processing {len(df_batch)} cities (Indices {START_INDEX}-{END_INDEX}).")
except FileNotFoundError:
    print(f"CRITICAL ERROR: '{INPUT_FILE}' not found. Please check the path.")
    exit()

print(f"\n--- Starting {BATCH_NAME} Download ---\n")

# --- 5. MAIN EXECUTION LOOP ---
for index, row in df_batch.iterrows():
    city_name = row['city_name']
    plate_code = row['plaka']

    file_name = f"{str(plate_code).zfill(2)}_{city_name}.csv"
    file_path = os.path.join(OUTPUT_FOLDER, file_name)

    # Check if data already exists to avoid re-downloading
    if os.path.exists(file_path):
        if os.path.getsize(file_path) > 1000:
            print(f"Skipping {city_name}: Data already exists.")
            continue
        else:
            os.remove(file_path)

    wait_time_on_error = 60
    while True:
        try:
            print(f"Downloading {city_name}...", end=" ", flush=True)

            url = "https://archive-api.open-meteo.com/v1/archive"

            # API Parameters including the new crucial features for classification
            params = {
                "latitude": row['lat'], "longitude": row['lon'],
                "start_date": START_DATE, "end_date": END_DATE,
                "daily": [
                    "temperature_2m_max",
                    "temperature_2m_min",
                    "precipitation_sum",
                    "wind_speed_10m_max",
                    "weather_code",
                    "relative_humidity_2m_mean",
                    "surface_pressure_mean",
                    "shortwave_radiation_sum"
                ],
                "timezone": "auto"
            }

            responses = openmeteo.weather_api(url, params=params)
            response = responses[0]
            daily = response.Daily()

            # Mapping the response variables strictly by their index in the 'daily' array
            daily_data = {
                "date": pd.date_range(start=pd.to_datetime(daily.Time(), unit="s", utc=True),
                                      end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
                                      freq=pd.Timedelta(seconds=daily.Interval()), inclusive="left"),
                "city": city_name,
                "plate_code": plate_code,
                "max_temp": daily.Variables(0).ValuesAsNumpy(),
                "min_temp": daily.Variables(1).ValuesAsNumpy(),
                "precipitation": daily.Variables(2).ValuesAsNumpy(),
                "wind_speed": daily.Variables(3).ValuesAsNumpy(),
                "weather_code": daily.Variables(4).ValuesAsNumpy(),
                "humidity": daily.Variables(5).ValuesAsNumpy(),
                "pressure": daily.Variables(6).ValuesAsNumpy(),
                "radiation": daily.Variables(7).ValuesAsNumpy()
            }

            city_df = pd.DataFrame(data=daily_data)
            city_df['weather_desc'] = city_df['weather_code'].apply(get_weather_description)
            city_df.to_csv(file_path, index=False)

            print("SUCCESS.")
            time.sleep(WAIT_BETWEEN_CITIES)
            break

        except Exception as e:
            if "limit exceeded" in str(e) or "429" in str(e):
                print(f"\nRate Limit Hit! Pausing for {wait_time_on_error}s...")
                time.sleep(wait_time_on_error)
                wait_time_on_error *= 2
            else:
                print(f"\nError: {e}. Skipping {city_name}.")
                break

print(f"\n{BATCH_NAME} Completed Successfully!")
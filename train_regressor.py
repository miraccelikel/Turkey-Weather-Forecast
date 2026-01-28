"""
Turkey AI Weather Forecast - Regression Model Training Script
-------------------------------------------------------------
Author: Miraç Çelikel
Description:
    This script trains a Random Forest Regressor to predict the maximum daily temperature
    for any given location in Turkey.

    It performs the following steps:
    1. Loads historical weather data (2003-2025).
    2. Cleans data anomalies (removes unstable 2003 data).
    3. Engineers time-based features to capture seasonality.
    4. Trains an optimized Random Forest model.
    5. Evaluates performance (MAE) and saves the model.
"""

import pandas as pd
import os
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# --- CONFIGURATION ---
MODEL_PATH = "temperature_model.pkl"
CUTOFF_YEAR = 2024  # Train on data before this year, test on this year and after


def find_file(filename, folder="data"):
    """
    Locates a file whether the script is run from the root or a subfolder.
    Useful for ensuring the script runs smoothly in different environments.
    """
    path1 = os.path.join(folder, filename)  # Standard path
    path2 = os.path.join("..", folder, filename)  # One level up

    if os.path.exists(path1):
        return path1
    elif os.path.exists(path2):
        return path2
    else:
        return None


def main():
    print("Starting Temperature Regression Model Training...")

    # 1. Load Data
    # ---------------------------------------------------------
    master_path = find_file("Turkey_Weather_Master.csv")
    locs_path = find_file("locations.csv")

    if not master_path or not locs_path:
        print("Error: Data files not found. Please check the 'data/' directory.")
        return

    df = pd.read_csv(master_path)
    locs = pd.read_csv(locs_path)

    # Merge coordinates: Latitude and Longitude are critical for temperature prediction
    # as they dictate climate zones (e.g., coastal vs inland).
    df = df.merge(locs[['plaka', 'lat', 'lon']], left_on='plate_code', right_on='plaka', how='left')
    df['date'] = pd.to_datetime(df['date'])

    print(f"Data Loaded. Initial Shape: {df.shape}")

    # 2. Preprocessing & Feature Engineering
    # ---------------------------------------------------------
    print("Engineering Features...")

    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day

    # Critical Feature: 'day_of_year' (1 to 365)
    # This captures the cyclic nature of seasons better than just 'month'.
    # It helps the model understand that Dec 31 is very close to Jan 1 in terms of temperature.
    df['day_of_year'] = df['date'].dt.dayofyear

    # Data Cleaning: Filter out 2003
    # Analysis showed that 2003 data contains missing summer months/anomalies
    # that negatively impact model accuracy.
    df = df[df['year'] > 2003]
    print(f"Filtered out 2003 data. New Shape: {df.shape}")

    # 3. Data Preparation
    # ---------------------------------------------------------
    features = ['lat', 'lon', 'year', 'month', 'day', 'day_of_year']
    target = 'max_temp'

    X = df[features]
    y = df[target]

    # Time-Based Split
    # We strictly split by time to prevent "data leakage".
    # Predicting the past using the future is cheating; we must predict the future using the past.
    X_train = X[df['year'] < CUTOFF_YEAR]
    y_train = y[df['year'] < CUTOFF_YEAR]
    X_test = X[df['year'] >= CUTOFF_YEAR]
    y_test = y[df['year'] >= CUTOFF_YEAR]

    print(f"Training Set: {len(X_train)} samples | Test Set: {len(X_test)} samples")

    # 4. Model Training
    # ---------------------------------------------------------
    print("Training Random Forest Regressor...")

    # Hyperparameters Optimization ("Diet" Random Forest):
    # - max_depth=12: Limits tree depth to prevent overfitting and keep model size small (<150MB).
    # - min_samples_leaf=20: Ensures leaf nodes represent general trends, not noise.
    # - n_jobs=-1: Utilizes all CPU cores for faster training.
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=12,
        min_samples_leaf=20,
        n_jobs=-1,
        random_state=42
    )

    rf_model.fit(X_train, y_train)

    # 5. Evaluation
    # ---------------------------------------------------------
    y_pred = rf_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"\nModel Performance (MAE): {mae:.2f} °C")
    print(f"(On average, the prediction is within {mae:.2f}°C of the actual temperature)")

    # 6. Save Model
    # ---------------------------------------------------------
    joblib.dump(rf_model, MODEL_PATH)
    print(f"Model saved successfully to '{MODEL_PATH}'")


if __name__ == "__main__":
    main()
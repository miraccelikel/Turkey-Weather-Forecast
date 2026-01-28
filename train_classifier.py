"""
Turkey AI Weather Forecast - Classification Model Training Script
-----------------------------------------------------------------
Author: Miraç Çelikel
Description:
    This script trains a Random Forest Classifier to predict daily weather conditions
    (Sunny, Cloudy, Rain, Snow) based on historical meteorological data.

    It performs the following steps:
    1. Loads and merges weather and location data.
    2. Engineers physics-based features (e.g., Diurnal Temperature Range).
    3. Simplifies complex WMO weather codes into 4 main classes.
    4. Trains a Random Forest model with class balancing.
    5. Evaluates accuracy and saves the model for the Streamlit app.
"""

import pandas as pd
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# --- CONFIGURATION ---
MODEL_PATH = "weather_classifier.pkl"
CUTOFF_YEAR = 2024  # Train on data before this year, test on this year and after


def find_file(filename, folder="data"):
    """
    Locates a file whether the script is run from the root or a subfolder.
    """
    path1 = os.path.join(folder, filename)  # Standard path
    path2 = os.path.join("..", folder, filename)  # One level up

    if os.path.exists(path1):
        return path1
    elif os.path.exists(path2):
        return path2
    else:
        return None


def simplify_weather(code):
    """
    Maps complex WMO weather codes to 4 simplified categories for better classification.

    Args:
        code (int): WMO weather code.
    Returns:
        str: 'Sunny', 'Cloudy', 'Snow', or 'Rain'.
    """
    code = int(code)
    if code in [0, 1, 2]:
        return "Sunny"
    elif code in [3, 45, 48]:
        return "Cloudy"
    elif code in [71, 73, 75, 77, 85, 86, 66, 67]:
        return "Snow"
    else:
        return "Rain"  # Default to rain for other precipitation codes


def main():
    print("Starting Classification Model Training...")

    # 1. Load Data
    # ---------------------------------------------------------
    master_path = find_file("Turkey_Weather_Master.csv")
    locs_path = find_file("locations.csv")

    if not master_path or not locs_path:
        print("Error: Data files not found. Please check the 'data/' directory.")
        return

    df = pd.read_csv(master_path)
    locs = pd.read_csv(locs_path)

    # Merge coordinates (Lat/Lon is crucial for spatial patterns)
    df = df.merge(locs[['plaka', 'lat', 'lon']], left_on='plate_code', right_on='plaka', how='left')
    df['date'] = pd.to_datetime(df['date'])

    print(f"Data Loaded. Shape: {df.shape}")

    # 2. Feature Engineering
    # ---------------------------------------------------------
    print("Engineering Features...")

    # Extract temporal features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day_of_year'] = df['date'].dt.dayofyear

    # Physics-Informed Feature: Diurnal Temperature Range
    # Rationale: Large difference between Max and Min temp usually implies clear skies (Sunny).
    # Small difference usually implies cloud cover (Cloudy/Rain).
    df['temp_range'] = df['max_temp'] - df['min_temp']

    # Target Processing
    df['weather_simple'] = df['weather_code'].apply(simplify_weather)

    # 3. Data Preparation
    # ---------------------------------------------------------
    # We include temperature data as features because they are physically linked to weather conditions.
    features = ['lat', 'lon', 'year', 'month', 'day_of_year', 'max_temp', 'min_temp', 'temp_range']
    target = 'weather_simple'

    # Remove rows with missing values to prevent training errors
    df_clean = df.dropna(subset=features + [target])

    # Time-Based Split
    # We split by year to simulate real-world forecasting (training on past, testing on future).
    X = df_clean[features]
    y = df_clean[target]

    X_train = X[df_clean['year'] < CUTOFF_YEAR]
    y_train = y[df_clean['year'] < CUTOFF_YEAR]
    X_test = X[df_clean['year'] >= CUTOFF_YEAR]
    y_test = y[df_clean['year'] >= CUTOFF_YEAR]

    print(f"Training Set: {len(X_train)} samples | Test Set: {len(X_test)} samples")

    # 4. Model Training
    # ---------------------------------------------------------
    print("Training Random Forest Classifier...")

    # Hyperparameters:
    # - n_jobs=-1: Use all CPU cores for speed.
    # - class_weight='balanced': Crucial for weather data where 'Sunny' is common but 'Snow' is rare.
    #   This forces the model to pay more attention to rare events.
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_leaf=10,
        class_weight='balanced',
        n_jobs=-1,
        random_state=42
    )

    rf_model.fit(X_train, y_train)

    # 5. Evaluation
    # ---------------------------------------------------------
    y_pred = rf_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\nModel Accuracy: {acc:.2%}")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred))

    # 6. Save Model
    # ---------------------------------------------------------
    joblib.dump(rf_model, MODEL_PATH)
    print(f"Model saved successfully to '{MODEL_PATH}'")


if __name__ == "__main__":
    main()
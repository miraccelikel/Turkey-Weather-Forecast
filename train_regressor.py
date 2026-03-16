"""
Turkey Weather Forecast - Multi-Output Regression Training Script
------------------------------------------------------------------
Author: Miraç Çelikel
Description:
    Trains regressors to predict 7 physical features for a given
    location and future date. These predictions feed the Classifier.

    Targets: max_temp, min_temp, precipitation, wind_speed,
             humidity, pressure, radiation

Changes:
    - Added 'region' feature (Turkey's 7 geographic regions)
      to improve humidity MAE (coastal vs inland variance)
"""

import os
import pandas as pd
import joblib
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# --- CONFIGURATION ---
MODEL_PATH = "models/temperature_models.pkl"
CUTOFF_YEAR = 2024

FEATURES = ['lat', 'lon', 'year', 'month', 'day', 'day_of_year', 'region_encoded']
TARGETS  = ['max_temp', 'min_temp', 'precipitation',
            'wind_speed', 'humidity', 'pressure', 'radiation']


def find_file(filename, folder="data"):
    path1 = os.path.join(folder, filename)
    path2 = os.path.join("..", folder, filename)
    if os.path.exists(path1):
        return path1
    elif os.path.exists(path2):
        return path2
    return None


def assign_region(lat, lon):
    """
    Assigns Turkey's 7 standard geographic regions based on coordinates.

    Regions and approximate boundaries:
    - Karadeniz        : lat > 40.5  (Black Sea coast)
    - Marmara          : lat > 39.5, lon < 31  (NW Turkey)
    - Ege              : lon < 29.5, lat < 40.5  (Aegean coast)
    - Akdeniz          : lat < 37.5, lon < 36  (Mediterranean)
    - Ic_Anadolu       : central plateau
    - Dogu_Anadolu     : lon > 38, lat > 37.5
    - Guneydogu_Anadolu: lon > 36, lat < 37.5
    """
    if lat > 40.5:
        return 'Karadeniz'
    elif lat > 39.5 and lon < 31:
        return 'Marmara'
    elif lon < 29.5 and lat < 40.5:
        return 'Ege'
    elif lat < 37.5 and lon < 36:
        return 'Akdeniz'
    elif lon > 38 and lat > 37.5:
        return 'Dogu_Anadolu'
    elif lon > 36 and lat < 37.5:
        return 'Guneydogu_Anadolu'
    else:
        return 'Ic_Anadolu'


def main():
    print("=" * 60)
    print("Turkey Weather Forecast — Multi-Output Regression Training")
    print("=" * 60)
    print(f"Targets ({len(TARGETS)}): {TARGETS}\n")

    # ------------------------------------------------------------------
    # 1. Load Data
    # ------------------------------------------------------------------
    master_path = find_file("Turkey_Weather_Master.csv")
    locs_path   = find_file("locations.csv")

    df   = pd.read_csv(master_path)
    locs = pd.read_csv(locs_path)

    df = df.merge(locs[['plaka', 'lat', 'lon']],
                  left_on='plate_code', right_on='plaka', how='left')
    df['date'] = pd.to_datetime(df['date'])

    # ------------------------------------------------------------------
    # 2. Feature Engineering
    # ------------------------------------------------------------------
    df['year']        = df['date'].dt.year
    df['month']       = df['date'].dt.month
    df['day']         = df['date'].dt.day
    df['day_of_year'] = df['date'].dt.dayofyear

    # Turkey's 7 geographic regions
    df['region'] = df.apply(
        lambda r: assign_region(r['lat'], r['lon']), axis=1
    )

    le_region = LabelEncoder()
    df['region_encoded'] = le_region.fit_transform(df['region'])

    print("Region distribution:")
    print(df['region'].value_counts().to_string())
    print()

    df = df[df['year'] > 2003]
    df = df.dropna(subset=FEATURES + TARGETS)

    # ------------------------------------------------------------------
    # 3. Train / Test Split  (temporal)
    # ------------------------------------------------------------------
    train_mask = df['year'] < CUTOFF_YEAR

    X_train = df.loc[train_mask,  FEATURES]
    y_train = df.loc[train_mask,  TARGETS]
    X_test  = df.loc[~train_mask, FEATURES]
    y_test  = df.loc[~train_mask, TARGETS]

    print(f"Training set : {len(X_train):,} samples")
    print(f"Test set     : {len(X_test):,} samples\n")

    # ------------------------------------------------------------------
    # 4. Model Definitions
    # ------------------------------------------------------------------
    models = {
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=20,
            n_jobs=-1,
            random_state=42
        ),
        "Gradient Boosting": MultiOutputRegressor(
            GradientBoostingRegressor(
                n_estimators=150,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42
            )
        ),
        "XGBoost": MultiOutputRegressor(
            XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                n_jobs=-1,
                random_state=42,
                verbosity=0
            )
        ),
        "LightGBM": MultiOutputRegressor(
            LGBMRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                n_jobs=-1,
                random_state=42,
                verbose=-1
            )
        ),
        "CatBoost": MultiOutputRegressor(
            CatBoostRegressor(
                iterations=200,
                depth=6,
                learning_rate=0.05,
                verbose=0,
                random_state=42
            )
        )
    }

    # ------------------------------------------------------------------
    # 5. Training Loop
    # ------------------------------------------------------------------
    trained_models = {}
    metrics        = {}
    per_target_mae = {}

    for name, model in models.items():
        print(f"{'─' * 40}")
        print(f"Training  →  {name}")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        y_pred_df    = pd.DataFrame(y_pred, columns=TARGETS)
        y_test_reset = y_test.reset_index(drop=True)

        target_maes = {}
        print(f"\n  {'Target':<15} {'MAE':>8}")
        print(f"  {'─' * 25}")
        for col in TARGETS:
            col_mae = mean_absolute_error(y_test_reset[col], y_pred_df[col])
            target_maes[col] = round(col_mae, 3)
            print(f"  {col:<15} {col_mae:>8.3f}")

        avg_mae = mean_absolute_error(y_test_reset, y_pred_df)
        print(f"  {'─' * 25}")
        print(f"  {'Average':<15} {avg_mae:>8.3f}\n")

        trained_models[name]  = model
        metrics[name]         = round(avg_mae, 2)
        per_target_mae[name]  = target_maes

    # ------------------------------------------------------------------
    # 6. Save — region encoder included for app.py inference
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    package = {
        'models':         trained_models,
        'metrics':        metrics,
        'per_target_mae': per_target_mae,
        'features':       FEATURES,
        'targets':        TARGETS,
        'region_encoder': le_region,
    }
    joblib.dump(package, MODEL_PATH)
    print("=" * 60)
    print(f"All regression models saved → '{MODEL_PATH}'")


if __name__ == "__main__":
    main()
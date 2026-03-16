"""
Turkey Weather Forecast - Multi-Model Classification Training Script
-----------------------------------------------------------------
Author: Miraç Çelikel
Description:
    Trains classifiers to predict daily weather conditions using
    physics-based features (Humidity, Pressure, Radiation).

Fixes applied:
    - Removed 'precipitation' from features (data leakage)
    - Explicit WMO code mapping (no blind else → Rain)
    - Consistent class balancing across all models
    - sample_weight applied to GBM and XGBoost
    - Classification report added for per-class diagnostics
"""

import os
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# --- CONFIGURATION ---
MODEL_PATH = "models/weather_classifiers.pkl"
CUTOFF_YEAR = 2024


def find_file(filename, folder="data"):
    path1 = os.path.join(folder, filename)
    path2 = os.path.join("..", folder, filename)
    if os.path.exists(path1):
        return path1
    elif os.path.exists(path2):
        return path2
    return None


def simplify_weather(code):
    """
    Maps WMO weather codes to 5 simplified categories.
    Dataset codes: [0, 1, 2, 3, 51, 53, 55, 61, 63, 65, 71, 73, 75]
    All codes are explicitly handled — no blind fallback.
    """
    code = int(code)

    if code in [0, 1,2]:
        return "Sunny"

    elif code in [3, 45, 48]:
        return "Cloudy"

    elif code in [51, 53, 55,   # Drizzle: slight / moderate / dense
                  61, 63, 65,   # Rain: slight / moderate / heavy
                  80, 81, 82,   # Rain showers
                  95, 96, 99]:  # Thunderstorm (not in dataset but safe to keep)
        return "Rain"

    elif code in [71, 73, 75,   # Snowfall: slight / moderate / heavy
                  77,           # Snow grains
                  85, 86,       # Snow showers
                  66, 67]:      # Freezing rain
        return "Snow"

    else:
        # Log unexpected codes instead of silently mislabeling
        return "Unknown"


def main():
    print("=" * 60)
    print("Turkey Weather Forecast — Classification Training")
    print("=" * 60)

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
    df['year']       = df['date'].dt.year
    df['month']      = df['date'].dt.month
    df['day_of_year']= df['date'].dt.dayofyear
    df['temp_range'] = df['max_temp'] - df['min_temp']

    df['weather_simple'] = df['weather_code'].apply(simplify_weather)

    # Sanity check — unexpected codes
    unknown_mask = df['weather_simple'] == 'Unknown'
    if unknown_mask.any():
        print(f"\n⚠️  WARNING: {unknown_mask.sum()} rows with unmapped weather_code!")
        print(df.loc[unknown_mask, 'weather_code'].value_counts())
        df = df[~unknown_mask].copy()

    # ------------------------------------------------------------------
    # 'precipitation' is REMOVED — it causes data leakage.
    # The model would learn "rain fell today → it's rainy" which is
    # trivially true but useless for forecasting future days.
    # ------------------------------------------------------------------
    features = [
        'lat', 'lon',
        'year', 'month', 'day_of_year',
        'max_temp', 'min_temp', 'temp_range',
        'wind_speed', 'humidity', 'pressure', 'radiation'
    ]
    target = 'weather_simple'

    df_clean = df.dropna(subset=features + [target]).copy()

    # ------------------------------------------------------------------
    # 3. Encode Labels
    # ------------------------------------------------------------------
    le = LabelEncoder()
    df_clean['weather_encoded'] = le.fit_transform(df_clean[target])
    target_encoded = 'weather_encoded'

    print(f"\nClasses: {list(le.classes_)}")
    print("\nClass distribution (train split):")
    train_mask = df_clean['year'] < CUTOFF_YEAR
    print(df_clean.loc[train_mask, target].value_counts())

    # ------------------------------------------------------------------
    # 4. Train / Test Split  (temporal — no shuffle)
    # ------------------------------------------------------------------
    X_train = df_clean.loc[train_mask, features]
    y_train = df_clean.loc[train_mask, target_encoded]
    X_test  = df_clean.loc[~train_mask, features]
    y_test  = df_clean.loc[~train_mask, target_encoded]

    print(f"\nTraining set : {len(X_train):,} samples")
    print(f"Test set     : {len(X_test):,} samples")

    # Sample weights for models that don't support class_weight natively
    sample_weights = compute_sample_weight('balanced', y_train)

    # ------------------------------------------------------------------
    # 5. Model Definitions
    # All models use balanced class weighting for fair multi-class learning.
    # ------------------------------------------------------------------
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_leaf=10,
            class_weight='balanced',   # built-in support
            n_jobs=-1,
            random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42
            # ← uses sample_weight at fit time (see loop below)
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=42,
            eval_metric='mlogloss',
            verbosity=0
            # ← uses sample_weight at fit time (see loop below)
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            n_jobs=-1,
            random_state=42,
            verbose=-1,
            class_weight='balanced'    # built-in support
        ),
        "CatBoost": CatBoostClassifier(
            iterations=200,
            depth=6,
            learning_rate=0.05,
            verbose=0,
            random_state=42,
            auto_class_weights='Balanced'  # built-in support
        )
    }

    # Models that need explicit sample_weight (no native class_weight param)
    NEEDS_SAMPLE_WEIGHT = {"Gradient Boosting", "XGBoost"}

    # ------------------------------------------------------------------
    # 6. Training Loop
    # ------------------------------------------------------------------
    trained_models = {}
    metrics = {}

    for name, model in models.items():
        print(f"\n{'─' * 40}")
        print(f"Training  →  {name}")

        if name in NEEDS_SAMPLE_WEIGHT:
            model.fit(X_train, y_train, sample_weight=sample_weights)
        else:
            model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc    = accuracy_score(y_test, y_pred)

        trained_models[name] = model
        metrics[name]        = round(acc * 100, 2)

        print(f"Accuracy: {acc:.2%}")
        print(classification_report(
            y_test, y_pred,
            target_names=le.classes_,
            zero_division=0
        ))

    # ------------------------------------------------------------------
    # 7. Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("MODEL SUMMARY")
    print("=" * 60)
    for name, acc in sorted(metrics.items(), key=lambda x: -x[1]):
        print(f"  {name:<25} {acc:.2f}%")

    # ------------------------------------------------------------------
    # 8. Save
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    package = {
        'models':        trained_models,
        'metrics':       metrics,
        'label_encoder': le,
        'features':      features
    }
    joblib.dump(package, MODEL_PATH)
    print(f"\nAll models saved → '{MODEL_PATH}'")


if __name__ == "__main__":
    main()
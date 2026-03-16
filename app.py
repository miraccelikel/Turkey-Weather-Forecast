"""
Turkey Weather Forecast - Streamlit App
----------------------------------------
Author: Miraç Çelikel
Description:
    Two-stage prediction pipeline:
      1. Regressor  → predicts 7 physical features (temp, humidity, etc.)
      2. Classifier → predicts weather condition from those features

    'precipitation' is excluded from Classifier input (data leakage).
    Region feature is encoded at inference using the saved LabelEncoder.
"""

import streamlit as st
import pandas as pd
import joblib
import os
import datetime

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Turkey Weather Forecast",
    page_icon="🌤️",
    layout="wide"
)

# --- 2. UI CUSTOMIZATION ---
st.markdown("""
<style>
    .block-container { padding-top: 1rem !important; padding-bottom: 0rem !important; }
    div.stMarkdown { margin-bottom: -10px !important; }
    hr { margin-top: 0px !important; margin-bottom: 15px !important; }
    [data-testid="stMap"] { height: 350px !important; border-radius: 12px; }
    .main-title { text-align: center; color: #ff4b4b; margin-bottom: 0px; padding-bottom: 0px; }
    .sub-title  { text-align: center; color: #555; font-size: 0.95rem; margin-top: -5px; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

# --- 3. CONSTANTS ---
REGRESSOR_FILE  = "models/temperature_models.pkl"
CLASSIFIER_FILE = "models/weather_classifiers.pkl"
LOCATION_FILE   = "data/locations.csv"

CLF_FEATURES_FALLBACK = [
    'lat', 'lon', 'year', 'month', 'day_of_year',
    'max_temp', 'min_temp', 'temp_range',
    'wind_speed', 'humidity', 'pressure', 'radiation'
]
REG_FEATURES_FALLBACK = ['lat', 'lon', 'year', 'month', 'day', 'day_of_year', 'region_encoded']
REG_TARGETS_FALLBACK  = ['max_temp', 'min_temp', 'precipitation',
                          'wind_speed', 'humidity', 'pressure', 'radiation']


# --- 4. REGION HELPER (mirrors training_regressor.py) ---
def assign_region(lat, lon):
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


# --- 5. MODEL LOADER ---
@st.cache_resource
def load_system():
    if not os.path.exists(REGRESSOR_FILE) or not os.path.exists(CLASSIFIER_FILE):
        return None, None, None, "❌ Error: ML Models not found in 'models/' directory. Please run training scripts."
    try:
        reg_package = joblib.load(REGRESSOR_FILE)
        clf_package = joblib.load(CLASSIFIER_FILE)
        locs = pd.read_csv(LOCATION_FILE).sort_values('city_name')
        return reg_package, clf_package, locs, None
    except Exception as e:
        return None, None, None, f"❌ System Error: {e}"


reg_data, clf_data, locations, error_msg = load_system()

if error_msg:
    st.error(error_msg)
    st.stop()

reg_features  = reg_data.get('features', REG_FEATURES_FALLBACK)
reg_targets   = reg_data.get('targets',  REG_TARGETS_FALLBACK)
clf_features  = clf_data.get('features', CLF_FEATURES_FALLBACK)
le_region     = reg_data.get('region_encoder', None)


# --- 6. HELPERS ---
def get_icon(condition):
    return {
        "Sunny":  "☀️ Sunny",
        "Cloudy": "☁️ Cloudy",
        "Rain":   "🌧️ Rainy",
        "Snow":   "❄️ Snowy"
    }.get(condition, condition)


def get_rain_detail(pred_precip):
    """Breaks Rain into intensity levels using regressor's precipitation output."""
    if pred_precip < 2.5:
        return "🌦️ Light Rain"
    elif pred_precip < 10:
        return "🌧️ Moderate Rain"
    else:
        return "⛈️ Heavy Rain"


def check_special_date(month, day):
    """Displays a custom styled birthday toast without an icon and with a specific color."""
    if month == 8 and day == 22:
        st.markdown(
            f"""
            <div style="
                position: fixed;
                top: 80px;
                right: 20px;
                background-color: palevioletred;
                color: white;
                padding: 15px 25px;
                border-radius: 12px;
                font-size: 16px;
                font-weight: bold;
                z-index: 9999;
                box-shadow: 0px 4px 12px rgba(0,0,0,0.2);
                border: none;
                animation: fade-in-out 5s ease-in-out forwards;
            ">
                🎂 The most beautiful day in the world 🌍
            </div>
            <style>
                @keyframes fade-in-out {{
                    0% {{ opacity: 0; transform: translateY(-20px); }}
                    10% {{ opacity: 1; transform: translateY(0); }}
                    90% {{ opacity: 1; }}
                    100% {{ opacity: 0; display: none; }}
                }}
            </style>
            """,
            unsafe_allow_html=True
        )


# --- 7. HEADER ---
st.markdown("<h1 class='main-title'>🌤️ Turkey Weather Forecast</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Advanced Climate Prediction powered by Ensemble Boosting AI.</p>",
            unsafe_allow_html=True)
st.markdown("---")

# --- 8. MAIN INTERFACE ---
col1, col2 = st.columns([1, 1])

map_data = locations[['lat', 'lon']]
map_zoom = 5

with col1:
    st.subheader("⚙️ Configuration")

    city = st.selectbox("📍 Select City:", locations['city_name'].unique())

    months = {i: datetime.date(2000, i, 1).strftime('%B') for i in range(1, 13)}
    col1_a, col1_b = st.columns(2)
    with col1_a:
        month_name = st.selectbox("📅 Select Month:", list(months.values()))
    with col1_b:
        day = st.number_input("📅 Select Day:", min_value=1, max_value=31, value=15)

    month_idx = list(months.keys())[list(months.values()).index(month_name)]

    available_models    = list(reg_data['models'].keys())
    selected_model_name = st.selectbox("🧠 Select ML Model:", available_models, index=0)

    if st.button("Generate Forecast 🚀", type="primary", use_container_width=True):

        loc_data = locations[locations['city_name'] == city].iloc[0]
        lat, lon = loc_data['lat'], loc_data['lon']
        map_data = pd.DataFrame({'lat': [lat], 'lon': [lon]})
        map_zoom = 8

        try:
            day_of_year = datetime.date(2026, month_idx, day).timetuple().tm_yday

            reg_model     = reg_data['models'][selected_model_name]
            clf_model     = clf_data['models'][selected_model_name]
            label_encoder = clf_data['label_encoder']

            # ----------------------------------------------------------
            # Stage 1 — Regressor: predict 7 physical features
            # ----------------------------------------------------------
            region_name = assign_region(lat, lon)
            region_enc  = le_region.transform([region_name])[0] if le_region else 0

            input_reg = pd.DataFrame(
                [[lat, lon, 2026, month_idx, day, day_of_year, region_enc]],
                columns=reg_features
            )
            pred_values = reg_model.predict(input_reg)[0]

            pred_map       = dict(zip(reg_targets, pred_values))
            pred_max_temp  = pred_map['max_temp']
            pred_min_temp  = pred_map['min_temp']
            pred_precip    = pred_map['precipitation']
            pred_wind      = pred_map['wind_speed']
            pred_humidity  = pred_map['humidity']
            pred_pressure  = pred_map['pressure']
            pred_radiation = pred_map['radiation']
            pred_temp_range = pred_max_temp - pred_min_temp

            # ----------------------------------------------------------
            # Stage 2 — Classifier: predict weather condition
            # 'precipitation' intentionally excluded (data leakage)
            # ----------------------------------------------------------
            input_cls = pd.DataFrame(
                [[lat, lon, 2026, month_idx, day_of_year,
                  pred_max_temp, pred_min_temp, pred_temp_range,
                  pred_wind, pred_humidity, pred_pressure, pred_radiation]],
                columns=clf_features
            )

            pred_cond_num = clf_model.predict(input_cls)[0]
            pred_cond     = label_encoder.inverse_transform([pred_cond_num])[0]

            # Rain intensity via regressor's precipitation output
            display_cond = get_rain_detail(pred_precip) if pred_cond == "Rain" else get_icon(pred_cond)

            # ----------------------------------------------------------
            # Results
            # ----------------------------------------------------------
            st.success(f"✅ Forecast for **{city}** · {region_name} · **{selected_model_name}**")

            res_c1, res_c2, res_c3 = st.columns(3)
            res_c1.metric("🌡️ Max Temp",  f"{pred_max_temp:.1f} °C")
            res_c2.metric("🌡️ Min Temp",  f"{pred_min_temp:.1f} °C")
            res_c3.metric("☁️ Condition", display_cond)

            with st.expander("🔍 View AI Physics Predictions (Under the Hood)"):
                ph1, ph2, ph3, ph4, ph5 = st.columns(5)
                ph1.metric("💧 Humidity",   f"{pred_humidity:.1f} %")
                ph2.metric("🔵 Pressure",   f"{pred_pressure:.1f} hPa")
                ph3.metric("💨 Wind",       f"{pred_wind:.1f} km/h")
                ph4.metric("☀️ Radiation",  f"{pred_radiation:.2f} MJ/m²")
                ph5.metric("🌧️ Precip.",   f"{pred_precip:.1f} mm")

            check_special_date(month_idx, day)

        except ValueError as ve:
            st.error(f"❌ Invalid date: {ve}")
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")

with col2:
    st.markdown("### 🗺️ Location Map")
    st.map(map_data, zoom=map_zoom)

# --- 9. MODEL PERFORMANCE ---
st.markdown("---")
st.subheader("📊 ML Models Performance Comparison")
st.markdown("Evaluation on held-out test data (2024 and onwards).")

perf_df = pd.DataFrame({
    "AI Algorithm":       list(reg_data['metrics'].keys()),
    "Regression Avg MAE": list(reg_data['metrics'].values()),
    "Weather Accuracy %": list(clf_data['metrics'].values())
})
st.dataframe(perf_df, use_container_width=True, hide_index=True)

if 'per_target_mae' in reg_data:
    with st.expander("🔬 Per-Target MAE Breakdown"):
        per_target_df = pd.DataFrame(reg_data['per_target_mae']).T
        per_target_df.index.name = "Model"
        st.dataframe(per_target_df, use_container_width=True)
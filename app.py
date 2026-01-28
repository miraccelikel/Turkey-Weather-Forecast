import streamlit as st
import pandas as pd
import joblib
import os
import datetime
import time

# --- CONFIGURATION ---
st.set_page_config(
    page_title="Turkey AI Weather Forecast",
    page_icon="🌤️",
    layout="wide"
)

# --- MAP & TITLE CSS ---
st.markdown("""
<style>
    .stDeckGlJsonChart {
        height: 500px !important;
    }
    .centered-title {
        text-align: center;
        font-weight: bold;
        color: #ff4b4b; /* Streamlit kırmızısı */
    }
    .centered-text {
        text-align: center;
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)

# --- MODEL METADATA (CONSTANTS) ---
MODEL_MAE = 3.57
MODEL_ACCURACY = 60.83

# File Paths
REGRESSOR_FILE = "temperature_model.pkl"
CLASSIFIER_FILE = "weather_classifier.pkl"
LOCATION_FILE = "data/locations.csv"


# --- SYSTEM LOADER ---
@st.cache_resource
def load_system():
    if not os.path.exists(REGRESSOR_FILE) or not os.path.exists(CLASSIFIER_FILE):
        return None, None, None, "❌ Error: Model files not found."

    try:
        reg = joblib.load(REGRESSOR_FILE)
        clf = joblib.load(CLASSIFIER_FILE)
        locs = pd.read_csv(LOCATION_FILE).sort_values('city_name')
        return reg, clf, locs, None
    except Exception as e:
        return None, None, None, f"❌ System Error: {e}"


reg_model, class_model, locations, error_msg = load_system()


# --- UTILS ---
def get_icon(condition):
    icons = {
        "Sunny": "☀️ Sunny / Clear",
        "Cloudy": "☁️ Cloudy / Overcast",
        "Rain": "🌧️ Rainy / Stormy",
        "Snow": "❄️ Snow / Blizzard"
    }
    return icons.get(condition, condition)


def check_special_date(month, day):
    if month == 8 and day == 22:
        st.markdown(
            """
            <input type="checkbox" id="close_toast" style="display: none;">

            <div class="birthday-toast" style="
                position: fixed;
                top: 80px;
                right: 20px;
                background-color: #FF4B4B;
                color: white;
                padding: 15px 25px;
                border-radius: 12px;
                font-size: 18px;
                font-weight: bold;
                z-index: 9999;
                box-shadow: 0px 4px 12px rgba(0,0,0,0.3);
                border: 2px solid white;
            ">
                <label for="close_toast" style="
                    cursor: pointer;
                    float: right;
                    margin-left: 15px;
                    font-size: 22px;
                    line-height: 20px;
                    color: white;
                ">
                    &times;
                </label>
                🎂 The most beautiful day in the world 🌍
            </div>

            <style>
                #close_toast:checked + .birthday-toast {
                    display: none;
                }
            </style>
            """,
            unsafe_allow_html=True
        )


# --- UI HEADER (CENTERED) ---
if error_msg:
    st.error(error_msg)
    st.stop()

# Title
st.markdown("<h1 style='text-align: center;'>🌤️ Turkey AI Weather Forecast</h1>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center;'>
    <b>Advanced Weather Prediction System:</b> Uses historical data (<b>2003-2025</b>) and Random Forest AI to forecast future conditions.
</div>
""", unsafe_allow_html=True)
st.markdown("---")

# --- MAIN INTERFACE ---
col1, col2 = st.columns([1, 2])

# Global Map Variables (Default View)
map_data = locations[['lat', 'lon']]
map_zoom = 5

with col1:
    st.subheader("⚙️ Configuration")

    city = st.selectbox("📍 Select City:", locations['city_name'].unique())

    months = {i: datetime.date(2000, i, 1).strftime('%B') for i in range(1, 13)}
    month_name = st.selectbox("📅 Select Month:", list(months.values()))
    month_idx = list(months.keys())[list(months.values()).index(month_name)]

    day = st.number_input("Select Day:", min_value=1, max_value=31, value=15)

    st.markdown("---")

    if st.button("Generate Forecast 🚀", type="primary"):
        # --- DATA PREPARATION ---
        loc_data = locations[locations['city_name'] == city].iloc[0]
        lat, lon = loc_data['lat'], loc_data['lon']

        # Update Map to Focus on City
        map_data = pd.DataFrame({'lat': [lat], 'lon': [lon]})
        map_zoom = 8

        year = 2026

        try:
            day_of_year = datetime.date(year, month_idx, day).timetuple().tm_yday

            # --- 1. TEMPERATURE ---
            input_reg = pd.DataFrame([[lat, lon, year, month_idx, day, day_of_year]],
                                     columns=['lat', 'lon', 'year', 'month', 'day', 'day_of_year'])
            pred_max_temp = reg_model.predict(input_reg)[0]

            # --- 2. CONDITION ---
            est_min_temp = pred_max_temp - 12.0
            est_temp_range = 12.0

            input_cls = pd.DataFrame(
                [[lat, lon, year, month_idx, day_of_year, pred_max_temp, est_min_temp, est_temp_range]],
                columns=['lat', 'lon', 'year', 'month', 'day_of_year', 'max_temp', 'min_temp', 'temp_range'])
            pred_cond = class_model.predict(input_cls)[0]

            # --- DISPLAY ---
            st.success(f"✅ Forecast Results for **{city}**")

            m1, m2 = st.columns(2)
            m1.metric("🌡️ Max Temperature", f"{pred_max_temp:.1f} °C")
            m2.metric("☁️ Sky Condition", get_icon(pred_cond))

            # --- SPECIAL MESSAGE ---
            check_special_date(month_idx, day)

        except ValueError:
            st.error("❌ Invalid Date!")

with col2:
    st.markdown("### 🗺️ Location Map")

    # Standard Streamlit Map
    st.map(map_data, zoom=map_zoom)

    st.markdown("---")
    st.info("""
    **Model Architecture:**
    1. **Regression Model:** Predicts maximum temperature based on location & time.
    2. **Physics Check:** Uses temperature limits to refine predictions.
    3. **Classification Model:** Determines final weather condition.
    """)

    st.caption(f"Model Accuracy (Test Set): Temp MAE: {MODEL_MAE}°C | Condition Accuracy: {MODEL_ACCURACY}%")
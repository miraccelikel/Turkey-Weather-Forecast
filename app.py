import streamlit as st
import pandas as pd
import joblib
import os
import datetime

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Turkey AI Weather Forecast",
    page_icon="🌤️",
    layout="wide"
)

# --- 2. UI CUSTOMIZATION (CSS) ---
st.markdown("""
<style>
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 0rem !important;
    }
    div.stMarkdown {
        margin-bottom: -10px !important;
    }
    hr {
        margin-top: 0px !important;
        margin-bottom: 15px !important;
    }

    [data-testid="stMap"] {
        height: 350px !important;
        border-radius: 12px;
    }
    .stDeckGlJsonChart {
        height: 350px !important;
    }

    .main-title {
        text-align: center;
        color: #ff4b4b;
        margin-bottom: 0px;
        padding-bottom: 0px;
    }
    .sub-title {
        text-align: center;
        color: #555;
        font-size: 0.95rem;
        margin-top: -5px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. CONSTANTS & LOADER ---
MODEL_MAE = 3.57
MODEL_ACCURACY = 60.83
REGRESSOR_FILE = "temperature_model.pkl"
CLASSIFIER_FILE = "weather_classifier.pkl"
LOCATION_FILE = "data/locations.csv"


@st.cache_resource
def load_system():
    if not os.path.exists(REGRESSOR_FILE) or not os.path.exists(CLASSIFIER_FILE):
        return None, None, None, "❌ Error: AI Models not found."
    try:
        reg = joblib.load(REGRESSOR_FILE)
        clf = joblib.load(CLASSIFIER_FILE)
        locs = pd.read_csv(LOCATION_FILE).sort_values('city_name')
        return reg, clf, locs, None
    except Exception as e:
        return None, None, None, f"❌ System Error: {e}"


reg_model, class_model, locations, error_msg = load_system()


# --- 4. UTILS & SPECIAL FUNCTIONS ---
def get_icon(condition):
    icons = {"Sunny": "☀️ Sunny", "Cloudy": "☁️ Cloudy", "Rain": "🌧️ Rainy", "Snow": "❄️ Snowy"}
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
                <label for="close_toast" style="cursor: pointer; float: right; margin-left: 15px; font-size: 22px; line-height: 20px;">
                    &times;
                </label>
                🎂 The most beautiful day in the world 🌍
            </div>
            <style>
                #close_toast:checked + .birthday-toast { display: none; }
            </style>
            """,
            unsafe_allow_html=True
        )


# --- 5. HEADER (CENTERED & COMPACT) ---
st.markdown("<h1 class='main-title'>🌤️ Turkey AI Weather Forecast</h1>", unsafe_allow_html=True)
st.markdown(
    "<p class='sub-title'>Advanced Climate Prediction powered by Random Forest AI, leveraging 20+ years of historical meteorological data (2003-2025).</p>",
    unsafe_allow_html=True)
st.markdown("---")

# --- 6. MAIN INTERFACE ---
col1, col2 = st.columns([1, 1])

map_data = locations[['lat', 'lon']]
map_zoom = 5

with col1:
    st.subheader("⚙️ Configuration")

    city = st.selectbox("📍 Select City:", locations['city_name'].unique())

    months = {i: datetime.date(2000, i, 1).strftime('%B') for i in range(1, 13)}
    month_name = st.selectbox("📅 Select Month:", list(months.values()))
    month_idx = list(months.keys())[list(months.values()).index(month_name)]

    day = st.number_input("Select Day:", min_value=1, max_value=31, value=15)

    if st.button("Generate Forecast 🚀", type="primary", use_container_width=True):
        loc_data = locations[locations['city_name'] == city].iloc[0]
        lat, lon = loc_data['lat'], loc_data['lon']
        map_data = pd.DataFrame({'lat': [lat], 'lon': [lon]})
        map_zoom = 8

        try:
            day_of_year = datetime.date(2026, month_idx, day).timetuple().tm_yday
            # Predictions
            input_reg = pd.DataFrame([[lat, lon, 2026, month_idx, day, day_of_year]],
                                     columns=['lat', 'lon', 'year', 'month', 'day', 'day_of_year'])
            pred_max_temp = reg_model.predict(input_reg)[0]
            input_cls = pd.DataFrame([[lat, lon, 2026, month_idx, day_of_year, pred_max_temp, pred_max_temp - 12, 12]],
                                     columns=['lat', 'lon', 'year', 'month', 'day_of_year', 'max_temp', 'min_temp',
                                              'temp_range'])
            pred_cond = class_model.predict(input_cls)[0]

            # Results
            st.success(f"✅ Forecast for **{city}**")
            res_c1, res_c2 = st.columns(2)
            res_c1.metric("🌡️ Max Temp", f"{pred_max_temp:.1f} °C")
            res_c2.metric("☁️ Condition", get_icon(pred_cond))

            check_special_date(month_idx, day)
        except Exception:
            st.error("❌ Invalid Date!")

    st.markdown("---")
    st.info(f"""
    **AI Model Architecture & Performance:**
    * **Regressor:** Temporal & Geospatial Temp Prediction.
    * **Classifier:** Sky Condition Mapping (RF).
    * **Accuracy:** {MODEL_ACCURACY}% | **MAE:** {MODEL_MAE}°C
    """)

with col2:
    st.markdown("### 🗺️ Location Map")
    st.map(map_data, zoom=map_zoom)

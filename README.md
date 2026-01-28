# 🌤️ Turkey AI Weather Forecast

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Scikit-Learn](https://img.shields.io/badge/ML-RandomForest-green)
![Status](https://img.shields.io/badge/Status-Completed-success)

**Advanced Weather Prediction System**

A machine learning project that forecasts daily weather conditions and maximum temperatures for all **81 cities in Turkey**, leveraging historical data from **2003 to 2025**.

</div>

---

## 📖 Project Overview

This project is a comprehensive **Data Science & Machine Learning pipeline**.  
It processes over **20 years of meteorological data** to train optimized **Random Forest** models.  
The system analyzes historical patterns to predict future climate conditions with high accuracy.

---


## 📡 Data Collection Architecture (ETL Pipeline)

To ensure data integrity and handle API rate limits effectively, a **modular batch processing system** was designed.

* **Batch Scraping (Extraction):** Data collection was split into 3 independent scripts (`data_collection/batch_xx.py`) using **Manual Sharding**. This prevents data loss during network interruptions and respects API limits.
* **Data Merging (Transformation):** A custom `merge_data.py` script aggregates the 81 raw city files, performs type casting, sorts temporally, and compiles the final `Turkey_Weather_Master.csv`.
* **Source:** Historical weather data (2003-2025) was retrieved via the [Open-Meteo API](https://open-meteo.com/).
---

## 📊 Data Insights & Analysis (EDA)

Before training the models, an in-depth exploratory data analysis was conducted on weather data from **2003 to 2025**.

### 1️⃣ Warming Trend in Turkey

The analysis reveals clear temperature trends and anomalies observed over the last two decades.

![Temperature Trend](assets/temp_trend_2003_2025.png)

---

### 2️⃣ Feature Correlations

Correlation analysis was performed to understand the relationships between meteorological variables.

![Correlation Matrix](assets/correlation_matrix.png)

---

## 🧠 Model Performance

The system uses a **Hybrid AI Approach** with two different models.  
Models were trained on **2003–2023** data and tested on **2024–2025**.

---

### 🌡️ Model 1: Temperature Prediction (Regression)

- **Algorithm:** Random Forest Regressor (Optimized)
- **Metric:** MAE (Mean Absolute Error)
- **Score:** `3.57°C`

The model successfully captures seasonal temperature patterns and closely follows real observations.

![Regression Results](assets/regression_actual_vs_pred.png)

---

### ☁️ Model 2: Weather Condition Prediction (Classification)

- **Algorithm:** Random Forest Classifier (Class Weighted)
- **Classes:** Sunny, Cloudy, Rain, Snow
- **Accuracy:** `~61%`

The model is especially sensitive to **snow events**, improving winter condition detection.

![Confusion Matrix](assets/confusion_matrix.png)

---

## 🚀 Features

- 🌍 **Nationwide Coverage:** Supports all 81 provinces of Turkey
- 📍 **Geographical Awareness:** City-level coordinate mapping
- ⚡ **Optimized Models:** Robust Random Forest implementations
- 🖥️ **Interactive Dashboard:** Built with Streamlit
- 📅 **Date-Based Forecasting:** Dynamic prediction interface

---

## 📂 Project Structure

```text
```text
Turkey-Weather-Forecast/
│
├── assets/                      # Generated Analysis Plots
│   ├── temp_trend_2003_2025.png
│   └── ...
│
├── data/                        # Data Storage
│   ├── Turkey_Weather_Master.csv  # Final merged dataset (Ready for ML)
│   ├── locations.csv              # City coordinates
│   └── city_weather_data/         # Raw CSV shards (81 files)
│
├── data_collection/             # ETL Pipeline & Scraping Scripts
│   ├── batch_01.py              # Cities 0-27
│   ├── batch_02.py              # Cities 27-54
│   ├── batch_03.py              # Cities 54-81
│   └── merge_data.py            # Aggregation Script
│
├── notebooks/               
│   └── Turkey_Weather_Analysis.ipynb
│
├── app.py                       # Main Streamlit Dashboard
├── train_regressor.py           # ML Training Script (Temp)
├── train_classifier.py          # ML Training Script (Condition)
├── requirements.txt             # Dependencies
└── README.md                    # Documentation
```
## 🛠️ Installation & Usage

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/miraccelikel/Turkey-Weather-Forecast.git
cd Turkey-Weather-Forecast
```
### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```
### 3️⃣ Data Collection (Optional)
*If you want to re-fetch the dataset from scratch, run the batch scripts sequentially:*

```bash
cd data_collection
python batch_01.py  # Fetches cities 0-27
python batch_02.py  # Fetches cities 27-54
python batch_03.py  # Fetches cities 54-81
python merge_data.py # Merges all shards into Master CSV
cd ..
```
### 3️⃣ Train the Models

Run the following scripts to train the machine learning models locally:

```bash
python train_regressor.py
python train_classifier.py
```
### 4️⃣ Run the Application

```bash
streamlit run app.py
```
##  Author

**Miraç Çelikel**  
Software Engineering Student  
Adana Alparslan Türkeş Science and Technology University (ATU)  



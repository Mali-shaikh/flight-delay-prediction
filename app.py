"""
Streamlit Application
Flight Delay Prediction Interactive UI
"""

import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path

# Need to add src to path if running from root
import sys
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.predict import load_model, build_feature_vector, predict

st.set_page_config(
    page_title="Flight Delay Predictor ✈️",
    page_icon="✈️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Constants & Mappings ---
CARRIERS = {
    "AA - American Airlines": "AA",
    "AS - Alaska Airlines": "AS",
    "B6 - JetBlue": "B6",
    "DL - Delta Air Lines": "DL",
    "F9 - Frontier Airlines": "F9",
    "HA - Hawaiian Airlines": "HA",
    "NK - Spirit Airlines": "NK",
    "UA - United Airlines": "UA",
    "WN - Southwest Airlines": "WN",
}

AIRPORTS = {
    "ATL - Atlanta Hartsfield": "ATL", "BOS - Boston Logan": "BOS",
    "CLT - Charlotte Douglas": "CLT", "DEN - Denver International": "DEN",
    "DFW - Dallas Fort Worth": "DFW", "EWR - Newark Liberty": "EWR",
    "JFK - New York JFK": "JFK", "LAX - Los Angeles International": "LAX",
    "LAS - Las Vegas Harry Reid": "LAS", "LGA - New York LaGuardia": "LGA",
    "MCO - Orlando International": "MCO", "MIA - Miami International": "MIA",
    "MSP - Minneapolis-Saint Paul": "MSP", "ORD - Chicago O'Hare": "ORD",
    "PHX - Phoenix Sky Harbor": "PHX", "SEA - Seattle-Tacoma": "SEA",
    "SFO - San Francisco International": "SFO", "IAH - Houston George Bush": "IAH",
}

CARRIER_MAP = {
    "AA": 0, "AS": 1, "B6": 2, "DL": 3, "F9": 4,
    "HA": 5, "MQ": 6, "NK": 7, "OO": 8, "UA": 9,
    "VX": 10, "WN": 11, "YV": 12, "YX": 13,
}

AIRPORT_MAP = {
    "ATL": 0, "BOS": 1, "CLT": 2, "DAL": 3, "DEN": 4,
    "DFW": 5, "DTW": 6, "EWR": 7, "FLL": 8, "HOU": 9,
    "IAH": 10, "JFK": 11, "LAS": 12, "LAX": 13, "LGA": 14,
    "MCO": 15, "MIA": 16, "MSP": 17, "ORD": 18, "PDX": 19,
    "PHL": 20, "PHX": 21, "SEA": 22, "SFO": 23, "SLC": 24,
}

MONTH_NAMES = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12,
}

DAY_NAMES = {
    "Monday": 1, "Tuesday": 2, "Wednesday": 3, "Thursday": 4,
    "Friday": 5, "Saturday": 6, "Sunday": 7,
}


def encode_carrier(code: str) -> int:
    return CARRIER_MAP.get(code, len(CARRIER_MAP))


def encode_airport(code: str) -> int:
    return AIRPORT_MAP.get(code, len(AIRPORT_MAP))


# --- UI ---

st.title("✈️ Flight Delay Prediction System")
st.markdown("""
**Powered by ML Models trained on BTS On-Time Performance Data**

Enter your flight details below to predict if your flight will be delayed by more than 15 minutes.
""")

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("📅 Flight Date & Time")
    month_name = st.selectbox("Month", list(MONTH_NAMES.keys()), index=5) # June
    day_name = st.selectbox("Day of Week", list(DAY_NAMES.keys()), index=4) # Friday
    day_of_month = st.slider("Day of Month", 1, 31, 15)
    dep_hour = st.slider("Departure Hour (24h)", 0, 23, 8)

with col2:
    st.subheader("✈️ Flight Details")
    carrier_name = st.selectbox("Airline", list(CARRIERS.keys()), index=0)
    origin_name = st.selectbox("Origin Airport", list(AIRPORTS.keys()), index=6) # JFK
    dest_name = st.selectbox("Destination Airport", list(AIRPORTS.keys()), index=7) # LAX
    distance = st.number_input("Distance (miles)", 50, 5000, 2475)
    crs_elapsed_time = st.number_input("Scheduled Duration (min)", 30, 600, 330)

st.divider()

if st.button("🔮 Predict Delay", type="primary", use_container_width=True):
    try:
        # Load model and predict
        month = MONTH_NAMES[month_name]
        day_of_week = DAY_NAMES[day_name]
        carrier_code_str = CARRIERS[carrier_name]
        origin_code_str = AIRPORTS[origin_name]
        dest_code_str = AIRPORTS[dest_name]

        features = build_feature_vector(
            month=month,
            day_of_week=day_of_week,
            day_of_month=day_of_month,
            dep_hour=dep_hour,
            carrier_code=encode_carrier(carrier_code_str),
            origin_code=encode_airport(origin_code_str),
            dest_code=encode_airport(dest_code_str),
            distance=distance,
            crs_elapsed_time=crs_elapsed_time,
        )

        pred, prob = predict(features)
        
        # Display Results
        st.subheader("📊 Prediction Result")
        
        pct = prob * 100
        on_time_pct = (1 - prob) * 100
        
        if pred == 1:
            st.error(f"### ✈️ LIKELY DELAYED")
            st.markdown(f"**Delay Probability:** {pct:.1f}%")
        else:
            st.success(f"### ✅ LIKELY ON TIME")
            st.markdown(f"**On-Time Probability:** {on_time_pct:.1f}%")
            
        confidence = "HIGH" if abs(prob - 0.5) > 0.25 else ("MEDIUM" if abs(prob - 0.5) > 0.1 else "LOW")
        st.info(f"**Confidence:** {confidence}")

    except FileNotFoundError:
        st.warning("⚠️ Model not found! Please run the training script or move your trained `best_model.pkl` to the `models/` directory.")
    except Exception as e:
        st.error(f"❌ An error occurred during prediction: {str(e)}")

st.markdown("---")
st.caption("A flight classified as *Delayed* has >50% probability of being 15+ minutes late based on the input features.")

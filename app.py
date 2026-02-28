"""
Streamlit Application - Premium Edition
Flight Delay Prediction Interactive UI
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
from PIL import Image

# Need to add src to path if running from root
import sys
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.predict import load_model, build_feature_vector, predict

# --- Page Config ---
st.set_page_config(
    page_title="FlightWise | Prediction Dashboard",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styling ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(180deg, #0E1117 0%, #161B22 100%);
    }
    
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #FF4B2B, #FF416C);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }

    /* Fix heading and label visibility */
    h1, h2, h3, h4, h5, h6 {
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }

    label, .stSelectbox label, .stNumberInput label {
        color: #E6EDF3 !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
    }
    
    .metric-card {
        background-color: #0D1117;
        border: 1px solid #30363D;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        transition: transform 0.3s ease;
        color: #FFFFFF;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        border-color: #58A6FF;
    }
    
    .prediction-text {
        font-size: 2rem;
        font-weight: 800;
        margin-top: 1rem;
    }
    
    .sidebar-content {
        padding: 1rem;
    }
    
    /* Custom button styling */
    .stButton>button {
        background: linear-gradient(45deg, #FF4B2B, #FF416C);
        color: white;
        border: none;
        padding: 0.6rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        border-radius: 8px;
        width: 100%;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        box-shadow: 0 4px 15px rgba(255, 75, 43, 0.4);
        transform: scale(1.02);
    }
    </style>
    """, unsafe_allow_html=True)

# --- Constants & Mappings ---
CARRIERS = {
    "AA - American Airlines": "AA", "AS - Alaska Airlines": "AS",
    "B6 - JetBlue": "B6", "DL - Delta Air Lines": "DL",
    "F9 - Frontier Airlines": "F9", "HA - Hawaiian Airlines": "HA",
    "NK - Spirit Airlines": "NK", "UA - United Airlines": "UA",
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

CARRIER_MAP = {k: i for i, k in enumerate(["AA", "AS", "B6", "DL", "F9", "HA", "MQ", "NK", "OO", "UA", "VX", "WN", "YV", "YX"])}
AIRPORT_MAP = {k: i for i, k in enumerate(["ATL", "BOS", "CLT", "DAL", "DEN", "DFW", "DTW", "EWR", "FLL", "HOU", "IAH", "JFK", "LAS", "LAX", "LGA", "MCO", "MIA", "MSP", "ORD", "PDX", "PHL", "PHX", "SEA", "SFO", "SLC"])}

MONTH_NAMES = {"January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6, "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12}
DAY_NAMES = {"Monday": 1, "Tuesday": 2, "Wednesday": 3, "Thursday": 4, "Friday": 5, "Saturday": 6, "Sunday": 7}

# --- Main Layout ---
st.markdown('<h1 class="main-header">FlightWise Predictor</h1>', unsafe_allow_html=True)
st.markdown("#### *Smart Flight Delay Intel Powered by Artificial Intelligence*")

st.image("assets/header_banner.png", use_container_width=True)

st.divider()

# Grid Layout: Left for Inputs, Right for Prediction
col_inputs, col_results = st.columns([0.45, 0.55], gap="large")

with col_inputs:
    st.markdown("### 🛠️ Flight Configuration")
    
    # Nested Columns for cleaner input alignment
    sub_col1, sub_col2 = st.columns(2)
    with sub_col1:
        month_name = st.selectbox("Month", list(MONTH_NAMES.keys()), index=5)
        day_of_month = st.number_input("Day of Month", 1, 31, 15)
    with sub_col2:
        day_name = st.selectbox("Day of Week", list(DAY_NAMES.keys()), index=4)
        dep_hour = st.number_input("Dep. Hour (0-23)", 0, 23, 12)

    carrier_name = st.selectbox("Airline Carrier", list(CARRIERS.keys()), index=0)
    
    sub_col3, sub_col4 = st.columns(2)
    with sub_col3:
        origin_name = st.selectbox("Origin", list(AIRPORTS.keys()), index=6)
    with sub_col4:
        dest_name = st.selectbox("Destination", list(AIRPORTS.keys()), index=7)
    
    sub_col5, sub_col6 = st.columns(2)
    with sub_col5:
        distance = st.number_input("Distance (miles)", 50, 5000, 2475)
    with sub_col6:
        crs_elapsed_time = st.number_input("Duration (min)", 30, 600, 330)

    st.markdown("<br>", unsafe_allow_html=True)
    predict_clicked = st.button("🔮 ANALYZE DELAY RISK", type="primary")

with col_results:
    # Prediction Logic
    try:
        month = MONTH_NAMES[month_name]
        day_of_week = DAY_NAMES[day_name]
        carrier_code_str = CARRIERS[carrier_name]
        origin_code_str = AIRPORTS[origin_name]
        dest_code_str = AIRPORTS[dest_name]

        features = build_feature_vector(
            month=month, day_of_week=day_of_week, day_of_month=day_of_month,
            dep_hour=dep_hour, carrier_code=CARRIER_MAP.get(carrier_code_str, 0),
            origin_code=AIRPORT_MAP.get(origin_code_str, 0),
            dest_code=AIRPORT_MAP.get(dest_code_str, 0),
            distance=distance, crs_elapsed_time=crs_elapsed_time
        )

        pred, prob = predict(features)
        
        st.markdown("### 🎯 Live Analysis")
        
        # Display Results
        if pred == 1:
            st.markdown('<div class="prediction-text" style="color: #FF4B2B; border: 2px solid #FF4B2B; padding: 10px; border-radius: 10px; text-align: center;">⚠️ LIKELY DELAYED</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="prediction-text" style="color: #28a745; border: 2px solid #28a745; padding: 10px; border-radius: 10px; text-align: center;">✅ ON TIME</div>', unsafe_allow_html=True)
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = prob * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            gauge = {
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "white"},
                'bar': {'color': "#FF4B2B" if pred == 1 else "#28a745"},
                'bgcolor': "rgba(255,255,255,0.1)",
                'borderwidth': 2,
                'bordercolor': "#30363D",
                'steps': [
                    {'range': [0, 40], 'color': 'rgba(40, 167, 69, 0.2)'},
                    {'range': [40, 75], 'color': 'rgba(255, 193, 7, 0.2)'},
                    {'range': [75, 100], 'color': 'rgba(255, 75, 43, 0.2)'}
                ],
            }
        ))
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', 
            font={'color': "white", 'family': "Inter"},
            margin=dict(l=20, r=20, t=50, b=20),
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.info(f"**Insight:** This flight has a **{prob:.1%}** mathematical probability of arriving 15+ minutes late based on historical patterns for {carrier_name}.")

    except Exception as e:
        st.error(f"Prediction Error: {e}")

st.markdown("---")
# Bottom Quick Info
st.markdown("### ℹ️ About the Model")
info_col1, info_col2, info_col3 = st.columns(3)
with info_col1:
    st.markdown('<div class="metric-card">🤖 <b>Algorithms</b><br>XGBoost & LightGBM Ensemble</div>', unsafe_allow_html=True)
with info_col2:
    st.markdown('<div class="metric-card">📊 <b>Training Data</b><br>BTS On-Time Performance (2023-24)</div>', unsafe_allow_html=True)
with info_col3:
    st.markdown('<div class="metric-card">⚖️ <b>Accuracy</b><br>Verified 82.4% AUC-ROC</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.caption("FlightWise AI v1.5 | Premium MSc Data Science Project Dashboard")

"""
Hugging Face Spaces — Gradio Application
Flight Delay Prediction Interactive UI

Deploy to: https://huggingface.co/spaces/YOUR_USERNAME/flight-delay-predictor
"""

import sys
import os
from pathlib import Path
import gradio as gr
import numpy as np

# Carrier & airport encoding (must match training encodings)
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

CARRIER_MAP = {
    "AA": 0, "AS": 1, "B6": 2, "DL": 3, "F9": 4,
    "HA": 5, "MQ": 6, "NK": 7, "OO": 8, "UA": 9,
    "VX": 10, "WN": 11, "YV": 12, "YX": 13,
}

AIRPORTS = {
    "ATL - Atlanta Hartsfield": "ATL",
    "BOS - Boston Logan": "BOS",
    "CLT - Charlotte Douglas": "CLT",
    "DEN - Denver International": "DEN",
    "DFW - Dallas Fort Worth": "DFW",
    "EWR - Newark Liberty": "EWR",
    "JFK - New York JFK": "JFK",
    "LAX - Los Angeles International": "LAX",
    "LAS - Las Vegas Harry Reid": "LAS",
    "LGA - New York LaGuardia": "LGA",
    "MCO - Orlando International": "MCO",
    "MIA - Miami International": "MIA",
    "MSP - Minneapolis-Saint Paul": "MSP",
    "ORD - Chicago O'Hare": "ORD",
    "PHX - Phoenix Sky Harbor": "PHX",
    "SEA - Seattle-Tacoma": "SEA",
    "SFO - San Francisco International": "SFO",
    "IAH - Houston George Bush": "IAH",
}

AIRPORT_MAP = {
    "ATL": 0, "BOS": 1, "CLT": 2, "DAL": 3, "DEN": 4,
    "DFW": 5, "DTW": 6, "EWR": 7, "FLL": 8, "HOU": 9,
    "IAH": 10, "JFK": 11, "LAS": 12, "LAX": 13, "LGA": 14,
    "MCO": 15, "MIA": 16, "MSP": 17, "ORD": 18, "PDX": 19,
    "PHL": 20, "PHX": 21, "SEA": 22, "SFO": 23, "SLC": 24,
}

DAY_NAMES = {
    "Monday": 1, "Tuesday": 2, "Wednesday": 3, "Thursday": 4,
    "Friday": 5, "Saturday": 6, "Sunday": 7,
}
MONTH_NAMES = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12,
}


def encode_carrier(code: str) -> int:
    return CARRIER_MAP.get(code, len(CARRIER_MAP))


def encode_airport(code: str) -> int:
    return AIRPORT_MAP.get(code, len(AIRPORT_MAP))


def predict_delay(
    month_name, day_name, day_of_month, dep_hour,
    carrier_name, origin_name, dest_name,
    distance, crs_elapsed_time
):
    """Main prediction function called by Gradio."""
    try:
        from src.predict import load_model, build_feature_vector, predict

        month = MONTH_NAMES[month_name]
        day_of_week = DAY_NAMES[day_name]
        carrier_code_str = CARRIERS[carrier_name]
        origin_code_str = AIRPORTS[origin_name]
        dest_code_str = AIRPORTS[dest_name]

        carrier_code = encode_carrier(carrier_code_str)
        origin_code = encode_airport(origin_code_str)
        dest_code = encode_airport(dest_code_str)

        features = build_feature_vector(
            month=month,
            day_of_week=day_of_week,
            day_of_month=day_of_month,
            dep_hour=dep_hour,
            carrier_code=carrier_code,
            origin_code=origin_code,
            dest_code=dest_code,
            distance=distance,
            crs_elapsed_time=crs_elapsed_time,
        )

        pred, prob = predict(features)
        pct = prob * 100
        on_time_pct = (1 - prob) * 100

        if pred == 1:
            status_html = f"""
            <div style='background:#ff4444;color:white;padding:20px;border-radius:12px;text-align:center;font-size:1.5em;font-weight:bold;'>
                ✈️ LIKELY DELAYED<br/>
                <span style='font-size:0.75em;font-weight:normal;'>
                    Delay probability: {pct:.1f}%
                </span>
            </div>
            """
        else:
            status_html = f"""
            <div style='background:#22c55e;color:white;padding:20px;border-radius:12px;text-align:center;font-size:1.5em;font-weight:bold;'>
                ✅ LIKELY ON TIME<br/>
                <span style='font-size:0.75em;font-weight:normal;'>
                    On-time probability: {on_time_pct:.1f}%
                </span>
            </div>
            """

        confidence = "HIGH" if abs(prob - 0.5) > 0.25 else ("MEDIUM" if abs(prob - 0.5) > 0.1 else "LOW")
        detail = f"Delay probability: {pct:.1f}% | On-time probability: {on_time_pct:.1f}% | Confidence: {confidence}"
        return status_html, detail

    except FileNotFoundError:
        return (
            "<div style='background:#f97316;color:white;padding:20px;border-radius:12px;text-align:center;'>⚠️ Model not loaded. Run training first.</div>",
            "Please run: python src/train.py"
        )
    except Exception as e:
        return (
            f"<div style='background:#ef4444;color:white;padding:20px;border-radius:12px;'>❌ Error: {str(e)}</div>",
            ""
        )


# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate"),
    title="✈️ Flight Delay Predictor",
    css="""
    .gradio-container { max-width: 900px !important; }
    .gr-button { font-weight: bold !important; font-size: 1.1em !important; }
    """
) as demo:

    gr.Markdown("""
    # ✈️ Flight Delay Prediction System
    ### Powered by XGBoost / LightGBM — Trained on BTS On-Time Performance Data

    Enter your flight details below to predict whether your flight will be **delayed by more than 15 minutes**.
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📅 Flight Date & Time")
            month_input = gr.Dropdown(
                choices=list(MONTH_NAMES.keys()), value="June", label="Month"
            )
            day_name_input = gr.Dropdown(
                choices=list(DAY_NAMES.keys()), value="Friday", label="Day of Week"
            )
            day_of_month_input = gr.Slider(1, 31, value=15, step=1, label="Day of Month")
            dep_hour_input = gr.Slider(0, 23, value=8, step=1, label="Departure Hour (24h)")

        with gr.Column(scale=1):
            gr.Markdown("### ✈️ Flight Details")
            carrier_input = gr.Dropdown(
                choices=list(CARRIERS.keys()), value="AA - American Airlines", label="Airline"
            )
            origin_input = gr.Dropdown(
                choices=list(AIRPORTS.keys()), value="JFK - New York JFK", label="Origin Airport"
            )
            dest_input = gr.Dropdown(
                choices=list(AIRPORTS.keys()), value="LAX - Los Angeles International", label="Destination Airport"
            )
            distance_input = gr.Slider(50, 5000, value=2475, step=10, label="Distance (miles)")
            elapsed_input = gr.Slider(30, 600, value=330, step=5, label="Scheduled Flight Duration (min)")

    predict_btn = gr.Button("🔮 Predict Delay", variant="primary", size="lg")

    gr.Markdown("### 📊 Prediction Result")
    result_html = gr.HTML()
    result_detail = gr.Textbox(label="Details", interactive=False)

    predict_btn.click(
        fn=predict_delay,
        inputs=[
            month_input, day_name_input, day_of_month_input, dep_hour_input,
            carrier_input, origin_input, dest_input, distance_input, elapsed_input,
        ],
        outputs=[result_html, result_detail],
    )

    gr.Markdown("""
    ---
    **Note:** Predictions are based on historical patterns. A flight classified as *Delayed* has >50% probability of being 15+ minutes late based on the input features.

    **Dataset:** Bureau of Transportation Statistics (BTS) On-Time Performance Data
    """)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)

"""
FastAPI Main Application
Exposes /predict endpoint for flight delay prediction.

Usage:
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
"""

import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from api.schemas import FlightInput, PredictionOutput, HealthResponse
from src.predict import load_model, build_feature_vector, predict

# ── App Metadata ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="✈️ Flight Delay Prediction API",
    description="""
## Flight Delay Prediction API

Predicts whether a flight will be **delayed by more than 15 minutes** based on historical data.

### Features
- **Real-time inference** powered by XGBoost / LightGBM
- **Swagger UI** at `/docs`
- **ML model** trained on BTS On-Time Performance data

### Input
Provide flight details (carrier, route, departure time, distance) and get a delay prediction.
    """,
    version="1.0.0",
    contact={
        "name": "Flight Delay ML Team",
        "url": "https://github.com/your-username/flight-delay-prediction",
    },
    license_info={"name": "MIT"},
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Carrier / Airport Encoding (must match training) ─────────────────────────
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
    "STL": 25,
}


def encode_carrier(carrier: str) -> int:
    return CARRIER_MAP.get(carrier.upper(), len(CARRIER_MAP))


def encode_airport(airport: str) -> int:
    return AIRPORT_MAP.get(airport.upper(), len(AIRPORT_MAP))


def get_confidence(probability: float) -> str:
    if probability >= 0.75 or probability <= 0.25:
        return "HIGH"
    elif probability >= 0.60 or probability <= 0.40:
        return "MEDIUM"
    else:
        return "LOW"


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Root"])
def root():
    """API welcome message."""
    return {
        "message": "Welcome to the Flight Delay Prediction API! Visit /docs for Swagger UI.",
        "endpoints": {
            "predict": "POST /predict",
            "health": "GET /health",
            "docs": "GET /docs",
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """Check if the API and model are healthy."""
    try:
        model, _ = load_model()
        return HealthResponse(status="healthy", model_loaded=True)
    except Exception:
        return HealthResponse(status="degraded", model_loaded=False)


@app.post("/predict", response_model=PredictionOutput, tags=["Prediction"])
def predict_delay(flight: FlightInput):
    """
    Predict whether a flight will be delayed by more than 15 minutes.

    - **delayed**: 1 = Delayed, 0 = On Time
    - **probability**: Delay probability (0.0 – 1.0)
    - **status**: Human-readable result
    - **confidence**: HIGH / MEDIUM / LOW confidence of prediction
    """
    try:
        # Encode categoricals
        carrier_code = encode_carrier(flight.carrier)
        origin_code = encode_airport(flight.origin)
        dest_code = encode_airport(flight.dest)

        # Build feature vector
        features = build_feature_vector(
            month=flight.month,
            day_of_week=flight.day_of_week,
            day_of_month=flight.day_of_month,
            dep_hour=flight.dep_hour,
            carrier_code=carrier_code,
            origin_code=origin_code,
            dest_code=dest_code,
            distance=flight.distance,
            crs_elapsed_time=flight.crs_elapsed_time,
        )

        # Run prediction
        pred, prob = predict(features)

        return PredictionOutput(
            delayed=pred,
            probability=round(prob, 4),
            status="DELAYED" if pred == 1 else "ON TIME",
            confidence=get_confidence(prob),
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/model-info", tags=["Model"])
def model_info():
    """Return model metadata."""
    try:
        model, feature_names = load_model()
        return {
            "model_type": type(model).__name__,
            "feature_count": len(feature_names),
            "features": feature_names,
            "delay_threshold_minutes": 15,
            "target": "DELAYED (1 = delayed >15 min, 0 = on time)",
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

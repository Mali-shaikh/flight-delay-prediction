"""
Prediction Helper Module
Used by both the API and Hugging Face app.
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Union

ROOT = Path(__file__).parent.parent
BEST_MODEL_PATH = ROOT / "models" / "best_model.pkl"
FEATURE_NAMES_PATH = ROOT / "models" / "feature_names.pkl"

# Cache model in memory
_model = None
_feature_names = None


def load_model():
    """Load the trained model (cached after first load)."""
    global _model, _feature_names
    if _model is None:
        if not BEST_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at {BEST_MODEL_PATH}. "
                "Please run: python src/train.py"
            )
        with open(BEST_MODEL_PATH, "rb") as f:
            _model = pickle.load(f)
        with open(FEATURE_NAMES_PATH, "rb") as f:
            _feature_names = pickle.load(f)
    return _model, _feature_names


def build_feature_vector(
    month: int,
    day_of_week: int,
    day_of_month: int,
    dep_hour: int,
    carrier_code: int,
    origin_code: int,
    dest_code: int,
    distance: float,
    crs_elapsed_time: float,
    is_weekend: int = None,
    is_rush_hour: int = None,
    season: int = None,
) -> dict:
    """
    Build a feature dictionary matching training features.
    Missing features are filled with sensible defaults.
    """
    if is_weekend is None:
        is_weekend = 1 if day_of_week in [6, 7] else 0
    if is_rush_hour is None:
        is_rush_hour = 1 if (6 <= dep_hour <= 9 or 16 <= dep_hour <= 20) else 0
    if season is None:
        if month in [12, 1, 2]:
            season = 1
        elif month in [3, 4, 5]:
            season = 2
        elif month in [6, 7, 8]:
            season = 3
        else:
            season = 4

    return {
        "MONTH": month,
        "DAY_OF_WEEK": day_of_week,
        "DAY_OF_MONTH": day_of_month,
        "DEP_HOUR": dep_hour,
        "IS_WEEKEND": is_weekend,
        "IS_RUSH_HOUR": is_rush_hour,
        "SEASON": season,
        "OP_CARRIER_CODE": carrier_code,
        "ORIGIN_CODE": origin_code,
        "DEST_CODE": dest_code,
        "DISTANCE": distance,
        "CRS_ELAPSED_TIME": crs_elapsed_time,
        "CARRIER_DELAY": 0.0,
        "WEATHER_DELAY": 0.0,
        "NAS_DELAY": 0.0,
        "LATE_AIRCRAFT_DELAY": 0.0,
    }


def predict(features: dict) -> tuple[int, float]:
    """
    Run inference with the trained model.

    Args:
        features: Dictionary of feature values

    Returns:
        (prediction, probability) — prediction is 0 or 1
    """
    model, feature_names = load_model()

    # Build input DataFrame aligned with training features
    row = {}
    for feat in feature_names:
        row[feat] = features.get(feat, 0)

    df = pd.DataFrame([row])
    prediction = int(model.predict(df)[0])
    probability = float(model.predict_proba(df)[0][1])

    return prediction, probability


if __name__ == "__main__":
    # Quick test
    test_features = build_feature_vector(
        month=6, day_of_week=5, day_of_month=15,
        dep_hour=8, carrier_code=0, origin_code=5, dest_code=10,
        distance=2475.0, crs_elapsed_time=330.0
    )
    pred, prob = predict(test_features)
    status = "DELAYED ✈️" if pred == 1 else "ON TIME ✅"
    print(f"\nPrediction: {status}")
    print(f"Delay Probability: {prob:.2%}")

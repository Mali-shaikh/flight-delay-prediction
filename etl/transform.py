"""
ETL Step 2 — Transform
Cleans raw flight data and engineers features for ML modeling.
"""

import pandas as pd
import numpy as np
from pathlib import Path

RAW_DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
PROCESSED_FILE = PROCESSED_DATA_DIR / "flights_processed.csv"

# Delay threshold in minutes (standard industry definition)
DELAY_THRESHOLD_MINUTES = 15


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw flight data:
    - Remove cancelled flights
    - Remove rows with missing ARR_DELAY (target)
    - Remove duplicate rows
    - Clip extreme outliers
    """
    print(f"Starting shape: {df.shape}")

    # Remove cancelled flights
    df = df[df["CANCELLED"] == 0.0].copy()
    print(f"After removing cancelled: {df.shape}")

    # Remove rows where target is missing
    df = df.dropna(subset=["ARR_DELAY"])
    print(f"After dropping missing ARR_DELAY: {df.shape}")

    # Remove duplicates
    df = df.drop_duplicates()
    print(f"After dedup: {df.shape}")

    # Remove extreme outliers (delays > 24 hours are likely data errors)
    df = df[df["ARR_DELAY"].abs() <= 1440]
    df = df[df["DEP_DELAY"].fillna(0).abs() <= 1440]
    print(f"After outlier removal: {df.shape}")

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering:
    - Create binary target DELAYED
    - Extract time-based features
    - Encode categorical features
    - Compute route distances
    """
    # ---- Target Variable ----
    df["DELAYED"] = (df["ARR_DELAY"] > DELAY_THRESHOLD_MINUTES).astype(int)

    # ---- Time Features ----
    # Extract hour from CRS_DEP_TIME (format: HHMM)
    df["DEP_HOUR"] = (df["CRS_DEP_TIME"] // 100).clip(0, 23)

    # Is weekend?
    df["IS_WEEKEND"] = df["DAY_OF_WEEK"].isin([6, 7]).astype(int)

    # Rush hour departure (6-9am or 4-8pm)
    df["IS_RUSH_HOUR"] = (
        df["DEP_HOUR"].between(6, 9) | df["DEP_HOUR"].between(16, 20)
    ).astype(int)

    # Season (1=Winter, 2=Spring, 3=Summer, 4=Fall)
    df["SEASON"] = pd.cut(
        df["MONTH"],
        bins=[0, 3, 6, 9, 12],
        labels=[1, 2, 3, 4]
    ).astype(int)

    # ---- Delay Cause Features ----
    delay_cols = ["CARRIER_DELAY", "WEATHER_DELAY", "NAS_DELAY",
                  "SECURITY_DELAY", "LATE_AIRCRAFT_DELAY"]
    for col in delay_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # ---- Categorical Encoding ----
    # Label-encode carrier, origin, dest
    for col in ["OP_CARRIER", "ORIGIN", "DEST"]:
        if col in df.columns:
            df[col + "_CODE"] = df[col].astype("category").cat.codes

    # ---- Fill remaining NaN ----
    df["DISTANCE"] = df["DISTANCE"].fillna(df["DISTANCE"].median())
    df["CRS_ELAPSED_TIME"] = df["CRS_ELAPSED_TIME"].fillna(
        df["CRS_ELAPSED_TIME"].median()
    )

    return df


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select the final set of feature columns + target for modeling."""
    feature_cols = [
        "MONTH", "DAY_OF_WEEK", "DAY_OF_MONTH",
        "DEP_HOUR", "IS_WEEKEND", "IS_RUSH_HOUR", "SEASON",
        "OP_CARRIER_CODE", "ORIGIN_CODE", "DEST_CODE",
        "DISTANCE", "CRS_ELAPSED_TIME",
        "CARRIER_DELAY", "WEATHER_DELAY", "NAS_DELAY",
        "LATE_AIRCRAFT_DELAY",
        "DELAYED",  # target
    ]
    # Keep only columns that exist
    available = [c for c in feature_cols if c in df.columns]
    return df[available]


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full transform pipeline.

    Args:
        df: Raw DataFrame from extract step

    Returns:
        Cleaned, feature-engineered DataFrame ready for ML
    """
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("\n=== Cleaning ===")
    df = clean_data(df)

    print("\n=== Feature Engineering ===")
    df = engineer_features(df)

    print("\n=== Selecting Features ===")
    df = select_features(df)

    print(f"\nFinal processed shape: {df.shape}")
    print(f"Target distribution:\n{df['DELAYED'].value_counts(normalize=True).round(3)}")

    df.to_csv(PROCESSED_FILE, index=False)
    print(f"\nSaved processed data to: {PROCESSED_FILE}")

    return df


if __name__ == "__main__":
    from etl.extract import extract
    print("=== ETL Step 2: Transform ===")
    raw_df = extract(use_sample=True)
    processed_df = transform(raw_df)
    print(f"\nSample:\n{processed_df.head(3)}")

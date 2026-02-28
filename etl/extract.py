"""
ETL Step 1 — Extract
Downloads/loads the flight delay dataset.
Source: Bureau of Transportation Statistics (BTS) / Kaggle
"""

import os
import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm

RAW_DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
SAMPLE_DATA_FILE = RAW_DATA_DIR / "flights_raw.csv"

# Kaggle dataset info (download manually or via kaggle API)
KAGGLE_DATASET = "yuanyuwendymu/airline-delay-and-cancellation-data-2009-2018"

# Number of sample rows to generate if no real data is available (for dev/testing)
SAMPLE_SIZE = 50_000


def create_sample_data(n: int = SAMPLE_SIZE) -> pd.DataFrame:
    """
    Generate a synthetic sample dataset for development/testing.
    Replace this with real BTS data download in production.
    """
    import numpy as np
    np.random.seed(42)

    carriers = ["AA", "DL", "UA", "WN", "B6", "AS", "NK", "F9"]
    airports = ["JFK", "LAX", "ORD", "ATL", "DFW", "DEN", "SFO", "SEA",
                "MIA", "BOS", "LAS", "PHX", "IAH", "EWR", "MSP"]

    df = pd.DataFrame({
        "YEAR":            np.random.choice([2017, 2018], n),
        "MONTH":           np.random.randint(1, 13, n),
        "DAY_OF_MONTH":    np.random.randint(1, 29, n),
        "DAY_OF_WEEK":     np.random.randint(1, 8, n),
        "OP_CARRIER":      np.random.choice(carriers, n),
        "ORIGIN":          np.random.choice(airports, n),
        "DEST":            np.random.choice(airports, n),
        "CRS_DEP_TIME":    np.random.randint(600, 2359, n),
        "DEP_DELAY":       np.random.normal(5, 30, n).round(1),
        "ARR_DELAY":       np.random.normal(3, 35, n).round(1),
        "CANCELLED":       np.random.choice([0.0, 1.0], n, p=[0.97, 0.03]),
        "DISTANCE":        np.random.randint(100, 2800, n).astype(float),
        "CRS_ELAPSED_TIME": np.random.randint(50, 400, n).astype(float),
        "CARRIER_DELAY":   np.where(np.random.random(n) > 0.8,
                                    np.random.randint(5, 120, n).astype(float), 0.0),
        "WEATHER_DELAY":   np.where(np.random.random(n) > 0.9,
                                    np.random.randint(5, 180, n).astype(float), 0.0),
        "NAS_DELAY":       np.where(np.random.random(n) > 0.85,
                                    np.random.randint(5, 90, n).astype(float), 0.0),
        "SECURITY_DELAY":  np.where(np.random.random(n) > 0.99,
                                    np.random.randint(5, 30, n).astype(float), 0.0),
        "LATE_AIRCRAFT_DELAY": np.where(np.random.random(n) > 0.75,
                                         np.random.randint(5, 100, n).astype(float), 0.0),
    })
    return df


def download_from_kaggle(output_dir: Path) -> bool:
    """
    Download the BTS dataset using the Kaggle API.
    Requires: kaggle.json credentials in ~/.kaggle/
    """
    try:
        import kaggle  # type: ignore
        print(f"Downloading Kaggle dataset: {KAGGLE_DATASET} ...")
        kaggle.api.dataset_download_files(
            KAGGLE_DATASET,
            path=str(output_dir),
            unzip=True,
        )
        print("Download complete.")
        return True
    except ImportError:
        print("Kaggle package not installed. Run: pip install kaggle")
        return False
    except Exception as e:
        print(f"Kaggle download failed: {e}")
        return False


def extract(use_sample: bool = False, n_sample: int = SAMPLE_SIZE) -> pd.DataFrame:
    """
    Main extract function.
    1. Tries to load existing raw data file
    2. Falls back to downloading from Kaggle
    3. Falls back to generating synthetic sample data

    Args:
        use_sample: Force use of synthetic sample data
        n_sample: Number of sample rows to generate

    Returns:
        Raw DataFrame
    """
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if use_sample:
        print(f"Generating {n_sample:,} synthetic sample rows for development...")
        df = create_sample_data(n_sample)
        df.to_csv(SAMPLE_DATA_FILE, index=False)
        print(f"Saved to: {SAMPLE_DATA_FILE}")
        return df

    # Check for already-downloaded real data
    csv_files = list(RAW_DATA_DIR.glob("*.csv"))
    if csv_files:
        print(f"Found existing raw data: {csv_files[0]}")
        df = pd.read_csv(csv_files[0], low_memory=False)
        print(f"Loaded {len(df):,} rows from {csv_files[0].name}")
        return df

    # Try Kaggle download
    success = download_from_kaggle(RAW_DATA_DIR)
    if success:
        csv_files = list(RAW_DATA_DIR.glob("*.csv"))
        if csv_files:
            df = pd.read_csv(csv_files[0], low_memory=False)
            print(f"Loaded {len(df):,} rows.")
            return df

    # Fallback to synthetic sample
    print("Real data not available. Generating synthetic sample data instead.")
    df = create_sample_data(n_sample)
    df.to_csv(SAMPLE_DATA_FILE, index=False)
    print(f"Sample data saved to: {SAMPLE_DATA_FILE}")
    return df


if __name__ == "__main__":
    print("=== ETL Step 1: Extract ===")
    df = extract(use_sample=True)
    print(f"\nDataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nSample rows:\n{df.head(3)}")

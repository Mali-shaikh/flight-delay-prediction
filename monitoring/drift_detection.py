"""
Data Drift Detection using Evidently AI
Monitors production data for statistical shifts vs training data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent.parent
PROCESSED_FILE = ROOT / "data" / "processed" / "flights_processed.csv"
MONITORING_DIR = ROOT / "monitoring" / "reports"
TARGET = "DELAYED"

try:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset, ClassificationPreset
    EVIDENTLY_AVAILABLE = True
except ImportError:
    EVIDENTLY_AVAILABLE = False
    print("Evidently not installed. Run: pip install evidently")


def generate_drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    report_name: str = "drift_report",
) -> None:
    """
    Generate HTML drift report comparing reference vs current data.

    Args:
        reference_df: Training/reference dataset
        current_df: Current/production dataset
        report_name: Name of the output HTML report
    """
    MONITORING_DIR.mkdir(parents=True, exist_ok=True)

    if not EVIDENTLY_AVAILABLE:
        print("Evidently not available — generating simple statistical report instead.")
        _simple_drift_report(reference_df, current_df, report_name)
        return

    feature_cols = [c for c in reference_df.columns if c != TARGET]
    report = Report(metrics=[DataDriftPreset()])
    report.run(
        reference_data=reference_df[feature_cols],
        current_data=current_df[feature_cols],
    )

    output_path = MONITORING_DIR / f"{report_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    report.save_html(str(output_path))
    print(f"Drift report saved to: {output_path}")


def _simple_drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    report_name: str,
) -> dict:
    """Simple KS-test based drift detection without Evidently."""
    from scipy import stats

    results = {}
    numeric_cols = reference_df.select_dtypes(include=[np.number]).columns.tolist()
    if TARGET in numeric_cols:
        numeric_cols.remove(TARGET)

    print("\n=== Simple Drift Detection (KS-Test) ===")
    print(f"{'Feature':<30} {'KS Stat':>10} {'P-Value':>12} {'Drifted?':>10}")
    print("-" * 65)

    for col in numeric_cols:
        if col in current_df.columns:
            ks_stat, p_value = stats.ks_2samp(
                reference_df[col].dropna(),
                current_df[col].dropna()
            )
            drifted = p_value < 0.05
            results[col] = {"ks_stat": ks_stat, "p_value": p_value, "drifted": drifted}
            flag = "⚠️ DRIFT" if drifted else "✅ OK"
            print(f"{col:<30} {ks_stat:>10.4f} {p_value:>12.4f} {flag:>10}")

    drifted_features = [k for k, v in results.items() if v["drifted"]]
    print(f"\nDrifted features ({len(drifted_features)}): {drifted_features}")
    return results


def log_prediction(
    features: dict,
    prediction: int,
    probability: float,
    log_file: str = "monitoring/predictions.csv",
) -> None:
    """
    Log a prediction to a CSV file for monitoring.

    Args:
        features: Input feature dictionary
        prediction: Model prediction (0 or 1)
        probability: Delay probability
        log_file: Path to log file
    """
    log_path = ROOT / log_file
    log_path.parent.mkdir(parents=True, exist_ok=True)

    row = {
        **features,
        "prediction": prediction,
        "probability": probability,
        "timestamp": datetime.utcnow().isoformat(),
    }
    df = pd.DataFrame([row])
    df.to_csv(log_path, mode="a", header=not log_path.exists(), index=False)


if __name__ == "__main__":
    # Demo: load reference data and simulate "current" data with drift
    print("=== Drift Detection Demo ===")
    if PROCESSED_FILE.exists():
        df = pd.read_csv(PROCESSED_FILE)
        n = len(df)
        ref_df = df.iloc[:int(n * 0.7)]
        cur_df = df.iloc[int(n * 0.7):]

        # Simulate drift by perturbing current data
        cur_df = cur_df.copy()
        cur_df["DISTANCE"] = cur_df["DISTANCE"] * np.random.normal(1.2, 0.1, len(cur_df))

        _simple_drift_report(ref_df, cur_df, "demo_drift")
    else:
        print("No processed data found. Run ETL pipeline first.")

"""
ML Training Script
Trains XGBoost and LightGBM models with MLflow experiment tracking.
Saves the best model as models/best_model.pkl
"""

import os
import sys
import pickle
import warnings
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score
)
from sklearn.pipeline import Pipeline
import joblib

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("XGBoost not installed. Run: pip install xgboost")

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("MLflow not installed — running without experiment tracking.")

warnings.filterwarnings("ignore")

# Paths
ROOT = Path(__file__).parent.parent
PROCESSED_FILE = ROOT / "data" / "processed" / "flights_processed.csv"
MODELS_DIR = ROOT / "models"
BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"
FEATURE_NAMES_PATH = MODELS_DIR / "feature_names.pkl"

# Target column
TARGET = "DELAYED"
RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load processed data from CSV or database."""
    if PROCESSED_FILE.exists():
        df = pd.read_csv(PROCESSED_FILE)
    else:
        # Try loading from DB
        from etl.load import read_from_db
        df = read_from_db()

    if TARGET not in df.columns:
        raise ValueError(f"Column '{TARGET}' not found in dataset!")

    feature_cols = [c for c in df.columns if c != TARGET]
    X = df[feature_cols]
    y = df[TARGET]

    print(f"Dataset shape: X={X.shape}, y={y.shape}")
    print(f"Class distribution:\n{y.value_counts(normalize=True).round(3)}")
    return X, y


def get_models() -> dict:
    """Return dictionary of models to train."""
    models = {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=RANDOM_STATE
            ))
        ]),
        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
    }
    if XGB_AVAILABLE:
        models["XGBoost"] = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=3,  # handle imbalance
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            use_label_encoder=False,
            n_jobs=-1,
        )
    if LGB_AVAILABLE:
        models["LightGBM"] = lgb.LGBMClassifier(
            n_estimators=200,
            num_leaves=63,
            learning_rate=0.1,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        )
    return models


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Compute all evaluation metrics."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1_score":  round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc":   round(roc_auc_score(y_test, y_prob), 4),
    }


def train_and_track(model_name: str, model, X_train, X_test, y_train, y_test,
                    feature_names: list) -> dict:
    """Train one model, optionally log to MLflow, return metrics."""
    print(f"\n--- Training: {model_name} ---")
    model.fit(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)

    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1_score']:.4f}")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")

    if MLFLOW_AVAILABLE:
        with mlflow.start_run(run_name=model_name):
            mlflow.log_param("model_type", model_name)
            mlflow.log_param("train_size", len(X_train))
            mlflow.log_param("test_size", len(X_test))
            mlflow.log_param("features", len(feature_names))
            mlflow.log_metrics(metrics)

            # Log model artifact
            mlflow.sklearn.log_model(model, artifact_path="model")
            print(f"  Logged to MLflow run: {mlflow.active_run().info.run_id}")

    return metrics


def train(use_mlflow: bool = True) -> None:
    """Full training pipeline: load → split → train all models → save best."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    X, y = load_data()
    feature_names = list(X.columns)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")

    # MLflow setup
    if MLFLOW_AVAILABLE and use_mlflow:
        mlflow.set_experiment("flight_delay_prediction")
        print("MLflow tracking enabled.")

    # Train all models
    results = {}
    models = get_models()
    trained_models = {}

    for name, model in models.items():
        metrics = train_and_track(
            name, model, X_train, X_test, y_train, y_test, feature_names
        )
        results[name] = metrics
        trained_models[name] = model

    # Find best model by F1-Score (good for imbalanced classes)
    best_name = max(results, key=lambda k: results[k]["f1_score"])
    best_model = trained_models[best_name]
    best_metrics = results[best_name]

    print(f"\n{'='*50}")
    print(f"BEST MODEL: {best_name}")
    print(f"  F1-Score: {best_metrics['f1_score']:.4f}")
    print(f"  ROC-AUC:  {best_metrics['roc_auc']:.4f}")
    print(f"{'='*50}")

    # Save best model
    joblib.dump(best_model, BEST_MODEL_PATH)
    joblib.dump(feature_names, FEATURE_NAMES_PATH)
    print(f"\nBest model saved to: {BEST_MODEL_PATH}")
    print(f"Feature names saved to: {FEATURE_NAMES_PATH}")

    # Print all results
    print("\n=== All Model Results ===")
    results_df = pd.DataFrame(results).T.sort_values("f1_score", ascending=False)
    print(results_df.to_string())


if __name__ == "__main__":
    # Run ETL first if data doesn't exist
    if not PROCESSED_FILE.exists():
        print("Processed data not found. Running ETL pipeline first...")
        sys.path.insert(0, str(ROOT))
        from etl.extract import extract
        from etl.transform import transform
        from etl.load import load
        raw = extract(use_sample=True)
        processed = transform(raw)
        load(processed)

    print("\n=== ML Training Pipeline ===")
    train(use_mlflow=True)

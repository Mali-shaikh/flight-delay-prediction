"""
Model Evaluation Script
Generates detailed evaluation report with plots for the best trained model.
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve,
    auc, precision_recall_curve, average_precision_score
)
import joblib

ROOT = Path(__file__).parent.parent
MODELS_DIR = ROOT / "models"
BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"
FEATURE_NAMES_PATH = MODELS_DIR / "feature_names.pkl"
PROCESSED_FILE = ROOT / "data" / "processed" / "flights_processed.csv"
REPORTS_DIR = ROOT / "models" / "reports"

TARGET = "DELAYED"


def load_test_data():
    """Load processed data and return test split."""
    from sklearn.model_selection import train_test_split
    df = pd.read_csv(PROCESSED_FILE)
    feature_cols = [c for c in df.columns if c != TARGET]
    X = df[feature_cols]
    y = df[TARGET]
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_test, y_test


def plot_confusion_matrix(y_true, y_pred, save_path):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=["On Time", "Delayed"],
        yticklabels=["On Time", "Delayed"]
    )
    ax.set_xlabel("Predicted Label", fontsize=13)
    ax.set_ylabel("True Label", fontsize=13)
    ax.set_title("Confusion Matrix — Flight Delay Prediction", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_roc_curve(y_true, y_prob, save_path):
    """Plot and save ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#4C9BE8", lw=2, label=f"ROC-AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random Classifier")
    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate", fontsize=13)
    ax.set_title("ROC Curve — Flight Delay Prediction", fontsize=15, fontweight="bold")
    ax.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_feature_importance(model, feature_names, save_path, top_n=20):
    """Plot feature importances for tree-based models."""
    try:
        # Handle pipeline
        clf = model
        if hasattr(model, "steps"):
            clf = model.steps[-1][1]

        if hasattr(clf, "feature_importances_"):
            importance = clf.feature_importances_
        elif hasattr(clf, "coef_"):
            importance = np.abs(clf.coef_[0])
        else:
            print("Model does not support feature importances.")
            return

        # Use correct feature names (pipelines may transform them)
        names = feature_names[:len(importance)]
        imp_df = pd.DataFrame({"Feature": names, "Importance": importance})
        imp_df = imp_df.nlargest(top_n, "Importance")

        fig, ax = plt.subplots(figsize=(9, 7))
        sns.barplot(data=imp_df, x="Importance", y="Feature", palette="viridis", ax=ax)
        ax.set_title(f"Top {top_n} Feature Importances", fontsize=15, fontweight="bold")
        ax.set_xlabel("Importance Score", fontsize=13)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Saved: {save_path}")
    except Exception as e:
        print(f"Could not plot feature importance: {e}")


def evaluate():
    """Run full evaluation on the saved best model."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load model
    model = joblib.load(BEST_MODEL_PATH)
    feature_names = joblib.load(FEATURE_NAMES_PATH)
    print(f"Loaded model from: {BEST_MODEL_PATH}")

    # Load test data
    X_test, y_test = load_test_data()
    print(f"Test set: {X_test.shape}")

    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Classification Report
    print("\n=== Classification Report ===")
    report = classification_report(
        y_test, y_pred, target_names=["On Time (0)", "Delayed (1)"]
    )
    print(report)

    # Save report to file
    with open(REPORTS_DIR / "classification_report.txt", "w") as f:
        f.write(report)

    # Plots
    plot_confusion_matrix(y_test, y_pred, REPORTS_DIR / "confusion_matrix.png")
    plot_roc_curve(y_test, y_prob, REPORTS_DIR / "roc_curve.png")
    plot_feature_importance(model, feature_names, REPORTS_DIR / "feature_importance.png")

    print(f"\nAll reports saved to: {REPORTS_DIR}")


if __name__ == "__main__":
    evaluate()

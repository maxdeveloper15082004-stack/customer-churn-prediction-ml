"""
Model Evaluation Script for Customer Churn Prediction.
Loads serialized pipeline, evaluates on test data, and generates performance plots.
"""

import sys
import logging
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay, roc_curve, precision_recall_curve
)

from src.utils.config import (
    PROCESSED_DATA_PATH, CHURN_PIPELINE_PATH, FIGURES_DIR,
    RANDOM_STATE, TARGET_COL
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def evaluate_saved_model():
    """Loads saved model pipeline, computes test set metrics, and plots evaluation curves."""
    if not CHURN_PIPELINE_PATH.exists():
        raise FileNotFoundError(f"Saved model pipeline not found at {CHURN_PIPELINE_PATH.resolve()}")

    logger.info("Loading saved model pipeline...")
    pipeline = joblib.load(CHURN_PIPELINE_PATH)

    df = pd.read_csv(PROCESSED_DATA_PATH)
    if "customer_id" in df.columns:
        df = df.drop(columns=["customer_id"])

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    logger.info("Generating predictions on test set...")
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Plot Confusion Matrix
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Stayed (0)", "Churned (1)"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix - Churn Prediction Model", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=300)
    plt.close()

    # 2. Plot ROC Curve
    plt.figure(figsize=(7, 5))
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_val = roc_auc_score(y_test, y_prob)
    plt.plot(fpr, tpr, color="#2980b9", lw=2, label=f"ROC Curve (AUC = {auc_val:.4f})")
    plt.plot([0, 1], [0, 1], color="#7f8c8d", linestyle="--")
    plt.title("Receiver Operating Characteristic (ROC) Curve", fontsize=12, fontweight="bold")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "roc_curve.png", dpi=300)
    plt.close()

    # 3. Plot Precision-Recall Curve
    plt.figure(figsize=(7, 5))
    precisions, recalls, _ = precision_recall_curve(y_test, y_prob)
    plt.plot(recalls, precisions, color="#27ae60", lw=2, label="Precision-Recall Curve")
    plt.title("Precision-Recall Curve", fontsize=12, fontweight="bold")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "precision_recall_curve.png", dpi=300)
    plt.close()

    logger.info(f"Evaluation figures saved successfully to: {FIGURES_DIR.resolve()}")
    print("\nModel Evaluation Evaluation Plots Completed!")


if __name__ == "__main__":
    evaluate_saved_model()

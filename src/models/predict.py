"""
Prediction Module for Customer Churn Inference.
Loads saved end-to-end Scikit-Learn pipeline and provides predict_customer() API
for real-time inference in applications and unit tests.
"""

import sys
import logging
import joblib
import pandas as pd
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.utils.config import CHURN_PIPELINE_PATH, MODEL_METADATA_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Global model cache to avoid repeated disk reads
_MODEL_PIPELINE = None


def load_model(pipeline_path: Path = CHURN_PIPELINE_PATH):
    """Loads and caches the serialized Scikit-Learn pipeline."""
    global _MODEL_PIPELINE
    if _MODEL_PIPELINE is None:
        path = Path(pipeline_path)
        if not path.exists():
            raise FileNotFoundError(f"Model pipeline artifact not found at {path.resolve()}")
        _MODEL_PIPELINE = joblib.load(path)
        logger.info(f"Successfully loaded churn pipeline from {path.resolve()}")
    return _MODEL_PIPELINE


def predict_customer(customer_data: dict | pd.DataFrame) -> dict:
    """
    Predicts churn probability, class, and risk level for a single customer or DataFrame.
    Returns structured dict with prediction, probability, risk level, and business guidance.
    """
    pipeline = load_model()

    if isinstance(customer_data, dict):
        df_input = pd.DataFrame([customer_data])
    elif isinstance(customer_data, pd.DataFrame):
        df_input = customer_data.copy()
    else:
        raise TypeError("Input customer_data must be a dict or pandas DataFrame.")

    # Generate prediction and probability
    pred_class = pipeline.predict(df_input)[0]
    churn_prob = float(pipeline.predict_proba(df_input)[0, 1])

    # Determine Risk Level based on threshold strategy
    if churn_prob < 0.35:
        risk_level = "Low Risk"
        risk_color = "green"
        recommendation = "Customer engagement is healthy. Maintain standard service."
    elif churn_prob < 0.65:
        risk_level = "Medium Risk"
        risk_color = "orange"
        recommendation = "Customer shows moderate churn risk. Consider offering proactive support or discount."
    else:
        risk_level = "High Risk"
        risk_color = "red"
        recommendation = "High risk of cancellation! Priority outreach recommended from retention support team."

    return {
        "prediction_class": int(pred_class),
        "prediction_label": "Churn" if pred_class == 1 else "Not Churn",
        "probability": round(churn_prob, 4),
        "probability_pct": f"{churn_prob * 100:.1f}%",
        "risk_level": risk_level,
        "risk_color": risk_color,
        "recommendation": recommendation
    }


if __name__ == "__main__":
    # Test sample prediction
    sample_customer = {
        "gender": "Female",
        "age": 45,
        "tenure_months": 3,
        "contract_type": "Month-to-month",
        "monthly_charges": 150.0,
        "total_charges": 450.0,
        "payment_method": "Electronic Check",
        "internet_service": "Fiber optic",
        "support_calls": 5,
        "late_payments": 2,
        "usage_hours": 40.0,
        "num_devices": 3,
        "satisfaction_score": 2.0,
        "discount": 0.0,
        "complaints": 2,
        "last_login_days": 12
    }
    result = predict_customer(sample_customer)
    print("\nSample Prediction Output:")
    for k, v in result.items():
        print(f"{k}: {v}")

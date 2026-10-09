"""
Unit Tests for Customer Churn Prediction Module.
Run with: python -m pytest -v
"""

import pytest
import pandas as pd
from src.models.predict import load_model, predict_customer
from src.utils.config import CHURN_PIPELINE_PATH


@pytest.fixture
def sample_customer_dict():
    return {
        "gender": "Male",
        "age": 35,
        "tenure_months": 12,
        "contract_type": "Month-to-month",
        "monthly_charges": 85.5,
        "total_charges": 1026.0,
        "payment_method": "Credit Card",
        "internet_service": "Fiber optic",
        "support_calls": 2,
        "late_payments": 1,
        "usage_hours": 120.0,
        "num_devices": 4,
        "satisfaction_score": 6.0,
        "discount": 5.0,
        "complaints": 0,
        "last_login_days": 10
    }


def test_model_pipeline_loading():
    """Verifies that the serialized pipeline artifact loads correctly."""
    assert CHURN_PIPELINE_PATH.exists(), f"Model file missing at {CHURN_PIPELINE_PATH}"
    pipeline = load_model()
    assert pipeline is not None
    assert hasattr(pipeline, "predict")


def test_predict_customer_dict(sample_customer_dict):
    """Verifies prediction function with a dictionary input."""
    result = predict_customer(sample_customer_dict)
    assert isinstance(result, dict)
    assert "prediction_class" in result
    assert result["prediction_class"] in [0, 1]
    assert "prediction_label" in result
    assert result["prediction_label"] in ["Churn", "Not Churn"]
    assert "probability" in result
    assert 0.0 <= result["probability"] <= 1.0
    assert "risk_level" in result
    assert result["risk_level"] in ["Low Risk", "Medium Risk", "High Risk"]
    assert "recommendation" in result


def test_predict_customer_dataframe(sample_customer_dict):
    """Verifies prediction function with a DataFrame input."""
    df_sample = pd.DataFrame([sample_customer_dict])
    result = predict_customer(df_sample)
    assert isinstance(result, dict)
    assert 0.0 <= result["probability"] <= 1.0


def test_invalid_input_type():
    """Verifies that non-dict and non-DataFrame inputs raise TypeError."""
    with pytest.raises(TypeError):
        predict_customer("invalid string input")

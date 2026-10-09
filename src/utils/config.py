"""
Central Configuration for Customer Churn Prediction Project.
Defines project directory paths, random seed, target variable, and column groupings.
"""

from pathlib import Path

# Base Paths (Relative to project root)
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Data Paths
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "customer_churn.csv"
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed" / "cleaned_customer_churn.csv"

# Model Paths
MODEL_DIR = BASE_DIR / "models"
CHURN_MODEL_PATH = MODEL_DIR / "churn_model.pkl"
PREPROCESSING_PIPELINE_PATH = MODEL_DIR / "preprocessing_pipeline.pkl"
CHURN_PIPELINE_PATH = MODEL_DIR / "churn_pipeline.joblib"
MODEL_METADATA_PATH = MODEL_DIR / "model_metadata.json"

# Report Paths
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODEL_COMPARISON_PATH = REPORTS_DIR / "model_comparison.csv"
EVALUATION_REPORT_PATH = REPORTS_DIR / "evaluation_report.txt"
FEATURE_IMPORTANCE_PATH = REPORTS_DIR / "feature_importance.csv"

# Global Constants
RANDOM_STATE = 42
TARGET_COL = "churn"

# Feature Column Groups
NUMERICAL_COLS = [
    "age",
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "support_calls",
    "late_payments",
    "usage_hours",
    "num_devices",
    "satisfaction_score",
    "discount",
    "complaints",
    "last_login_days"
]

CATEGORICAL_COLS = [
    "gender",
    "contract_type",
    "payment_method",
    "internet_service"
]

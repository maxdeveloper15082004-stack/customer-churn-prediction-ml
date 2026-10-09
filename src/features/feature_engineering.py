"""
Feature Engineering Module for Customer Churn Prediction.
Creates domain-specific features and provides a Scikit-Learn compliant
FeatureEngineerTransformer for seamless integration into training and serving pipelines.
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers domain-specific features from raw/cleaned customer DataFrame:
    1. average_monthly_spend: total_charges / max(tenure_months, 1)
    2. support_call_rate: support_calls / (tenure_months + 1)
    3. usage_per_device: usage_hours / (num_devices + 1)
    4. high_risk_flag: binary flag for high support calls & late payments
    5. tenure_group: categorical binning of tenure
    6. engagement_score: (usage_hours * satisfaction_score) / (complaints + 1)
    """
    df = df.copy()

    # 1. Average Monthly Spend Ratio
    tenure_safe = np.maximum(df["tenure_months"].values, 1)
    df["average_monthly_spend"] = df["total_charges"] / tenure_safe

    # 2. Support Call Frequency Rate per Month
    df["support_call_rate"] = df["support_calls"] / (df["tenure_months"] + 1)

    # 3. Usage Hours per Device
    df["usage_per_device"] = df["usage_hours"] / (df["num_devices"] + 1)

    # 4. High Risk Behavioral Flag
    df["high_risk_flag"] = (
        (df["support_calls"] >= 3) & (df["late_payments"] >= 2)
    ).astype(int)

    # 5. Engagement Score
    df["engagement_score"] = (df["usage_hours"] * df["satisfaction_score"]) / (df["complaints"] + 1)

    # 6. Tenure Group Categorical Binning
    bins = [-1, 12, 24, 48, 72, 100]
    labels = ["0-12_months", "13-24_months", "25-48_months", "49-72_months", "72+_months"]
    df["tenure_group"] = pd.cut(df["tenure_months"], bins=bins, labels=labels).astype(str)

    return df


class FeatureEngineerTransformer(BaseEstimator, TransformerMixin):
    """
    Scikit-Learn compatible transformer for feature engineering.
    Can be used inside Pipeline before ColumnTransformer.
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return create_features(X)
        else:
            raise TypeError("FeatureEngineerTransformer expects a pandas DataFrame as input.")


ENGINEERED_NUMERICAL_COLS = [
    "average_monthly_spend",
    "support_call_rate",
    "usage_per_device",
    "high_risk_flag",
    "engagement_score"
]

ENGINEERED_CATEGORICAL_COLS = [
    "tenure_group"
]

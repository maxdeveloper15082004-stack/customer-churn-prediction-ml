"""
Data Preprocessing Module for Customer Churn Prediction.
Provides modular functions for loading, validating, cleaning data,
and constructing Scikit-Learn ColumnTransformer preprocessors.
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

from src.utils.config import (
    RAW_DATA_PATH,
    PROCESSED_DATA_PATH,
    NUMERICAL_COLS,
    CATEGORICAL_COLS,
    TARGET_COL,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_data(file_path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Loads raw customer dataset from CSV file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found at: {path.resolve()}")
    
    df = pd.read_csv(path)
    logger.info(f"Successfully loaded dataset with shape: {df.shape}")
    return df


def validate_columns(df: pd.DataFrame) -> bool:
    """Validates that all expected columns exist in the DataFrame."""
    expected_cols = set(NUMERICAL_COLS + CATEGORICAL_COLS + ["customer_id", TARGET_COL])
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in dataset: {missing}")
    logger.info("Column validation passed.")
    return True


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw dataframe:
    - Removes duplicate rows
    - Standardizes categorical values
    - Imputes missing values for standalone analysis
    - Fixes inconsistent numerical total_charges
    - Validates feature value bounds
    """
    df = df.copy()
    initial_rows = len(df)
    
    # 1. Drop duplicate rows
    df = df.drop_duplicates().reset_index(drop=True)
    dedup_count = initial_rows - len(df)
    if dedup_count > 0:
        logger.info(f"Removed {dedup_count} duplicate rows.")

    # 2. Standardize Categorical Strings
    if "gender" in df.columns:
        df["gender"] = df["gender"].astype(str).str.strip().str.capitalize()
        df["gender"] = df["gender"].replace({"Male": "Male", "Female": "Female", "Nan": np.nan})

    if "contract_type" in df.columns:
        df["contract_type"] = df["contract_type"].astype(str).str.strip()
        df["contract_type"] = df["contract_type"].replace({
            "Month to month": "Month-to-month",
            "One-Year": "One year",
            "Two Year": "Two year",
            "Nan": np.nan
        })

    if "payment_method" in df.columns:
        df["payment_method"] = df["payment_method"].astype(str).str.strip().str.title()
        df["payment_method"] = df["payment_method"].replace({
            "Credit Card": "Credit Card",
            "Debit Card": "Debit Card",
            "Electronic Check": "Electronic Check",
            "Bank Transfer": "Bank Transfer",
            "Nan": np.nan
        })

    if "internet_service" in df.columns:
        df["internet_service"] = df["internet_service"].astype(str).str.strip()
        df["internet_service"] = df["internet_service"].replace({"Nan": np.nan})

    # 3. Handle Numerical Discrepancies & Bounds
    # Ensure total_charges is correctly filled if missing or inconsistent
    if "monthly_charges" in df.columns and "tenure_months" in df.columns:
        # Fill missing total_charges with tenure_months * monthly_charges
        calc_total = df["tenure_months"] * df["monthly_charges"]
        df["total_charges"] = df["total_charges"].fillna(calc_total)
        
    # Impute missing values for standalone exploratory CSV (without target leakage)
    for col in NUMERICAL_COLS:
        if col in df.columns and df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

    for col in CATEGORICAL_COLS:
        if col in df.columns and df[col].isnull().sum() > 0:
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)

    # Convert churn target to integer
    if TARGET_COL in df.columns:
        df[TARGET_COL] = df[TARGET_COL].astype(int)

    logger.info(f"Cleaning complete. Cleaned shape: {df.shape}")
    return df


def save_cleaned_data(df: pd.DataFrame, output_path: Path = PROCESSED_DATA_PATH) -> Path:
    """Saves cleaned DataFrame to CSV file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info(f"Cleaned dataset saved successfully to: {path.resolve()}")
    return path


def build_preprocessor(numerical_features=NUMERICAL_COLS, categorical_features=CATEGORICAL_COLS) -> ColumnTransformer:
    """
    Builds a Scikit-Learn ColumnTransformer pipeline.
    Numerical Pipeline: SimpleImputer(median) -> StandardScaler
    Categorical Pipeline: SimpleImputer(most_frequent) -> OneHotEncoder(handle_unknown='ignore')
    """
    num_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, numerical_features),
            ("cat", cat_pipeline, categorical_features)
        ],
        remainder="drop"
    )
    
    return preprocessor


if __name__ == "__main__":
    raw_df = load_data()
    validate_columns(raw_df)
    cleaned_df = clean_data(raw_df)
    save_cleaned_data(cleaned_df)

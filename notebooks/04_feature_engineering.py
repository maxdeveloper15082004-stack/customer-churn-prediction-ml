"""
Notebook 04: Feature Engineering
Applies domain transformation logic to create new features and validates their correlation with churn.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.utils.config import PROCESSED_DATA_PATH, TARGET_COL
from src.features.feature_engineering import create_features, ENGINEERED_NUMERICAL_COLS


def main():
    print("=== STEP 4: FEATURE ENGINEERING ===")
    df = pd.read_csv(PROCESSED_DATA_PATH)
    df_fe = create_features(df)
    
    print(f"Original features: {df.shape[1] - 1}, Engineered features shape: {df_fe.shape[1] - 1}")
    print("\nEngineered Feature Correlation with Churn Target:")
    num_fe = ENGINEERED_NUMERICAL_COLS + [TARGET_COL]
    print(df_fe[num_fe].corr()[TARGET_COL].sort_values(ascending=False))


if __name__ == "__main__":
    main()

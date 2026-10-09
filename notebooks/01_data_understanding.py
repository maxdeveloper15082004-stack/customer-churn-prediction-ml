"""
Notebook 01: Data Understanding & Initial Inspection
Inspects shape, columns, null values, data types, duplicates, and target distribution.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.utils.config import RAW_DATA_PATH, TARGET_COL


def main():
    print("=== STEP 1: DATA UNDERSTANDING ===")
    df = pd.read_csv(RAW_DATA_PATH)
    
    print(f"\n1. Raw Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    
    print("\n2. Data Types & Missing Values Summary:")
    info_df = pd.DataFrame({
        "DataType": df.dtypes,
        "NullCount": df.isnull().sum(),
        "NullPercentage": (df.isnull().sum() / len(df)) * 100
    })
    print(info_df[info_df["NullCount"] > 0])

    print(f"\n3. Duplicate Records Count: {df.duplicated().sum()}")
    
    print(f"\n4. Target Column ('{TARGET_COL}') Distribution:")
    print(df[TARGET_COL].value_counts(dropna=False, normalize=True))


if __name__ == "__main__":
    main()

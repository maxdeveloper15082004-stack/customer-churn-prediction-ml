"""
Notebook 02: Data Cleaning & Preprocessing Execution
Cleans raw customer data and saves output to data/processed/cleaned_customer_churn.csv.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.preprocessing import load_data, validate_columns, clean_data, save_cleaned_data


def main():
    print("=== STEP 2: DATA CLEANING ===")
    df_raw = load_data()
    validate_columns(df_raw)
    df_clean = clean_data(df_raw)
    save_cleaned_data(df_clean)
    print("Data cleaning successfully completed and saved!")


if __name__ == "__main__":
    main()

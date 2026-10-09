"""
Notebook 08: Hyperparameter Tuning Summary
Displays hyperparameter tuning results and candidate comparison matrix.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.utils.config import MODEL_COMPARISON_PATH, MODEL_METADATA_PATH


def main():
    print("=== STEP 8: HYPERPARAMETER TUNING & MODEL COMPARISON SUMMARY ===")
    if MODEL_COMPARISON_PATH.exists():
        comp_df = pd.read_csv(MODEL_COMPARISON_PATH)
        print("\nCandidate Cross-Validation Rankings:")
        print(comp_df.to_string(index=False))
    
    if MODEL_METADATA_PATH.exists():
        with open(MODEL_METADATA_PATH, "r") as f:
            print("\nFinal Tuned Model Metadata:")
            print(f.read())


if __name__ == "__main__":
    main()

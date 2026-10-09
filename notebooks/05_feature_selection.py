"""
Notebook 05: Feature Selection
Inspects feature importances, correlations, and tree-based importances.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.utils.config import FEATURE_IMPORTANCE_PATH


def main():
    print("=== STEP 5: FEATURE SELECTION & IMPORTANCE ANALYSIS ===")
    if FEATURE_IMPORTANCE_PATH.exists():
        fi_df = pd.read_csv(FEATURE_IMPORTANCE_PATH)
        print("\nTop 15 Most Important Features:")
        print(fi_df.head(15).to_string(index=False))
    else:
        print("Feature importance report not yet generated. Run src.models.train first.")


if __name__ == "__main__":
    main()

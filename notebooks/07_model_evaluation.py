"""
Notebook 07: Model Evaluation
Evaluates final model performance on test set and displays reports.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.utils.config import EVALUATION_REPORT_PATH
from src.models.evaluate import evaluate_saved_model


def main():
    print("=== STEP 7: MODEL EVALUATION ===")
    evaluate_saved_model()
    if EVALUATION_REPORT_PATH.exists():
        with open(EVALUATION_REPORT_PATH, "r") as f:
            print("\n" + f.read())


if __name__ == "__main__":
    main()

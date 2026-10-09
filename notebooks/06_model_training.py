"""
Notebook 06: Model Training Execution
Triggers model training workflow and candidate comparisons.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.models.train import train_and_evaluate_system

if __name__ == "__main__":
    print("=== STEP 6: MODEL TRAINING ===")
    train_and_evaluate_system()

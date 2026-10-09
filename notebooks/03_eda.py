"""
Exploratory Data Analysis (EDA) Script for Customer Churn Prediction.
Generates univariate, bivariate visualizations and correlation heatmaps,
saving figures to reports/figures/.
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.config import PROCESSED_DATA_PATH, FIGURES_DIR, NUMERICAL_COLS, TARGET_COL

# Configure Matplotlib style
plt.style.use("ggplot")
sns.set_palette("Set2")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def run_eda():
    print("Loading cleaned dataset for EDA...")
    df = pd.read_csv(PROCESSED_DATA_PATH)
    print(f"Cleaned dataset shape: {df.shape}")

    # 1. Target Distribution
    plt.figure(figsize=(6, 4))
    ax = sns.countplot(data=df, x=TARGET_COL, hue=TARGET_COL, palette=["#2ecc71", "#e74c3c"], legend=False)
    plt.title("Target Distribution (0 = Stayed, 1 = Churned)", fontsize=12, fontweight="bold")
    plt.xlabel("Churn Status")
    plt.ylabel("Customer Count")
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                    ha="center", va="center", color="white", fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "target_distribution.png", dpi=300)
    plt.close()

    # 2. Contract Type vs Churn
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x="contract_type", hue=TARGET_COL, palette=["#3498db", "#e74c3c"])
    plt.title("Churn Rate by Contract Type", fontsize=12, fontweight="bold")
    plt.xlabel("Contract Type")
    plt.ylabel("Count")
    plt.legend(title="Churn", labels=["Stayed", "Churned"])
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "contract_type_vs_churn.png", dpi=300)
    plt.close()

    # 3. Support Calls vs Churn
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x="support_calls", y=TARGET_COL, errorbar=None, color="#e74c3c")
    plt.title("Churn Probability by Number of Support Calls", fontsize=12, fontweight="bold")
    plt.xlabel("Support Calls")
    plt.ylabel("Churn Rate")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "support_calls_vs_churn.png", dpi=300)
    plt.close()

    # 4. Tenure vs Churn Boxplot
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x=TARGET_COL, y="tenure_months", hue=TARGET_COL, palette=["#2ecc71", "#e74c3c"], legend=False)
    plt.title("Tenure Distribution by Churn Status", fontsize=12, fontweight="bold")
    plt.xlabel("Churn Status (0 = Stayed, 1 = Churned)")
    plt.ylabel("Tenure (Months)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "tenure_vs_churn.png", dpi=300)
    plt.close()

    # 5. Correlation Heatmap
    plt.figure(figsize=(10, 8))
    num_cols_with_target = NUMERICAL_COLS + [TARGET_COL]
    corr = df[num_cols_with_target].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
    plt.title("Numerical Features Correlation Heatmap", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "correlation_heatmap.png", dpi=300)
    plt.close()

    print(f"EDA visual charts successfully saved to: {FIGURES_DIR.resolve()}")


if __name__ == "__main__":
    run_eda()

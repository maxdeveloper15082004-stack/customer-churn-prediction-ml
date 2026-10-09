# 🔮 Customer Churn Prediction & Retention Analytics

> **End-to-End Machine Learning System** for predicting customer churn risk, explaining predictive key drivers, and delivering real-time retention recommendations via an interactive Streamlit web application.

---

## 📌 1. Project Overview & Business Problem

In subscription-based industries (telecom, SaaS, digital services), acquiring new customers is **5x to 25x more expensive** than retaining existing ones. Customer churn directly impacts recurring revenue and long-term business growth.

### Business Objective
Build a production-grade machine learning system that:
1. Identifies customers likely to churn before they cancel their subscriptions.
2. Calculates exact churn probability and assigns actionable Risk Levels (**Low**, **Medium**, **High**).
3. Highlights key behavioral drivers behind customer attrition to empower retention outreach teams.
4. Prevents **Training-Serving Skew** by encapsulating preprocessing and feature engineering into a single serialized Scikit-Learn `Pipeline`.

---

## 📊 2. Dataset Disclosure & Schema

> **Disclosure**: The dataset used in this project is synthetically generated for educational and portfolio demonstration purposes. It contains realistic statistical patterns, missing values, duplicates, and feature correlations.

- **File Location**: `data/raw/customer_churn.csv`
- **Total Records**: 10,020 raw observations (10,000 cleaned)
- **Target Variable**: `churn` (`0 = Stayed`, `1 = Churned`)
- **Target Distribution**: `53.7% Stayed` (5,373), `46.3% Churned` (4,627)

### Feature Schema
| Feature | Type | Description |
| :--- | :--- | :--- |
| `customer_id` | Identifier | Unique customer tracking ID (dropped from predictive features) |
| `gender` | Categorical | Customer gender (`Male`, `Female`) |
| `age` | Numerical | Customer age in years |
| `tenure_months` | Numerical | Subscription duration in months |
| `contract_type` | Categorical | Billing contract (`Month-to-month`, `One year`, `Two year`) |
| `monthly_charges` | Numerical | Current recurring monthly bill ($) |
| `total_charges` | Numerical | Cumulative charges paid ($) |
| `payment_method` | Categorical | Payment mode (`Electronic Check`, `Credit Card`, `Debit Card`, `Bank Transfer`) |
| `internet_service` | Categorical | Connection type (`Fiber optic`, `DSL`, `Cable`, `No internet`) |
| `support_calls` | Numerical | Customer service calls logged |
| `late_payments` | Numerical | Number of delayed payments |
| `usage_hours` | Numerical | Monthly service engagement hours |
| `num_devices` | Numerical | Number of active registered devices |
| `satisfaction_score` | Numerical | Customer feedback score (1 to 10 scale) |
| `discount` | Numerical | Monthly promotional discount applied ($) |
| `complaints` | Numerical | Formal complaints registered |
| `last_login_days` | Numerical | Days elapsed since last account login |

---

## 🏗️ 3. Machine Learning System Architecture

```text
Raw Dataset (data/raw/customer_churn.csv)
                     │
                     ▼
Data Cleaning & Validation (src/data/preprocessing.py)
                     │
                     ▼
Feature Engineering (src/features/feature_engineering.py)
  ├── average_monthly_spend
  ├── support_call_rate
  ├── usage_per_device
  ├── high_risk_flag
  └── engagement_score
                     │
                     ▼
Stratified Train/Test Split (80/20)
                     │
                     ▼
Scikit-Learn ColumnTransformer Pipeline
  ├── Numerical Imputer (Median) + StandardScaler
  └── Categorical Imputer (Mode) + OneHotEncoder
                     │
                     ▼
Candidate Comparison (5-Fold Stratified Cross-Validation)
  ├── Logistic Regression (Baseline)
  ├── Decision Tree Classifier
  ├── Random Forest Classifier (Best Candidate)
  └── Gradient Boosting Classifier
                     │
                     ▼
GridSearchCV Hyperparameter Tuning (n_estimators=150, max_depth=12)
                     │
                     ▼
Model Serialization & Metadata (models/churn_pipeline.joblib)
                     │
                     ▼
Production Streamlit App (app/app.py) & Test Suite (tests/test_prediction.py)
```

---

## 📈 4. Measured Candidate Comparison & Final Metrics

### A. 5-Fold Stratified Cross-Validation Performance (Training Set)
| Model | CV Accuracy (Mean) | CV Precision | CV Recall | CV F1 Score | CV ROC-AUC (Mean) | CV ROC-AUC (Std) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **71.65%** | **0.6832** | **0.7212** | **0.7016** | **0.7845** | **±0.0074** |
| **Random Forest (Tuned)** | **69.88%** | **0.6719** | **0.6807** | **0.6761** | **0.7649** | **±0.0098** |
| **Gradient Boosting** | 69.98% | 0.6863 | 0.6452 | 0.6650 | 0.7620 | ±0.0091 |
| **Decision Tree** | 66.21% | 0.6301 | 0.6539 | 0.6414 | 0.7105 | ±0.0161 |

---

### B. Final Model Evaluation on Held-Out Test Set (2,000 Samples)
| Evaluation Metric | Value | Business Meaning |
| :--- | :---: | :--- |
| **ROC-AUC** | **0.7747** | High discrimination ability between churners and non-churners |
| **Recall (Sensitivity)** | **68.61%** | Successfully catches ~69 out of 100 potential churners |
| **Precision** | **68.76%** | 68.7% of flagged high-risk customers actually churn |
| **F1 Score** | **0.6869** | Balanced harmonic mean of Precision and Recall |
| **Accuracy** | **71.10%** | Overall correct classification rate |

#### Test Confusion Matrix:
- **True Negatives (TN)**: 788 (Correctly predicted Stays)
- **False Positives (FP)**: 288 (False churn alarms)
- **False Negatives (FN)**: 290 (Missed churners)
- **True Positives (TP)**: 634 (Correctly identified Churners)

---

## 🛠️ 5. Repository Directory Structure

```text
customer-churn-prediction-ml/
│
├── app/
│   └── app.py                           <-- Streamlit interactive web application
│
├── data/
│   ├── raw/
│   │   └── customer_churn.csv           <-- Raw dataset (10,020 rows)
│   └── processed/
│       └── cleaned_customer_churn.csv   <-- Cleaned dataset (10,000 rows)
│
├── notebooks/
│   ├── 01_data_understanding.py
│   ├── 02_data_cleaning.py
│   ├── 03_eda.py
│   ├── 04_feature_engineering.py
│   ├── 05_feature_selection.py
│   ├── 06_model_training.py
│   ├── 07_model_evaluation.py
│   └── 08_model_tuning.py
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── preprocessing.py             <-- Data cleaning & preprocessor builder
│   ├── features/
│   │   ├── __init__.py
│   │   └── feature_engineering.py       <-- Domain feature engineering transformer
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train.py                     <-- Cross-validation, tuning, and training script
│   │   ├── evaluate.py                  <-- Performance evaluation & plot generator
│   │   └── predict.py                   <-- Inference API function
│   └── utils/
│       ├── __init__.py
│       └── config.py                    <-- Central configuration & paths
│
├── models/
│   ├── churn_pipeline.joblib            <-- Serialized end-to-end Sklearn pipeline
│   └── model_metadata.json              <-- Model metadata & metrics JSON
│
├── reports/
│   ├── model_comparison.csv             <-- CV results table
│   ├── evaluation_report.txt            <-- Detailed evaluation report
│   ├── feature_importance.csv           <-- Key feature driver rankings
│   └── figures/                         <-- High-resolution PNG plots
│       ├── target_distribution.png
│       ├── contract_type_vs_churn.png
│       ├── support_calls_vs_churn.png
│       ├── tenure_vs_churn.png
│       ├── correlation_heatmap.png
│       ├── confusion_matrix.png
│       ├── roc_curve.png
│       └── precision_recall_curve.png
│
├── tests/
│   └── test_prediction.py               <-- Pytest unit test suite
│
├── requirements.txt                     <-- Pinpoint dependencies
├── README.md                            <-- Documentation
├── .gitignore                           <-- Git exclusions
└── LICENSE                              <-- MIT License
```

---

## 💻 6. How to Run the Project (Windows PowerShell)

### Step 1: Clone Repository & Navigate to Directory
```powershell
git clone https://github.com/your-username/customer-churn-prediction-ml.git
cd customer-churn-prediction-ml
```

### Step 2: Create & Activate Virtual Environment
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Run Data Cleaning Pipeline
```powershell
python -m src.data.preprocessing
```

### Step 5: Train Models & Run Cross-Validation
```powershell
python -m src.models.train
```

### Step 6: Generate Evaluation Plots & Metrics
```powershell
python -m src.models.evaluate
```

### Step 7: Run Pytest Test Suite
```powershell
python -m pytest -v
```

### Step 8: Launch Streamlit Web Application
```powershell
streamlit run app/app.py
```

---

## 🚀 7. Streamlit Web Application & Deployment Guide

### Live Local Access
After running `streamlit run app/app.py`, open your browser to:
`http://localhost:8501`

### Online Deployment (Streamlit Community Cloud)
1. Push this repository to GitHub.
2. Log into [share.streamlit.io](https://share.streamlit.io/).
3. Click **"New App"** and select repository `customer-churn-prediction-ml`.
4. Set Main file path to: `app/app.py`.
5. Deploy!

---

## 📜 8. License & Author

- **Author**: Machine Learning Engineer Portfolio Project
- **License**: MIT License ([LICENSE](file:///d:/gen%20ai/machine%20learning/projects/customer-churn-prediction-ml/LICENSE))

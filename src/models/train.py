"""
Model Training Pipeline for Customer Churn Prediction.
Trains, evaluates, tunes, and serializes multiple classification models
within Scikit-Learn Pipelines to eliminate training-serving skew.
"""

import sys
import json
import logging
import pandas as pd
import numpy as np

from datetime import datetime
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)

from src.utils.config import (
    PROCESSED_DATA_PATH, CHURN_PIPELINE_PATH, MODEL_METADATA_PATH,
    MODEL_COMPARISON_PATH, EVALUATION_REPORT_PATH, FEATURE_IMPORTANCE_PATH,
    RANDOM_STATE, TARGET_COL, NUMERICAL_COLS, CATEGORICAL_COLS
)
from src.data.preprocessing import build_preprocessor
from src.features.feature_engineering import (
    FeatureEngineerTransformer, ENGINEERED_NUMERICAL_COLS, ENGINEERED_CATEGORICAL_COLS
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logger = logging.getLogger(__name__)


def load_processed_data() -> tuple[pd.DataFrame, pd.Series]:
    """Loads cleaned dataset and splits into X (features) and y (target)."""
    df = pd.read_csv(PROCESSED_DATA_PATH)
    if "customer_id" in df.columns:
        df = df.drop(columns=["customer_id"])

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    logger.info(f"Loaded feature matrix X shape: {X.shape}, target y shape: {y.shape}")
    return X, y


def build_candidate_pipelines() -> dict[str, Pipeline]:
    """Builds full end-to-end pipelines (FeatureEngineering -> Preprocessing -> Classifier) for candidates."""
    all_num_cols = NUMERICAL_COLS + ENGINEERED_NUMERICAL_COLS
    all_cat_cols = CATEGORICAL_COLS + ENGINEERED_CATEGORICAL_COLS

    preprocessor = build_preprocessor(numerical_features=all_num_cols, categorical_features=all_cat_cols)

    candidates = {
        "Logistic Regression": Pipeline([
            ("feature_engineer", FeatureEngineerTransformer()),
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE))
        ]),
        "Decision Tree": Pipeline([
            ("feature_engineer", FeatureEngineerTransformer()),
            ("preprocessor", preprocessor),
            ("classifier", DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=RANDOM_STATE))
        ]),
        "Random Forest": Pipeline([
            ("feature_engineer", FeatureEngineerTransformer()),
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=100, max_depth=10, class_weight="balanced", random_state=RANDOM_STATE))
        ]),
        "Gradient Boosting": Pipeline([
            ("feature_engineer", FeatureEngineerTransformer()),
            ("preprocessor", preprocessor),
            ("classifier", GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=RANDOM_STATE))
        ])
    }
    return candidates


def run_cross_validation(candidates: dict[str, Pipeline], X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
    """Performs Stratified 5-Fold Cross-Validation across candidate models."""
    logger.info("Starting 5-Fold Cross-Validation on training set...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    results = []

    for name, pipeline in candidates.items():
        logger.info(f"Evaluating model fold splits: {name}...")
        cv_scores = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        
        row = {
            "Model": name,
            "CV_Accuracy_Mean": cv_scores["test_accuracy"].mean(),
            "CV_Accuracy_Std": cv_scores["test_accuracy"].std(),
            "CV_Precision_Mean": cv_scores["test_precision"].mean(),
            "CV_Recall_Mean": cv_scores["test_recall"].mean(),
            "CV_F1_Mean": cv_scores["test_f1"].mean(),
            "CV_ROC_AUC_Mean": cv_scores["test_roc_auc"].mean(),
            "CV_ROC_AUC_Std": cv_scores["test_roc_auc"].std(),
        }
        results.append(row)

    comparison_df = pd.DataFrame(results).sort_values(by="CV_ROC_AUC_Mean", ascending=False)
    MODEL_COMPARISON_PATH.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(MODEL_COMPARISON_PATH, index=False)
    logger.info(f"Cross-validation completed. Comparison saved to {MODEL_COMPARISON_PATH.resolve()}")
    return comparison_df


def tune_best_model(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """Tunes hyper-parameters of the best candidate (Random Forest) via GridSearchCV."""
    logger.info("Tuning Random Forest Classifier using GridSearchCV...")
    all_num_cols = NUMERICAL_COLS + ENGINEERED_NUMERICAL_COLS
    all_cat_cols = CATEGORICAL_COLS + ENGINEERED_CATEGORICAL_COLS
    preprocessor = build_preprocessor(numerical_features=all_num_cols, categorical_features=all_cat_cols)

    base_pipeline = Pipeline([
        ("feature_engineer", FeatureEngineerTransformer()),
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=RANDOM_STATE, class_weight="balanced"))
    ])

    param_grid = {
        "classifier__n_estimators": [100, 150],
        "classifier__max_depth": [8, 12],
        "classifier__min_samples_split": [2, 5],
        "classifier__min_samples_leaf": [1, 2]
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    grid_search = GridSearchCV(
        estimator=base_pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)
    logger.info(f"GridSearchCV best ROC-AUC: {grid_search.best_score_:.4f}")
    logger.info(f"Best parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_


def extract_and_save_feature_importance(pipeline: Pipeline, X_train: pd.DataFrame):
    """Extracts feature names and feature importances from the fitted pipeline."""
    feature_engineer = pipeline.named_steps["feature_engineer"]
    X_engineered = feature_engineer.transform(X_train)

    preprocessor = pipeline.named_steps["preprocessor"]
    cat_encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]
    
    all_num_cols = NUMERICAL_COLS + ENGINEERED_NUMERICAL_COLS
    all_cat_cols = CATEGORICAL_COLS + ENGINEERED_CATEGORICAL_COLS

    cat_feature_names = cat_encoder.get_feature_names_out(all_cat_cols).tolist()
    feature_names = all_num_cols + cat_feature_names

    classifier = pipeline.named_steps["classifier"]
    if hasattr(classifier, "feature_importances_"):
        importances = classifier.feature_importances_
    elif hasattr(classifier, "coef_"):
        importances = np.abs(classifier.coef_[0])
    else:
        importances = np.zeros(len(feature_names))

    fi_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    FEATURE_IMPORTANCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fi_df.to_csv(FEATURE_IMPORTANCE_PATH, index=False)
    logger.info(f"Feature importance saved to: {FEATURE_IMPORTANCE_PATH.resolve()}")
    return fi_df


def train_and_evaluate_system():
    """Executes the full training, evaluation, tuning, and serialization pipeline."""
    X, y = load_processed_data()

    # Stratified Train/Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    logger.info(f"Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")

    # 1. Candidate Comparison via 5-Fold CV
    candidates = build_candidate_pipelines()
    comparison_df = run_cross_validation(candidates, X_train, y_train)
    print("\n=== Model Comparison (5-Fold Cross Validation) ===")
    print(comparison_df.to_string(index=False))

    # 2. Hyperparameter Tuning on Best Model
    best_pipeline = tune_best_model(X_train, y_train)

    # 3. Fit Final Pipeline on Full Training Set
    best_pipeline.fit(X_train, y_train)

    # 4. Final Evaluation on Held-Out Test Set
    y_pred = best_pipeline.predict(X_test)
    y_prob = best_pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    print("\n=== Final Model Performance on Held-Out Test Set ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")
    print(f"Confusion Matrix:\n{cm}")

    # 5. Save Evaluation Report
    report_text = f"""==================================================
CUSTOMER CHURN PREDICTION - FINAL EVALUATION REPORT
==================================================
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Random State: {RANDOM_STATE}
Train Samples: {len(X_train)}
Test Samples: {len(X_test)}

HELD-OUT TEST SET METRICS:
--------------------------------------------------
Accuracy:  {acc:.4f}
Precision: {prec:.4f}
Recall:    {rec:.4f}
F1 Score:  {f1:.4f}
ROC-AUC:   {auc:.4f}

CONFUSION MATRIX:
--------------------------------------------------
True Negatives (TN):  {cm[0, 0]} (Correctly predicted Stays)
False Positives (FP): {cm[0, 1]} (False churn alarms)
False Negatives (FN): {cm[1, 0]} (Missed churners - High Cost!)
True Positives (TP):  {cm[1, 1]} (Correctly caught Churners)

CLASSIFICATION REPORT:
--------------------------------------------------
{classification_report(y_test, y_pred, target_names=['Stayed (0)', 'Churned (1)'])}
"""
    EVALUATION_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(EVALUATION_REPORT_PATH, "w") as f:
        f.write(report_text)
    logger.info(f"Saved text evaluation report to: {EVALUATION_REPORT_PATH.resolve()}")

    # 6. Extract Feature Importance
    extract_and_save_feature_importance(best_pipeline, X_train)

    # 7. Serialize Final Model Pipeline & Metadata
    CHURN_PIPELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, CHURN_PIPELINE_PATH)
    logger.info(f"Serialized complete Scikit-Learn pipeline to: {CHURN_PIPELINE_PATH.resolve()}")

    metadata = {
        "model_name": "Random Forest Classifier (Tuned)",
        "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "random_state": RANDOM_STATE,
        "classification_threshold": 0.5,
        "test_metrics": {
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4),
            "roc_auc": round(float(auc), 4)
        }
    }
    with open(MODEL_METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=4)
    logger.info(f"Saved model metadata to: {MODEL_METADATA_PATH.resolve()}")


if __name__ == "__main__":
    train_and_evaluate_system()

"""
Streamlit Web Application for Customer Churn Prediction & Retention Analytics.
Provides interactive UI for customer risk assessment, prediction explanation,
model performance metrics, and retention recommendations.
"""

import sys
import json
import pandas as pd
import streamlit as st
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.models.predict import predict_customer, load_model
from src.utils.config import (
    MODEL_METADATA_PATH, FEATURE_IMPORTANCE_PATH, CHURN_PIPELINE_PATH
)

# 1. Page Configuration
st.set_page_config(
    page_title="Customer Churn Prediction & Retention Analytics",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1e3a8a;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4b5563;
        text-align: center;
        margin-bottom: 2rem;
    }
    .card-high {
        background-color: #fee2e2;
        border-left: 6px solid #ef4444;
        padding: 1.5rem;
        border-radius: 8px;
        margin-top: 1rem;
    }
    .card-medium {
        background-color: #fef3c7;
        border-left: 6px solid #f59e0b;
        padding: 1.5rem;
        border-radius: 8px;
        margin-top: 1rem;
    }
    .card-low {
        background-color: #dcfce7;
        border-left: 6px solid #22c55e;
        padding: 1.5rem;
        border-radius: 8px;
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)


def main():
    st.markdown('<div class="main-header">🔮 Customer Churn Prediction & Retention Analytics</div>', unsafe_allow_dict=True)
    st.markdown('<div class="sub-header">Production ML System for Customer Risk Scoring & Proactive Retention Guidance</div>', unsafe_allow_dict=True)

    # Sidebar: System Metadata & Model Info
    st.sidebar.title("ℹ️ Model Information")
    if MODEL_METADATA_PATH.exists():
        with open(MODEL_METADATA_PATH, "r") as f:
            metadata = json.load(f)
        st.sidebar.subheader("Selected Model")
        st.sidebar.info(metadata.get("model_name", "Random Forest Classifier"))
        
        st.sidebar.subheader("Test Metrics")
        metrics = metadata.get("test_metrics", {})
        col_m1, col_m2 = st.sidebar.columns(2)
        col_m1.metric("ROC-AUC", f"{metrics.get('roc_auc', 0.7747):.4f}")
        col_m2.metric("F1 Score", f"{metrics.get('f1_score', 0.6869):.4f}")
        col_m3, col_m4 = st.sidebar.columns(2)
        col_m3.metric("Recall", f"{metrics.get('recall', 0.6861):.4f}")
        col_m4.metric("Accuracy", f"{metrics.get('accuracy', 0.7110):.4f}")
    else:
        st.sidebar.warning("Model metadata file not found. Ensure models/train.py has run.")

    # Main Tabs
    tab_pred, tab_importance, tab_about = st.tabs(["🎯 Live Churn Prediction", "📊 Feature Importance", "📘 About Project"])

    with tab_pred:
        st.subheader("📋 Enter Customer Account Profile")
        
        with st.form("customer_form"):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("##### 👤 Demographic & Contract")
                gender = st.selectbox("Gender", ["Male", "Female"])
                age = st.number_input("Age (Years)", min_value=18, max_value=100, value=42)
                tenure_months = st.number_input("Tenure (Months)", min_value=0, max_value=120, value=12)
                contract_type = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
                payment_method = st.selectbox("Payment Method", ["Electronic Check", "Bank Transfer", "Credit Card", "Debit Card"])
                internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "Cable", "No internet"])

            with col2:
                st.markdown("##### 💳 Billing & Usage")
                monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=500.0, value=85.5)
                total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=20000.0, value=1026.0)
                discount = st.number_input("Discount Received ($)", min_value=0.0, max_value=100.0, value=5.0)
                usage_hours = st.number_input("Monthly Usage (Hours)", min_value=0.0, max_value=1000.0, value=120.0)
                num_devices = st.number_input("Connected Devices", min_value=1, max_value=20, value=4)

            with col3:
                st.markdown("##### 📞 Support & Engagement")
                support_calls = st.number_input("Support Calls (Last 6 Mos)", min_value=0, max_value=20, value=2)
                late_payments = st.number_input("Late Payments", min_value=0, max_value=10, value=1)
                satisfaction_score = st.slider("Satisfaction Score (1-10)", min_value=1.0, max_value=10.0, value=6.0)
                complaints = st.number_input("Formal Complaints Filed", min_value=0, max_value=10, value=0)
                last_login_days = st.number_input("Days Since Last Login", min_value=0, max_value=180, value=10)

            submitted = st.form_submit_button("🚀 Predict Customer Churn Risk", use_container_width=True)

        if submitted:
            customer_data = {
                "gender": gender,
                "age": age,
                "tenure_months": tenure_months,
                "contract_type": contract_type,
                "monthly_charges": monthly_charges,
                "total_charges": total_charges,
                "payment_method": payment_method,
                "internet_service": internet_service,
                "support_calls": support_calls,
                "late_payments": late_payments,
                "usage_hours": usage_hours,
                "num_devices": num_devices,
                "satisfaction_score": satisfaction_score,
                "discount": discount,
                "complaints": complaints,
                "last_login_days": last_login_days
            }

            try:
                res = predict_customer(customer_data)
                
                st.markdown("---")
                st.subheader("🎯 Prediction & Retention Assessment")

                c1, c2, c3 = st.columns(3)
                c1.metric("Predicted Status", res["prediction_label"])
                c2.metric("Churn Probability", res["probability_pct"])
                c3.metric("Risk Classification", res["risk_level"])

                # Display Risk Card
                if res["risk_level"] == "High Risk":
                    card_style = "card-high"
                    icon = "🚨"
                elif res["risk_level"] == "Medium Risk":
                    card_style = "card-medium"
                    icon = "⚠️"
                else:
                    card_style = "card-low"
                    icon = "✅"

                st.markdown(f"""
                    <div class="{card_style}">
                        <h4>{icon} {res['risk_level']} Assessment</h4>
                        <p><b>Estimated Probability of Churn:</b> {res['probability_pct']}</p>
                        <p><b>Recommended Action:</b> {res['recommendation']}</p>
                    </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error executing prediction: {e}")

    with tab_importance:
        st.subheader("📊 Key Drivers of Customer Churn")
        if FEATURE_IMPORTANCE_PATH.exists():
            fi_df = pd.read_csv(FEATURE_IMPORTANCE_PATH).head(10)
            st.dataframe(fi_df, use_container_width=True)
            st.bar_chart(data=fi_df.set_index("Feature")["Importance"])
        else:
            st.warning("Feature importance report missing.")

    with tab_about:
        st.subheader("📘 Project Architecture & Methodology")
        st.markdown("""
        ### End-to-End Customer Churn Prediction & Retention System
        - **Pipeline Integration**: Implements complete Scikit-Learn `Pipeline` with custom feature engineering (`FeatureEngineerTransformer`) and preprocessor (`ColumnTransformer`).
        - **Leakage Prevention**: All imputation, scaling, and categorical encoding are fitted strictly on training data folds.
        - **Model Selection**: 5-Fold Stratified Cross Validation across Logistic Regression, Decision Trees, Random Forest, and Gradient Boosting.
        """)


if __name__ == "__main__":
    main()

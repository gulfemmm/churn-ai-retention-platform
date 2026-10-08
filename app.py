import time
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from google import genai
import requests
from pdf_generator import create_retention_pdf

# Page Config
st.set_page_config(
    page_title="ChurnAI - Retention & Analytics Platform",
    page_icon="⚡",
    layout="wide"
)

# Title & Header
st.title("⚡ ChurnAI: Enterprise Customer Churn & Retention Platform")
st.markdown("Predict customer churn, analyze risk factors via **SHAP (XAI)**, and generate GenAI-powered retention campaigns.")

# Load Local Model & Columns
@st.cache_resource
def load_assets():
    try:
        model = joblib.load("best_churn_model.pkl")
        columns = joblib.load("model_columns.pkl")
        return model, columns
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        return None, None

model, model_columns = load_assets()

# API Key Arka Planda Sessizce Okunuyor
gemini_api_key = st.secrets.get("GEMINI_API_KEY", "")

# Sidebar Settings
st.sidebar.header("⚙️ Configuration")
api_mode = st.sidebar.toggle("Use FastAPI Backend", value=False, help="Connect to FastAPI backend at http://127.0.0.1:8000")

# Main Navigation
tab1, tab2 = st.tabs(["👤 Single Customer Analysis & PDF Report", "📂 Batch CSV Analysis"])

# --- TAB 1: SINGLE CUSTOMER ---
with tab1:
    st.subheader("Interactive Customer Risk Profiler")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        customer_id = st.text_input("Customer ID", "CUST-8921")
        tenure = st.number_input("Tenure (Months)", min_value=1, max_value=120, value=12)
        satisfaction = st.slider("Satisfaction Score (1-5)", 1, 5, 2)
    with col2:
        support_calls = st.number_input("Support Tickets (Last 6 Months)", 0, 20, 5)
        complain = st.selectbox("Has Active Complaint?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
        cashback = st.number_input("Cashback Amount ($)", 0.0, 500.0, 120.0)
    with col3:
        order_count = st.number_input("Order Count", 1, 100, 3)
        day_since_last_order = st.number_input("Days Since Last Order", 0, 365, 45)

    input_data = {
        "Tenure": tenure,
        "SatisfactionScore": satisfaction,
        "Complain": complain,
        "SupportCalls": support_calls,
        "CashbackAmount": cashback,
        "OrderCount": order_count,
        "DaySinceLastOrder": day_since_last_order
    }

    if st.button("🚀 Analyze Churn Risk"):
        churn_prob = 0.0
        risk_level = "Low Risk"
        
        if api_mode:
            try:
                res = requests.post("http://127.0.0.1:8000/predict", json={"data": input_data})
                if res.status_code == 200:
                    res_data = res.json()
                    churn_prob = res_data["churn_probability"]
                    risk_level = res_data["risk_level"]
                else:
                    st.error("FastAPI error. Check if api.py is running.")
            except Exception as e:
                st.error(f"Failed to connect to FastAPI: {e}")
        else:
            try:
                df_single = pd.DataFrame([input_data]).reindex(columns=model_columns, fill_value=0)
                
                # Model indeksini [0][1] yapıyoruz (Churn / Sınıf 1 Olasılığı)
                churn_prob = float(model.predict_proba(df_single)[0][1])
                
                risk_level = "High Risk" if churn_prob >= 0.6 else ("Medium Risk" if churn_prob >= 0.3 else "Low Risk")
            except Exception as predict_err:
                st.error(f"Prediction Error: {predict_err}")

        st.divider()
        m1, m2 = st.columns(2)
        m1.metric("Predicted Churn Probability", f"{churn_prob * 100:.1f}%")
        m2.metric("Risk Assessment", risk_level, delta="-High Action Needed" if churn_prob >= 0.6 else "Stable")

        retention_text = "N/A"
        if gemini_api_key:
            st.markdown("### 🤖 GenAI Retention Action Plan")
            with st.spinner("Generating personalized retention strategy via Gemini..."):
                client = genai.Client(api_key=gemini_api_key)
                prompt = f"""
                Act as an expert Customer Retention Specialist for an e-commerce platform.
                Customer Profile: {input_data}
                Predicted Churn Probability: {churn_prob * 100:.1f}%

                Provide a concise 3-step action plan to prevent this customer from churning:
                1. Root cause summary.
                2. Personalized offer / incentive.
                3. Recommended communication channel & messaging tone.
                """
                
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        response = client.models.generate_content(
                            model='gemini-flash-latest',
                            contents=prompt
                        )
                        retention_text = response.text
                        st.info(retention_text)
                        break
                    except Exception as e:
                        if "503" in str(e) and attempt < max_retries - 1:
                            time.sleep(2)
                            continue
                        else:
                            st.error(f"Gemini API Error: {e}")
                            break
        else:
            st.warning("Gemini API Key bulunamadı! Lütfen .streamlit/secrets.toml dosyasını kontrol edin.")

        st.divider()
        pdf_file = create_retention_pdf(
            customer_id=customer_id,
            customer_data=input_data,
            churn_prob=churn_prob,
            risk_level=risk_level,
            retention_strategy=retention_text
        )
        
        st.download_button(
            label="📄 Download Executive PDF Report",
            data=pdf_file,
            file_name=f"Retention_Report_{customer_id}.pdf",
            mime="application/pdf"
        )

# --- TAB 2: BATCH CSV ---
with tab2:
    st.subheader("Bulk Batch CSV Processing")
    uploaded_file = st.file_uploader("Upload Customer Dataset (CSV)", type=["csv"])
    if uploaded_file is not None:
        df_batch = pd.read_csv(uploaded_file)
        st.write("Uploaded Sample Data:", df_batch.head(3))
        
        if st.button("Run Batch Prediction"):
            df_encoded = pd.get_dummies(df_batch).reindex(columns=model_columns, fill_value=0)
            # Batch işlemlerinde indeks 1 (Sınıf 1 - Churn Olasılığı) alınıyor
            preds_prob = model.predict_proba(df_encoded)[:, 1]
            df_batch["Churn_Probability"] = preds_prob
            df_batch["Risk_Level"] = ["High Risk" if p >= 0.6 else ("Medium Risk" if p >= 0.3 else "Low Risk") for p in preds_prob]
            
            st.success("Batch Prediction Complete!")
            st.dataframe(df_batch)
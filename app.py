import time
import io
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from google import genai
import requests
from pdf_generator import create_retention_pdf

# PDF Oluşturma İçin ReportLab Kütüphaneleri (Yatay Düzen eklendi)
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

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

# Sayfaya Tam Sığan Yatay (Landscape) PDF Oluşturucu
def create_bulk_high_risk_pdf(high_risk_df, display_cols):
    buffer = io.BytesIO()
    # Sayfayı Yatay (Landscape) Olarak Ayarlıyoruz
    doc = SimpleDocTemplate(
        buffer, 
        pagesize=landscape(letter), 
        rightMargin=20, 
        leftMargin=20, 
        topMargin=20, 
        bottomMargin=20
    )
    story = []
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], fontSize=16, textColor=colors.HexColor('#1E3A8A'), spaceAfter=8)
    normal_style = styles['Normal']
    cell_style = ParagraphStyle('CellStyle', parent=styles['Normal'], fontSize=7.5, leading=9)
    header_style = ParagraphStyle('HeaderStyle', parent=styles['Normal'], fontSize=8, leading=10, fontName='Helvetica-Bold', textColor=colors.white)

    # Başlık ve Özet Bilgi
    story.append(Paragraph("⚡ ChurnAI - High Risk Customer Retention Report", title_style))
    story.append(Paragraph(f"<b>Total High Risk Customers Flagged:</b> {len(high_risk_df)}", normal_style))
    story.append(Spacer(1, 10))

    # PDF Tablosu İçin Veri Hazırlama
    pdf_table_data = []
    
    # Header Satırı
    header_row = [Paragraph(str(col), header_style) for col in display_cols]
    pdf_table_data.append(header_row)

    # Veri Satırları
    for _, row in high_risk_df[display_cols].iterrows():
        row_data = []
        for col in display_cols:
            val_str = str(row[col])
            row_data.append(Paragraph(val_str, cell_style))
        pdf_table_data.append(row_data)

    # Sütun Genişliklerini Dinamik Hesaplama (Yatay Sayfa Genişliği ~750pt)
    num_cols = len(display_cols)
    if "AI_Action_Plan" in display_cols:
        # AI Stratejisi için daha geniş sütun, diğerleri için dar sütun
        other_cols_width = 450 / (num_cols - 1) if num_cols > 1 else 100
        col_widths = [300 if col == "AI_Action_Plan" else other_cols_width for col in display_cols]
    else:
        col_widths = [750 / num_cols] * num_cols

    table = Table(pdf_table_data, colWidths=col_widths, repeatRows=1)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1E3A8A')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8FAFC')]),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
    ]))
    
    story.append(table)
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

# Main Navigation
tab1, tab2 = st.tabs(["👤 Single Customer Analysis & PDF Report", "📂 Bulk Batch Analysis & AI Action Plan"])

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
                churn_prob = float(model.predict_proba(df_single)[0][1])
                risk_level = "High Risk" if churn_prob >= 0.6 else ("Medium Risk" if churn_prob >= 0.3 else "Low Risk")
            except Exception as predict_err:
                st.error(f"Prediction Error: {predict_err}")

        churn_prob100 = churn_prob * 100

        st.divider()
        m1, m2 = st.columns(2)
        m1.metric("Predicted Churn Probability", f"{churn_prob100:.1f}%")
        m2.metric("Risk Assessment", risk_level, delta="-High Action Needed" if churn_prob >= 0.6 else "Stable")

        retention_text = "N/A"
        if gemini_api_key:
            st.markdown("### 🤖 GenAI Retention Action Plan")
            with st.spinner("Generating personalized retention strategy via Gemini..."):
                try:
                    client = genai.Client(api_key=gemini_api_key)
                    prompt = f"""
                    Act as an expert Customer Retention Specialist for an e-commerce platform.
                    Customer Profile: {input_data}
                    Predicted Churn Probability: {churn_prob100:.1f}%

                    Provide a concise 3-step action plan to prevent this customer from churning:
                    1. Root cause summary.
                    2. Personalized offer / incentive.
                    3. Recommended communication channel & messaging tone.
                    """
                    
                    candidate_models = ['gemini-3.5-flash', 'gemini-3.5-flash-lite']
                    success = False
                    last_err = ""
                    
                    for model_name in candidate_models:
                        try:
                            response = client.models.generate_content(
                                model=model_name,
                                contents=prompt
                            )
                            retention_text = response.text
                            st.info(retention_text)
                            success = True
                            break
                        except Exception as e:
                            last_err = str(e)
                            time.sleep(1)
                            continue
                            
                    if not success:
                        st.error(f"Gemini API Error: {last_err}")
                except Exception as e:
                    st.error(f"Gemini Client Error: {e}")
        else:
            st.warning("⚠️ Gemini API Key bulunamadı! `.streamlit/secrets.toml` dosyasını kontrol edin.")

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


# --- TAB 2: BULK BATCH ANALYSIS & AI ACTION PLAN ---
with tab2:
    st.subheader("Bulk Batch CSV Processing & High Risk Filtering")
    uploaded_file = st.file_uploader("Upload Customer Dataset (CSV)", type=["csv"])
    
    if uploaded_file is not None:
        df_batch = pd.read_csv(uploaded_file)
        st.write("📋 Uploaded Dataset Sample:", df_batch.head(3))
        
        risk_threshold = st.slider("High Risk Threshold (%)", 50, 90, 60) / 100.0
        
        if st.button("🚀 Run Batch Prediction & Filter High Risk"):
            with st.spinner("Classifying all customers in dataset..."):
                df_encoded = pd.get_dummies(df_batch).reindex(columns=model_columns, fill_value=0)
                preds_prob = model.predict_proba(df_encoded)[:, 1]
                
                df_batch["Churn_Probability (%)"] = (preds_prob * 100).round(1)
                df_batch["Risk_Level"] = ["High Risk" if p >= risk_threshold else ("Medium Risk" if p >= 0.3 else "Low Risk") for p in preds_prob]
                
                high_risk_df = df_batch[df_batch["Risk_Level"] == "High Risk"].sort_values(by="Churn_Probability (%)", ascending=False).copy()
                
                st.success(f"✅ Batch Analysis Complete! Out of {len(df_batch)} customers, **{len(high_risk_df)} are flagged as HIGH RISK**.")
                
                st.divider()
                st.subheader("⚠️ High-Risk Customer List")
                
                id_col = None
                for col_name in ["CustomerID", "Customer_ID", "customer_id", "ID"]:
                    if col_name in high_risk_df.columns:
                        id_col = col_name
                        break
                
                remaining_cols = [c for c in high_risk_df.columns if c not in [id_col, "Churn_Probability (%)", "Risk_Level"]]
                
                if id_col:
                    cols = [id_col, "Churn_Probability (%)", "Risk_Level"] + remaining_cols
                else:
                    cols = ["Churn_Probability (%)", "Risk_Level"] + remaining_cols
                
                cols = list(dict.fromkeys(cols))
                
                # Her müşteri için özel AI önerisi oluşturma
                ai_strategies = []
                if len(high_risk_df) > 0 and gemini_api_key:
                    st.divider()
                    st.subheader("🤖 Generating Individual AI Retention Strategies...")
                    client = genai.Client(api_key=gemini_api_key)
                    
                    progress_bar = st.progress(0)
                    total_cust = len(high_risk_df)
                    
                    for idx, (_, row) in enumerate(high_risk_df.iterrows()):
                        row_dict = row.to_dict()
                        prompt = f"""
                        Act as an E-commerce Retention Specialist. 
                        Customer Data: {row_dict}
                        
                        In maximum 1-2 short sentences without bullet points or quotation marks, provide a direct actionable retention offer and communication approach for this specific customer.
                        """
                        
                        strat = "Special discount & VIP call recommended."
                        candidate_models = ['gemini-3.5-flash', 'gemini-3.5-flash-lite']
                        
                        for model_name in candidate_models:
                            try:
                                response = client.models.generate_content(
                                    model=model_name,
                                    contents=prompt
                                )
                                strat = response.text.replace('"', '').replace("'", "").strip()
                                break
                            except Exception:
                                continue
                        
                        ai_strategies.append(strat)
                        progress_bar.progress((idx + 1) / total_cust)
                        
                elif not gemini_api_key:
                    st.warning("⚠️ Gemini API Key bulunamadı! Varsayılan stratejiler eklendi.")
                    ai_strategies = ["Standard VIP Retention Outreach"] * len(high_risk_df)
                else:
                    ai_strategies = []

                if ai_strategies:
                    high_risk_df["AI_Action_Plan"] = ai_strategies
                    display_cols = cols + ["AI_Action_Plan"]
                else:
                    display_cols = cols

                # Tabloyu Ekranda Göster
                st.dataframe(high_risk_df[display_cols], use_container_width=True)
                
                # PDF Olarak İndirme Butonu
                st.divider()
                pdf_bytes = create_bulk_high_risk_pdf(high_risk_df, display_cols)
                
                st.download_button(
                    label="📄 Download Bulk High-Risk Retention PDF Report",
                    data=pdf_bytes,
                    file_name="High_Risk_Customers_AI_Retention_Report.pdf",
                    mime="application/pdf"
                )
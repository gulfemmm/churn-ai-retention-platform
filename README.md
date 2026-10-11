# ⚡ ChurnAI: AI-Powered Customer Churn Prediction & Retention Platform

**ChurnAI** is an end-to-end B2B Decision Support System designed for e-commerce platforms. It predicts customer churn risks in advance, explains the underlying root causes using **Explainable AI (SHAP)**, and provides tailored, actionable retention strategies via **Google Gemini LLM (GenAI)** integration.

---

## 🚀 Key Features

- **End-to-End Machine Learning Pipeline:** Comprehensive data preprocessing, Data Leakage prevention, missing value imputation, and One-Hot Encoding.
- **Model Benchmarking & High Performance:** Comparative analysis between Logistic Regression, Random Forest, and **XGBoost**.
  - 🏆 **Champion Model (XGBoost):** Achieved an impressive **0.9696 ROC-AUC Score**.
- **Explainable AI (XAI with SHAP):** Breaks the "black box" nature of ML models by visualizing key churn drivers (e.g., support ticket volume, satisfaction score, return frequency, and recency).
- **Google Gemini-Powered Retention Agent:** Generates real-time, personalized retention offers and engagement strategies for high-risk customers.
- **Automated Executive PDF Reports:** Exports single-customer churn analyses and AI retention plans into formatted PDF summaries.
- **Interactive Streamlit Web Dashboard:**
  - 📂 **Batch CSV Processing:** Upload bulk customer records to predict churn probability at scale and download analyzed reports.
  - 👤 **Single-Customer Analysis & GenAI Chatbot:** Simulate customer scenarios in real-time and get instant AI-driven retention advice.

---

## 🛠️ Tech Stack & Dependencies

- **Core Language:** Python 3.12
- **Backend API:** FastAPI, Uvicorn, Pydantic, Requests
- **Frontend & Web Dashboard:** Streamlit
- **Machine Learning & XAI:** Scikit-Learn, XGBoost, SHAP
- **Generative AI:** Google GenAI SDK (Gemini Flash API)
- **Document Generation:** ReportLab
- **Data Manipulation & Viz:** Pandas, NumPy, Seaborn, Matplotlib

---

## 📂 Repository Structure

```text
├── E-commerce_Customer_Churn_2026.csv   # Anonymized 50K Customer Dataset
├── 1_veri_temizleme_ve_eda.ipynb        # Data Cleaning & Exploratory Data Analysis
├── 2_model_egitimi_ve_shap.ipynb        # Model Training, Evaluation & SHAP Analysis
├── train_model.py                       # Standalone Model Training & Pipeline Script
├── best_churn_model.pkl                 # Trained Champion Model (XGBoost)
├── model_columns.pkl                    # Feature Column Structure
├── api.py                               # FastAPI Backend REST API Service
├── app.py                               # Streamlit Web Application Dashboard
├── pdf_generator.py                     # ReportLab PDF Report Generator Module
├── requirements.txt                     # Project Dependencies
└── README.md 
```
## 💻Installation & Quick Start 

1. Clone the Repository
```bash
git clone https://github.com/gulfemmm/churn-ai-retention-platform.git
cd churn-ai-retention-platform
```
2. Install Dependencies
```bash
pip install -r requirements.txt
```
3. Train / Regenerate Model (Optional)
python train_model.py
4. Launch the Backend API (FastAPI - Optional)
```bash
uvicorn api:app --reload --port 8000
```
5. Launch the Streamlit Dashboard
```bash
streamlit run app.py
👤 Author
Gülfem - GitHub Profile

📜 License
This project is licensed under the CC0 1.0 Universal (Public Domain) license.
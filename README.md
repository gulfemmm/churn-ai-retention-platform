# ⚡ ChurnAI: AI-Powered Customer Churn Prediction & Retention Platform

**ChurnAI** is an end-to-end B2B Decision Support System designed for e-commerce platforms. It predicts customer churn risks in advance, explains the underlying root causes using **Explainable AI (SHAP)**, and provides tailored, actionable retention strategies via **Google Gemini LLM (GenAI)** integration.

---

## 🚀 Key Features

- **End-to-End Machine Learning Pipeline:** Comprehensive data preprocessing, Data Leakage prevention, missing value imputation, and One-Hot Encoding.
- **Model Benchmarking & High Performance:** Comparative analysis between Logistic Regression, Random Forest, and **XGBoost**.
  - 🏆 **Champion Model (XGBoost):** Achieved an impressive **0.9696 ROC-AUC Score**.
- **Explainable AI (XAI with SHAP):** Breaks the "black box" nature of ML models by visualizing key churn drivers (e.g., support ticket volume, satisfaction score, return frequency, and recency).
- **Google Gemini 3.5 Integration:** Generates real-time, highly personalized retention offers and outreach strategies using `gemini-3.5-flash` for both individual and batch customer profiles.
- **Automated Executive PDF Reports:** 
  - **Single Customer:** Generates detailed single-customer retention profile reports.
  - **Bulk High-Risk Batch Analysis:** Exports custom landscape PDF reports containing filtered high-risk customer lists along with individual GenAI action plans.
- **Interactive Streamlit Web Dashboard:**
  - 📂 **Bulk Batch Processing:** Upload CSV datasets, customize risk thresholds, dynamically generate AI retention strategies for high-risk users, and download landscape executive PDF reports.
  - 👤 **Single-Customer Analysis:** Simulate real-time scenarios, inspect churn risk probability metrics, and get instant GenAI retention advice.
  - ⚙️ **Dual Architecture Mode:** Switch seamlessly between local monolithic prediction execution and a decoupled FastAPI backend REST service via UI toggle.

---

## 🛠️ Tech Stack & Dependencies

- **Core Language:** Python 3.12
- **Backend API:** FastAPI, Uvicorn, Pydantic, Requests
- **Frontend & Web Dashboard:** Streamlit
- **Machine Learning & XAI:** Scikit-Learn, XGBoost, SHAP
- **Generative AI:** Google GenAI SDK (`google-genai` using Gemini 3.5 Flash)
- **Document Generation:** ReportLab
- **Data Manipulation & Viz:** Pandas, NumPy, Seaborn, Matplotlib

---

## 📂 Repository Structure

```text
├── E-commerce_Customer_Churn_2026.csv   # Anonymized 50K Customer Dataset
├── 1_veri_temizleme.ipynb               # Data Cleaning & Exploratory Data Analysis
├── 2_model_egitimi_ve_shap.ipynb        # Model Training, Evaluation & SHAP Analysis
├── train_model.py                       # Standalone Model Training & Pipeline Script
├── best_churn_model.pkl                 # Trained Champion Model (XGBoost)
├── model_columns.pkl                    # Feature Column Structure
├── api.py                               # FastAPI Backend REST API Service
├── app.py                               # Streamlit Web Application Dashboard
├── pdf_generator.py                     # ReportLab PDF Report Generator Module
├── requirements.txt                     # Project Dependencies
└── README.md                            # Project Documentation
💻 Installation & Quick Start
1. Clone the Repository
Bash
git clone [https://github.com/gulfemmm/churn-ai-retention-platform.git](https://github.com/gulfemmm/churn-ai-retention-platform.git)
cd churn-ai-retention-platform
2. Install Dependencies
Bash
pip install -r requirements.txt
3. Configure Gemini API Key
Create a .streamlit/secrets.toml file in your root folder and add your Gemini API Key:

Ini, TOML
GEMINI_API_KEY = "your_google_gemini_api_key_here"
4. Train / Regenerate Model (Optional)
Bash
python train_model.py
5. Launch the Backend API (FastAPI - Optional for Microservice Mode)
Bash
uvicorn api:app --reload --port 8000
6. Launch the Streamlit Dashboard
Bash
streamlit run app.py
👤 Author
Gülfem - GitHub Profile

📜 License
This project is licensed under the CC0 1.0 Universal (Public Domain) license.
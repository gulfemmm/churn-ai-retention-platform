

# ⚡ ChurnAI: AI-Powered Customer Churn Prediction & Retention Platform

**ChurnAI** is an end-to-end B2B Decision Support System designed for e-commerce platforms. It predicts customer churn risks in advance, explains the underlying root causes using **Explainable AI (SHAP)**, and provides tailored, actionable retention strategies via **Google Gemini LLM (GenAI)** integration.

---

## 🚀 Key Features

- **End-to-End Machine Learning Pipeline:** Comprehensive data preprocessing, Data Leakage prevention, missing value imputation, and One-Hot Encoding.
- **Model Benchmarking & High Performance:** Comparative analysis between Logistic Regression, Random Forest, and **XGBoost**.
  - 🏆 **Champion Model (XGBoost):** Achieved an impressive **0.9696 ROC-AUC Score**.
- **Explainable AI (XAI with SHAP):** Breaks the "black box" nature of ML models by visualizing key churn drivers (e.g., support ticket volume, satisfaction score, return frequency, and recency).
- **Google Gemini-Powered Retention Agent:** Generates real-time, personalized retention offers and engagement strategies for high-risk customers.
- **Interactive Streamlit Web Dashboard:**
  - 📂 **Batch CSV Processing:** Upload bulk customer records to predict churn probability at scale and download analyzed reports.
  - 👤 **Single-Customer Analysis & GenAI Chatbot:** Simulate customer scenarios in real-time and get instant AI-driven retention advice.

---

## 🛠️ Tech Stack & Dependencies

- **Core Language:** Python 3.12
- **Data Preprocessing & EDA:** Pandas, NumPy, Seaborn, Matplotlib
- **Machine Learning & XAI:** Scikit-Learn, XGBoost, SHAP
- **Generative AI:** Google GenAI SDK (Gemini API)
- **Web Interface:** Streamlit
- **Model Serialization:** Joblib (`.pkl`)

---

## 📂 Repository Structure

```text
├── E-commerce_Customer_Churn_2026.csv   # Anonymized 50K Customer Dataset
├── 1_veri_temizleme_ve_eda.ipynb        # Data Cleaning & Exploratory Data Analysis
├── 2_model_egitimi_ve_shap.ipynb        # Model Training, Evaluation & SHAP Analysis
├── best_churn_model.pkl                 # Trained Champion Model (XGBoost)
├── model_columns.pkl                    # Feature Column Structure
├── app.py                               # Streamlit Web Application & Gemini Chatbot
├── requirements.txt                     # Python Dependencies
└── README.md                            # Project Documentation
💻 Installation & Quick Start
Clone the Repository:

Bash
git clone [https://github.com/gulfemmm/churn-ai-retention-platform.git](https://github.com/gulfemmm/churn-ai-retention-platform.git)
cd churn-ai-retention-platform
Install Dependencies:

Bash
pip install -r requirements.txt
Launch the Streamlit Dashboard:

Bash
streamlit run app.py
👤 Author
Gülfem - GitHub Profile

📜 License
This project is licensed under the CC0 1.0 Universal (Public Domain) license.
>>>>>>> f151b52 (Final check and clean README)

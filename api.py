import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List
from google import genai

app = FastAPI(
    title="ChurnAI Retention API",
    description="End-to-End Churn Prediction & GenAI Retention Engine API",
    version="1.0"
)

# Model ve Sütun Yapısını Yükle
MODEL_PATH = "best_churn_model.pkl"
COLUMNS_PATH = "model_columns.pkl"

if os.path.exists(MODEL_PATH) and os.path.exists(COLUMNS_PATH):
    model = joblib.load(MODEL_PATH)
    model_columns = joblib.load(COLUMNS_PATH)
else:
    model = None
    model_columns = None

class PredictionInput(BaseModel):
    data: Dict[str, Any]

class RetentionInput(BaseModel):
    customer_data: Dict[str, Any]
    churn_probability: float
    api_key: str

@app.get("/")
def home():
    return {"status": "online", "message": "ChurnAI Backend API is running."}

@app.post("/predict")
def predict_churn(payload: PredictionInput):
    if not model or not model_columns:
        raise HTTPException(status_code=500, detail="Model files not found on server.")
    
    try:
        df = pd.DataFrame([payload.data])
        df_encoded = pd.get_dummies(df)
        df_encoded = df_encoded.reindex(columns=model_columns, fill_value=0)
        
        prob = float(model.predict_proba(df_encoded)[0][1])
        prediction = int(prob >= 0.5)
        
        return {
            "churn_prediction": prediction,
            "churn_probability": round(prob, 4),
            "risk_level": "High Risk" if prob >= 0.6 else ("Medium Risk" if prob >= 0.3 else "Low Risk")
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/generate-retention")
def generate_retention_strategy(payload: RetentionInput):
    try:
        client = genai.Client(api_key=payload.api_key)
        prompt = f"""
        Act as an expert Customer Retention Specialist for an e-commerce platform.
        Customer Profile: {payload.customer_data}
        Predicted Churn Probability: {payload.churn_probability * 100:.1f}%

        Provide a concise 3-step action plan to prevent this customer from churning:
        1. Root cause summary based on profile.
        2. Personalized offer / incentive.
        3. Recommended communication channel & messaging tone.
        Keep it executive-ready and professional.
        """
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        return {"retention_strategy": response.text}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Gemini API Error: {str(e)}")
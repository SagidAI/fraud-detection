from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import joblib
import numpy as np

app = FastAPI()
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
model = joblib.load('fraud_model.pkl')
scaler = joblib.load('scaler.pkl')

class Transaction(BaseModel):
    features: List[float]

@app.get("/")
def home():
    return {"message": "Fraud Detection API is running!"}

@app.post("/predict")
def predict(transaction: Transaction):
    features = transaction.features
    features[-1] = scaler.transform([[features[-1]]])[0][0]
    features = np.array(features).reshape(1, -1)
    
    prob = model.predict_proba(features)[0][1]
    is_fraud = prob >= 0.3
    
    return {
        "is_fraud": bool(is_fraud),
        "confidence_score": round(float(prob), 4)
    }
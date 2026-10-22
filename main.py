from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np

# Initialize FastAPI app ONCE
app = FastAPI()

# Enable CORS so your HTML frontend can talk to it
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the saved models
reg_model = joblib.load('regression_model.pkl')
clf_model = joblib.load('classification_model.pkl')

# Define the expected input data structure from the user
class ModelInput(BaseModel):
    txn_count_last_24h: float
    customer_age: float
    card_age_months: float
    distance_from_home_km: float
    cvv_retry_count: float

# Endpoint 1: Regression Prediction
@app.post("/predict/regression")
def predict_regression(data: ModelInput):
    features = np.array([[
        data.txn_count_last_24h,
        data.customer_age,
        data.card_age_months,
        data.distance_from_home_km,
        data.cvv_retry_count
    ]])
    prediction = reg_model.predict(features)[0]
    return {"prediction": float(prediction)}

# Endpoint 2: Classification Prediction
@app.post("/predict/classification")
def predict_classification(data: ModelInput):
    features = np.array([[
        data.txn_count_last_24h,
        data.customer_age,
        data.card_age_months,
        data.distance_from_home_km,
        data.cvv_retry_count
    ]])
    prediction = int(clf_model.predict(features)[0])
    return {"prediction": prediction}

import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="Customer Churn Prediction API")

# Load the trained model
model = joblib.load("churn_model.pkl")


class Customer(BaseModel):
    tenure: int = Field(ge=1, le=72)
    monthly_charges: float = Field(gt=0, le=200)
    support_calls: int = Field(ge=0, le=20)


@app.get("/")
def home():
    return {"message": "Customer Churn Prediction API is running"}


@app.post("/predict")
def predict_churn(customer: Customer):
    features = np.array([[
        customer.tenure,
        customer.monthly_charges,
        customer.support_calls
    ]])

    prediction = model.predict(features)[0]
    result = "Churn" if prediction == 1 else "Stay"

    return {"prediction": result}

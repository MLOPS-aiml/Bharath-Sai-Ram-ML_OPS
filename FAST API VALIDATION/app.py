
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="Customer Churn Prediction API")

# Load the saved model
model = joblib.load("churn_model.pkl")


# Input validation
class CustomerData(BaseModel):
    tenure: int = Field(ge=1, le=72)
    monthly_charges: float = Field(gt=0, le=200)
    support_calls: int = Field(ge=0, le=20)


@app.get("/")
def home():
    return {
        "status": "success",
        "message": "Customer Churn Prediction API is Running"
    }


@app.post("/predict")
def predict(data: CustomerData):
    try:
        features = np.array([[
            data.tenure,
            data.monthly_charges,
            data.support_calls
        ]])

        prediction = model.predict(features)[0]

        result = (
            "Customer Will Churn"
            if prediction == 1
            else "Customer Will Stay"
        )

        return {
            "status": "success",
            "prediction": result
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction Error: {str(e)}"
        )

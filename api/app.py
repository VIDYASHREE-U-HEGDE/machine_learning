from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os
import pandas as pd


app = FastAPI()


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


model_path = os.path.join(
    BASE_DIR,
    "models",
    "model.pkl"
)


model = joblib.load(model_path)


class Customer(BaseModel):
    Gender: str
    Age: int
    Tenure: int
    MonthlyCharges: float
    TotalCharges: float
    ContractType: str
    InternetService: str
    PaymentMethod: str


@app.get("/")
def home():
    return {
        "message": "Customer Churn Prediction API Running"
    }


@app.post("/predict")
def predict(customer: Customer):

    data = pd.DataFrame(
        [customer.dict()]
    )

    prediction = model.predict(data)

    if prediction[0] == 1:
        result = "Customer will churn"
    else:
        result = "Customer will stay"

    return {
        "prediction": result
    }
import os
import joblib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel



# -----------------------------------
# Create FastAPI App
# -----------------------------------

app = FastAPI(

    title="Customer Churn Prediction API",

    description="API for predicting customer churn",

    version="1.0"

)



# -----------------------------------
# Load Model
# -----------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


MODEL_PATH = os.path.join(

    BASE_DIR,

    "models",

    "model.pkl"

)


model = joblib.load(
    MODEL_PATH
)


print("Model loaded successfully")



# -----------------------------------
# Input Schema
# -----------------------------------

class CustomerData(BaseModel):

    Gender: str

    Age: int

    Tenure: int

    MonthlyCharges: float

    TotalCharges: float

    ContractType: str

    InternetService: str

    PaymentMethod: str




# -----------------------------------
# Home API
# -----------------------------------

@app.get("/")
def home():

    return {

        "message": "Customer Churn Prediction API is running"

    }




# -----------------------------------
# Prediction API
# -----------------------------------

@app.post("/predict")
def predict(customer: CustomerData):


    try:


        # Convert input into dataframe

        data = pd.DataFrame(

            [customer.dict()]

        )



        # Prediction

        prediction = model.predict(

            data

        )[0]



        # Probability

        probability = model.predict_proba(

            data

        )[0][1]



        if prediction == 1:

            result = "Customer will Churn"

        else:

            result = "Customer will Stay"



        return {


            "prediction": int(prediction),


            "status": result,


            "churn_probability": round(

                float(probability),

                3

            )

        }



    except Exception as e:


        return {


            "error": str(e)

        }




# -----------------------------------
# Run Server
# -----------------------------------

if __name__ == "__main__":

    import uvicorn


    uvicorn.run(

        "app:app",

        host="127.0.0.1",

        port=8000,

        reload=True

    )
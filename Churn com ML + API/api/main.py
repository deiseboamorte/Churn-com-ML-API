from fastapi import FastAPI
from src.schemas import Customer
from src.predict import predict

app = FastAPI(
    title="Churn Prediction API",
    version="1.0"
)

@app.get("/")
def root():
    return {"status": "running"}

@app.post("/predict")
def get_prediction(customer: Customer):
    result = predict(customer.dict())
    return result
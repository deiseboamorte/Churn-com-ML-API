import joblib
import pandas as pd

from src.config import MODEL_PATH

model = joblib.load(MODEL_PATH)

def predict(data: dict):
    df = pd.DataFrame([data])
    proba = model.predict_proba(df)[0][1]
    pred = model.predict(df)[0]

    return {
        "churn": int(pred),
        "probability": float(proba)
    }
import os
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

model = joblib.load(
    os.path.join(PROJECT_ROOT, "models", "model.pkl")
)

FEATURE_ORDER = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']

THRESHOLD = 0.5


class Transaction(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float


@app.post("/predict")
def predict(transaction: Transaction):

    data = transaction.model_dump()

    X = pd.DataFrame(
        [[data[feature] for feature in FEATURE_ORDER]],
        columns=FEATURE_ORDER
    )

    probabilities = model.predict_proba(X)[0]

    probability = probabilities[1]

    class_id = int(probability >= THRESHOLD)

    return {
        "prediction": "Fraud" if class_id == 1 else "Legitimate",
        "class_id": class_id,
        "probability": probability,
        "threshold": THRESHOLD,
        "status": "success"
    }
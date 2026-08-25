import sys
import os
import json
import joblib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

FEATURE_ORDER = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]


def load_artifacts():
    model = joblib.load(os.path.join(MODEL_DIR, "model.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    with open(os.path.join(MODEL_DIR, "model_config.json")) as f:
        threshold = json.load(f)["threshold"]
    return model, scaler, threshold


def predict_transaction(model, scaler, threshold, transaction: dict):
    missing = [f for f in FEATURE_ORDER if f not in transaction]
    if missing:
        return {"status": "error", "message": f"missing features: {missing}"}

    X = pd.DataFrame([[transaction[f] for f in FEATURE_ORDER]], columns=FEATURE_ORDER)
    X_scaled = scaler.transform(X)

    proba = model.predict_proba(X_scaled)[0][1]
    class_id = int(proba >= threshold)

    return {
        "prediction": "Fraud" if class_id == 1 else "Legitimate",
        "class_id": class_id,
        "probability": float(proba),
        "threshold": threshold,
        "status": "success"
    }


def main():
    if len(sys.argv) < 2:
        print(json.dumps({"status": "error", "message": "usage: python predict.py input.json"}))
        sys.exit(1)

    with open(sys.argv[1], "r") as f:
        transaction = json.load(f)

    model, scaler, threshold = load_artifacts()
    result = predict_transaction(model, scaler, threshold, transaction)

    print(json.dumps(result, indent=2))
    with open("output.json", "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()

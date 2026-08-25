import os
import json
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)  # one level up from the script's folder

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "creditcard.csv")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

FINAL_THRESHOLD = 0.1168


def main():
    data = pd.read_csv(DATA_PATH)
    data = data.drop_duplicates().reset_index(drop=True)

    X = data.drop(columns="Class")
    y = data["Class"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression()
    model.fit(X_scaled, y)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODEL_DIR, "model.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

    with open(os.path.join(MODEL_DIR, "model_config.json"), "w") as f:
        json.dump({"threshold": FINAL_THRESHOLD}, f, indent=2)

    print(f"Trained on {len(data)} rows.")
    print(f"Saved model.pkl, scaler.pkl, model_config.json to {MODEL_DIR}/")


if __name__ == "__main__":
    main()
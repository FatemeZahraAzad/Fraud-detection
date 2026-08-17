import sys
import json
import joblib
import pandas as pd

FEATURE_ORDER = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
THRESHOLD = 0.5

# نکته: این مدل بدون StandardScaler آموزش دیده، چون در Experiment 1 ثابت شد
# Decision Tree به Scaling حساس نیست (نتیجه تغییری نمی‌کند). بنابراین scaler.pkl
# در این پروژه عمداً وجود ندارد. به همین ترتیب encoder.pkl هم لازم نیست،
# چون هیچ Feature دسته‌ای (Categorical) در این دیتاست وجود ندارد.

def load_model(path='models/model.pkl'):
    return joblib.load(path)

def predict_transaction(model, transaction: dict):
    missing = [f for f in FEATURE_ORDER if f not in transaction]
    if missing:
        return {"status": "error", "message": f"missing features: {missing}"}

    X = pd.DataFrame([[transaction[f] for f in FEATURE_ORDER]], columns=FEATURE_ORDER)

    # نکته: predict_proba این مدل خاص (max_depth=None) فقط 0.0 یا 1.0 می‌دهد،
    # چون برگ‌های درخت کاملاً Pure شده‌اند (رجوع کنید به Experiment 3، بخش 12 README).
    proba = model.predict_proba(X)[0][1]
    class_id = int(proba >= THRESHOLD)

    return {
        "prediction": "Fraud" if class_id == 1 else "Legitimate",
        "class_id": class_id,
        "probability": float(proba),
        "threshold": THRESHOLD,
        "status": "success"
    }

def main():
    if len(sys.argv) < 2:
        print(json.dumps({"status": "error", "message": "usage: python predict.py input.json"}))
        sys.exit(1)

    with open(sys.argv[1], 'r') as f:
        transaction = json.load(f)

    model = load_model()
    result = predict_transaction(model, transaction)

    print(json.dumps(result, indent=2))
    with open('output.json', 'w') as f:
        json.dump(result, f, indent=2)

if __name__ == "__main__":
    main()
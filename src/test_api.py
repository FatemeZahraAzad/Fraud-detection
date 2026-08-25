from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_valid_transaction_returns_200():
    payload = {"Time": 0.0, **{f"V{i}": 0.0 for i in range(1, 29)}, "Amount": 100.0}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    assert body["prediction"] in ("Fraud", "Legitimate")
    assert 0.0 <= body["probability"] <= 1.0

def test_missing_field_returns_422():
    resp = client.post("/predict", json={"Time": 1.0, "V1": 0.1})
    assert resp.status_code == 422
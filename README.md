# Credit Card Fraud Detection

A machine learning project for classifying credit card transactions as legitimate or fraudulent, with a particular focus on handling severe class imbalance, preventing data leakage, and principled tuning of the Precision/Recall trade-off.

## Project Structure

```
Fraud-detection/
├── data/
│   └── creditcard.csv          # Raw dataset (download separately, not tracked in git)
├── models/
│   ├── model.pkl                # Final model (Logistic Regression) — produced by train.py
│   ├── scaler.pkl                # StandardScaler fitted on the full dataset
│   └── model_config.json         # Final decision threshold
├── reports/
│   ├── experiments.md            # Full experiment log, results, and analysis
│   └── assets/                   # Figures (stability boxplot, max_depth curve, PR curve)
├── src/
│   ├── data_prep.ipynb           # Full notebook: EDA, preprocessing, training, experiments
│   ├── train.py                  # Trains the final model from scratch and saves artifacts
│   ├── predict.py                # Command-line prediction (single-record JSON in/out)
│   ├── app.py                    # Prediction API service (FastAPI)
│   └── test_api.py               # Smoke tests for app.py
└── README.md
```

## Installation

```bash
pip install pandas numpy scikit-learn joblib fastapi uvicorn pytest httpx
```

## 1. Train the Model

The raw dataset (`creditcard.csv`) must be placed at `data/creditcard.csv` ([dataset source](https://www.kaggle.com/mlg-ulb/creditcardfraud)). Then:

```bash
cd src
python train.py
```

This script:
- Reads the data and drops duplicate rows
- Fits `StandardScaler` on the full dataset
- Trains `LogisticRegression` on the full dataset
- Saves `model.pkl`, `scaler.pkl`, and `model_config.json` to `models/`

## 2. Predict from the Command Line

Input is a single-record JSON file containing `Time`, `V1`...`V28`, `Amount`:

```bash
cd src
python predict.py input.json
```

Output is printed to the terminal and also written to `output.json`:

```json
{
  "prediction": "Fraud",
  "class_id": 1,
  "probability": 0.87,
  "threshold": 0.1168,
  "status": "success"
}
```

## 3. Run the API

```bash
cd src
uvicorn app:app --reload
```

The server starts at `http://127.0.0.1:8000`. Interactive documentation (Swagger UI) is available at `http://127.0.0.1:8000/docs`.

Example request:
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d @input.json
```

### Running Tests

```bash
cd src
pytest test_api.py -v
```

## Results Summary

| Final Model | Precision | Recall | F1 |
|---|---:|---:|---:|
| Logistic Regression (Threshold=0.1168) | 0.833 | 0.737 | 0.782 |

The full methodology, three controlled experiments (Scaling effect, tree depth effect, Threshold tuning), model comparison, and business interpretation are documented in [`reports/experiments.md`](./reports/experiments.md).

## Key Design Notes

- **Leakage prevention:** duplicates removed before the Train/Test split; the scaler fitted only inside each Cross-Validation fold's Train portion (never on Test); the decision threshold selected solely from out-of-fold Train probabilities.
- **Handling class imbalance:** instead of `class_weight` (which severely sacrificed Precision), the Precision-Recall curve was used to tune the decision threshold precisely.
- **Separated model and scaler:** unlike the single combined Pipeline used in the experimentation notebook, the production code (`train.py` / `predict.py` / `app.py`) saves and applies the model and scaler as two separate artifacts.
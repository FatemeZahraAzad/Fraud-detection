import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)   # یک پوشه به عقب: Fraud-detection/

data = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "creditcard.csv"))
data = data.drop_duplicates()

X = data.drop(labels="Class", axis=1)
y = data["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

final_model = DecisionTreeClassifier(max_depth=None, class_weight='balanced', random_state=42)
final_model.fit(X_train, y_train)

os.makedirs(os.path.join(PROJECT_ROOT, "models"), exist_ok=True)
joblib.dump(final_model, os.path.join(PROJECT_ROOT, "models", "model.pkl"))
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from src.pipeline import create_pipeline
from src.config import TARGET, MODEL_PATH

df = pd.read_csv("data/raw.csv")

df[TARGET] = df[TARGET].map({"Yes": 1, "No": 0})

X = df.drop(TARGET, axis=1)
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline = create_pipeline()
pipeline.fit(X_train, y_train)

preds = pipeline.predict_proba(X_test)[:, 1]
score = roc_auc_score(y_test, preds)

print(f"ROC-AUC: {score:.4f}")

joblib.dump(pipeline, MODEL_PATH)
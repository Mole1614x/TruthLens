from pathlib import Path

import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

PROJECT_DIR = Path(__file__).resolve().parent

TEST_DATA = PROJECT_DIR / "data/processed/ann/test_sequences.npz"
MODEL_PATH = PROJECT_DIR / "models/truthlens_ann_best.keras"
REPORT_PATH = PROJECT_DIR / "reports/ann_test_report.csv"
SUMMARY_PATH = PROJECT_DIR / "reports/ann_test_summary.csv"

print("Loading test sequences...")
with np.load(TEST_DATA) as data:
    X_test = data["X"]
    y_test = data["y"]

print("Test input shape:", X_test.shape)

print("\nLoading best ANN model...")
model = keras.models.load_model(MODEL_PATH, compile=False)

print("\nGenerating predictions...")
probabilities = model.predict(
    X_test, batch_size=128, verbose=1
).ravel()

y_pred = (probabilities >= 0.5).astype(np.int32)

accuracy = accuracy_score(y_test, y_pred)
weighted_f1 = f1_score(
    y_test, y_pred, average="weighted", zero_division=0
)

print("\n========== ANN TEST RESULTS ==========")
print(f"Accuracy:    {accuracy:.4%}")
print(f"Weighted F1: {weighted_f1:.4f}")

print("\nClassification report:")
print(classification_report(
    y_test,
    y_pred,
    labels=[0, 1],
    target_names=["Fake", "Real"],
    digits=4,
    zero_division=0,
))

print("Confusion matrix (actual rows, predicted columns):")
print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

report = classification_report(
    y_test,
    y_pred,
    labels=[0, 1],
    target_names=["Fake", "Real"],
    output_dict=True,
    zero_division=0,
)
pd.DataFrame(report).transpose().to_csv(REPORT_PATH)

pd.DataFrame([{
    "model": "ANN",
    "test_samples": len(y_test),
    "accuracy": accuracy,
    "weighted_f1": weighted_f1,
    "correct_predictions": int((y_test == y_pred).sum()),
    "incorrect_predictions": int((y_test != y_pred).sum()),
}]).to_csv(SUMMARY_PATH, index=False)

print("\nSaved report:", REPORT_PATH)
print("Saved summary:", SUMMARY_PATH)

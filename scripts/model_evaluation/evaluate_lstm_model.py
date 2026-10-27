from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# Load validation data
data = np.load("data/processed/ann/validation_sequences.npz")
X_val = data["X"]
y_val = data["y"]

# Load the best LSTM checkpoint
model = tf.keras.models.load_model(
    "models/truthlens_lstm_best.keras",
    compile=False
)

# Generate predictions
scores = model.predict(X_val, batch_size=128, verbose=1).ravel()
y_pred = (scores >= 0.5).astype("int32")

# Calculate metrics
print("\n========== LSTM VALIDATION RESULTS ==========")
print(f"Accuracy:          {accuracy_score(y_val, y_pred):.4f}")
print(f"Macro precision:   {precision_score(y_val, y_pred, average='macro', zero_division=0):.4f}")
print(f"Macro recall:      {recall_score(y_val, y_pred, average='macro', zero_division=0):.4f}")
print(f"Macro F1:          {f1_score(y_val, y_pred, average='macro', zero_division=0):.4f}")
print(f"Weighted F1:       {f1_score(y_val, y_pred, average='weighted', zero_division=0):.4f}")

print("\nClassification report:")
print(
    classification_report(
        y_val,
        y_pred,
        labels=[0, 1],
        target_names=["Fake", "Real"],
        digits=4,
        zero_division=0
    )
)

print("Confusion matrix [actual rows, predicted columns]:")
print(confusion_matrix(y_val, y_pred, labels=[0, 1]))

# Save predictions for error analysis
output_path = Path("models/lstm_validation_predictions.csv")
pd.DataFrame({
    "actual_label": y_val,
    "predicted_label": y_pred,
    "real_score": scores,
    "predicted_class": np.where(y_pred == 1, "Real", "Fake")
}).to_csv(output_path, index=False)

print(f"\nPredictions saved to: {output_path}")
print("\nANN validation accuracy baseline: 0.9658")

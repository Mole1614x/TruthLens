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

# File paths
DATA_DIR = Path("data/processed/ann")
MODEL_PATH = Path("models/truthlens_cnn_best.keras")
OUTPUT_PATH = Path("models/cnn_validation_predictions.csv")

# Load validation sequences
validation_data = np.load(
    DATA_DIR / "validation_sequences.npz"
)
X_val = validation_data["X"]
y_val = validation_data["y"]

print("Validation data shape:", X_val.shape)

# Load the best CNN checkpoint
model = keras.models.load_model(
    MODEL_PATH,
    compile=False
)

# Generate prediction probabilities
probabilities = model.predict(
    X_val,
    batch_size=128,
    verbose=1
).ravel()

# Convert probabilities into class labels
y_pred = (probabilities >= 0.5).astype("int32")

# Calculate metrics
accuracy = accuracy_score(y_val, y_pred)
weighted_f1 = f1_score(
    y_val,
    y_pred,
    average="weighted"
)

print("\nCNN Validation Results")
print("----------------------")
print(f"Accuracy: {accuracy:.4f}")
print(f"Weighted F1: {weighted_f1:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_val,
        y_pred,
        labels=[0, 1],
        target_names=["Fake", "Real"],
        digits=4,
        zero_division=0,
    )
)

print("Confusion Matrix (actual rows, predicted columns):")
print(
    confusion_matrix(
        y_val,
        y_pred,
        labels=[0, 1]
    )
)

# Save predictions for later error analysis
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

pd.DataFrame({
    "actual_label": y_val,
    "predicted_label": y_pred,
    "real_probability": probabilities,
}).to_csv(OUTPUT_PATH, index=False)

print("\nPredictions saved to:", OUTPUT_PATH)

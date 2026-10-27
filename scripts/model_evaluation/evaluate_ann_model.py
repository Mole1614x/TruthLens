from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data" / "processed" / "ann"
MODEL_DIR = PROJECT_DIR / "models"

MODEL_PATH = MODEL_DIR / "truthlens_ann.keras"
VALIDATION_PATH = DATA_DIR / "validation_sequences.npz"
HISTORY_PATH = MODEL_DIR / "truthlens_ann_training_history.csv"

# 1. Load validation arrays
print("\nLoading validation data...")

with np.load(VALIDATION_PATH) as data:
    X_val = data["X"]
    y_val = data["y"]

print("Validation samples:", len(y_val))
print("Input shape:", X_val.shape)

# 2. Load the trained model
print("\nLoading trained ANN...")
model = keras.models.load_model(MODEL_PATH)

# 3. Generate probabilities and class predictions
print("\nGenerating predictions...")

probabilities = model.predict(
    X_val,
    batch_size=128,
    verbose=1
).ravel()

# Threshold 0.5: probability >= 0.5 means Real (1)
y_pred = (probabilities >= 0.5).astype(np.int32)

# 4. Calculate evaluation metrics
accuracy = accuracy_score(y_val, y_pred)
precision_macro = precision_score(
    y_val, y_pred, average="macro", zero_division=0
)
recall_macro = recall_score(
    y_val, y_pred, average="macro", zero_division=0
)
f1_macro = f1_score(
    y_val, y_pred, average="macro", zero_division=0
)
f1_weighted = f1_score(
    y_val, y_pred, average="weighted", zero_division=0
)

print("\n========== VALIDATION METRICS ==========")
print(f"Accuracy:          {accuracy:.4f}")
print(f"Macro precision:   {precision_macro:.4f}")
print(f"Macro recall:      {recall_macro:.4f}")
print(f"Macro F1-score:    {f1_macro:.4f}")
print(f"Weighted F1-score: {f1_weighted:.4f}")

print("\nClassification report:")
print(
    classification_report(
        y_val,
        y_pred,
        labels=[0, 1],
        target_names=["Fake (0)", "Real (1)"],
        digits=4,
        zero_division=0
    )
)

# Rows = actual class; columns = predicted class
cm = confusion_matrix(y_val, y_pred, labels=[0, 1])

print("Confusion matrix")
print("Rows = actual [Fake, Real]")
print("Columns = predicted [Fake, Real]")
print(cm)

# 5. Save validation predictions for error analysis
predictions_path = MODEL_DIR / "ann_validation_predictions.csv"

pd.DataFrame({
    "actual_label": y_val,
    "predicted_label": y_pred,
    "probability_real": probabilities,
    "correct": (y_val == y_pred)
}).to_csv(predictions_path, index=False)

print("\nSaved validation predictions to:", predictions_path)

# 6. Inspect the training history
if HISTORY_PATH.exists():
    history = pd.read_csv(HISTORY_PATH)

    print("\n========== TRAINING HISTORY ==========")
    print(history.round(4).to_string(index=False))

    best_row = history.loc[history["val_loss"].idxmin()]

    print("\nBest epoch:", int(best_row["epoch"]))
    print(f"Training loss at best epoch: {best_row['loss']:.4f}")
    print(f"Validation loss at best epoch: {best_row['val_loss']:.4f}")
    print(f"Training accuracy at best epoch: {best_row['accuracy']:.4f}")
    print(f"Validation accuracy at best epoch: {best_row['val_accuracy']:.4f}")

    print(
        "\nCompare training and validation loss across epochs. "
        "A growing gap can indicate overfitting."
    )
else:
    print("\nTraining history file was not found.")

print("\nValidation evaluation complete. The test set was not used.")

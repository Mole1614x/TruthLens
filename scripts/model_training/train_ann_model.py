from pathlib import Path
import gc
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras


# --------------------------------------------------
# 1. Paths and configuration
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent

DATA_DIR = PROJECT_DIR / "data" / "processed" / "ann"
MODEL_DIR = PROJECT_DIR / "models"

TRAIN_DATA = DATA_DIR / "train_sequences.npz"
VALIDATION_DATA = DATA_DIR / "validation_sequences.npz"

INITIAL_MODEL = MODEL_DIR / "truthlens_ann_untrained.keras"
BEST_MODEL = MODEL_DIR / "truthlens_ann_best.keras"
FINAL_MODEL = MODEL_DIR / "truthlens_ann.keras"
HISTORY_FILE = MODEL_DIR / "truthlens_ann_training_history.csv"

BATCH_SIZE = 128
EPOCHS = 8
PATIENCE = 2

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Load the prepared arrays
# --------------------------------------------------

print("\nLoading training and validation arrays...")

with np.load(TRAIN_DATA) as data:
    X_train = data["X"]
    y_train = data["y"]

with np.load(VALIDATION_DATA) as data:
    X_val = data["X"]
    y_val = data["y"]

print("Training input shape:", X_train.shape)
print("Training labels shape:", y_train.shape)
print("Validation input shape:", X_val.shape)
print("Validation labels shape:", y_val.shape)

assert X_train.shape[0] == y_train.shape[0]
assert X_val.shape[0] == y_val.shape[0]
assert X_train.shape[1] == X_val.shape[1] == 300

print("\nLabel mapping: 0 = Fake, 1 = Real")


# --------------------------------------------------
# 3. Load the untrained architecture
# --------------------------------------------------

print("\nLoading the ANN architecture...")

model = keras.models.load_model(INITIAL_MODEL)

model.summary()


# --------------------------------------------------
# 4. Configure callbacks
# --------------------------------------------------

early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=PATIENCE,
    restore_best_weights=True,
    verbose=1
)

checkpoint = keras.callbacks.ModelCheckpoint(
    filepath=str(BEST_MODEL),
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

callbacks = [
    early_stopping,
    checkpoint
]


# --------------------------------------------------
# 5. Train the model
# --------------------------------------------------

print("\nStarting ANN training...")
print("Maximum epochs:", EPOCHS)
print("Batch size:", BATCH_SIZE)

start_time = time.time()

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    shuffle=True,
    verbose=1
)

elapsed_time = time.time() - start_time

print(f"\nTraining time: {elapsed_time / 60:.2f} minutes")


# --------------------------------------------------
# 6. Restore the best validation checkpoint
# --------------------------------------------------

print("\nLoading the best checkpoint...")

del model
gc.collect()

model = keras.models.load_model(BEST_MODEL)


# --------------------------------------------------
# 7. Save the trained model
# --------------------------------------------------

model.save(FINAL_MODEL)

print("\nTrained model saved to:", FINAL_MODEL)


# --------------------------------------------------
# 8. Save training history
# --------------------------------------------------

history_df = pd.DataFrame(history.history)
history_df.insert(
    0,
    "epoch",
    range(1, len(history_df) + 1)
)

history_df.to_csv(HISTORY_FILE, index=False)

print("Training history saved to:", HISTORY_FILE)


# --------------------------------------------------
# 9. Report the best validation results
# --------------------------------------------------

best_epoch_index = int(
    np.argmin(history.history["val_loss"])
)

best_epoch = best_epoch_index + 1
best_val_loss = history.history["val_loss"][best_epoch_index]
best_val_accuracy = history.history["val_accuracy"][best_epoch_index]

print("\n========== TRAINING SUMMARY ==========")
print("Epochs completed:", len(history.history["loss"]))
print("Best epoch:", best_epoch)
print(f"Best validation loss: {best_val_loss:.4f}")
print(f"Validation accuracy at best epoch: {best_val_accuracy:.4f}")
print(f"Total training time: {elapsed_time / 60:.2f} minutes")
print("Model status: TRAINED")
print("Test data used: No")
print("======================================")

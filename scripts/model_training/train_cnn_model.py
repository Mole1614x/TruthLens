from pathlib import Path
import random

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Paths
DATA_DIR = Path("data/processed/ann")
MODEL_DIR = Path("models")

TRAIN_FILE = DATA_DIR / "train_sequences.npz"
VALIDATION_FILE = DATA_DIR / "validation_sequences.npz"
MODEL_FILE = MODEL_DIR / "truthlens_cnn_untrained.keras"

BEST_MODEL_FILE = MODEL_DIR / "truthlens_cnn_best.keras"
FINAL_MODEL_FILE = MODEL_DIR / "truthlens_cnn.keras"
HISTORY_FILE = MODEL_DIR / "truthlens_cnn_training_history.csv"

# Load prepared sequences
train_data = np.load(TRAIN_FILE)
validation_data = np.load(VALIDATION_FILE)

X_train = train_data["X"]
y_train = train_data["y"]

X_val = validation_data["X"]
y_val = validation_data["y"]

print("Training data shape:", X_train.shape)
print("Validation data shape:", X_val.shape)

# Load the untrained architecture
model = keras.models.load_model(
    MODEL_FILE,
    compile=False
)

# Configure training
model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Save the checkpoint with the lowest validation loss
checkpoint = keras.callbacks.ModelCheckpoint(
    filepath=BEST_MODEL_FILE,
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

# Stop when validation loss stops improving
early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True,
    verbose=1
)

# Train
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    batch_size=128,
    epochs=8,
    callbacks=[checkpoint, early_stopping],
    verbose=1
)

# Save the trained model and training history
MODEL_DIR.mkdir(parents=True, exist_ok=True)
model.save(FINAL_MODEL_FILE)

history_df = pd.DataFrame(history.history)
history_df.insert(0, "epoch", range(1, len(history_df) + 1))
history_df.to_csv(HISTORY_FILE, index=False)

print("\nTraining completed.")
print(
    "Best validation loss:",
    round(min(history.history["val_loss"]), 4)
)
print(
    "Best validation accuracy:",
    round(max(history.history["val_accuracy"]), 4)
)
print("Best checkpoint:", BEST_MODEL_FILE)
print("Final model:", FINAL_MODEL_FILE)
print("Training history:", HISTORY_FILE)

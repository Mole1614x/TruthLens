from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

# Reproducibility
keras.utils.set_random_seed(42)

# Load preprocessed sequences
train_data = np.load("data/processed/ann/train_sequences.npz")
validation_data = np.load("data/processed/ann/validation_sequences.npz")

X_train = train_data["X"]
y_train = train_data["y"]

X_val = validation_data["X"]
y_val = validation_data["y"]

print("Training data:", X_train.shape, y_train.shape)
print("Validation data:", X_val.shape, y_val.shape)

# Load the untrained architecture
model = keras.models.load_model(
    "models/truthlens_lstm_untrained.keras",
    compile=False
)

# Compile with a fresh optimizer
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Prepare output paths
output_dir = Path("models")
output_dir.mkdir(parents=True, exist_ok=True)

best_model_path = output_dir / "truthlens_lstm_best.keras"
final_model_path = output_dir / "truthlens_lstm.keras"

# Stop training if validation loss stops improving
early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True,
    verbose=1
)

# Save the best model observed during training
checkpoint = keras.callbacks.ModelCheckpoint(
    filepath=str(best_model_path),
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

print("\nStarting LSTM training...")

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=8,
    batch_size=128,
    callbacks=[early_stopping, checkpoint],
    verbose=1
)

# Save the model with the best restored weights
model.save(final_model_path)

# Save training history for later analysis
history_df = pd.DataFrame(history.history)
history_df.insert(
    0,
    "epoch",
    range(1, len(history_df) + 1)
)
history_df.to_csv(
    output_dir / "truthlens_lstm_training_history.csv",
    index=False
)

best_epoch = int(np.argmin(history.history["val_loss"]) + 1)

print("\nTraining completed.")
print("Epochs completed:", len(history.history["loss"]))
print("Best epoch by validation loss:", best_epoch)
print("Best validation loss:", round(min(history.history["val_loss"]), 4))
print("Best validation accuracy:", round(max(history.history["val_accuracy"]), 4))
print("Saved model:", final_model_path)
print("Saved best checkpoint:", best_model_path)
print("Saved history: models/truthlens_lstm_training_history.csv")

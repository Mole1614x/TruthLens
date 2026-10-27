from pathlib import Path

import numpy as np
import keras
import keras_hub

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

TRAIN_DATA_PATH = "data/processed/distilbert/train.npz"
VALIDATION_DATA_PATH = "data/processed/distilbert/validation.npz"

MODEL_PATH = "models/truthlens_distilbert_untrained.keras"

OUTPUT_DIR = Path("models")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 8
EPOCHS = 3
LEARNING_RATE = 2e-5

# --------------------------------------------------
# 2. Load prepared datasets
# --------------------------------------------------

train_data = np.load(TRAIN_DATA_PATH)
validation_data = np.load(VALIDATION_DATA_PATH)

train_input_ids = train_data["input_ids"]
train_attention_masks = train_data["attention_mask"]
train_labels = train_data["labels"]

val_input_ids = validation_data["input_ids"]
val_attention_masks = validation_data["attention_mask"]
val_labels = validation_data["labels"]

print("Training examples:", len(train_labels))
print("Validation examples:", len(val_labels))

# --------------------------------------------------
# 3. Load the built DistilBERT model
# --------------------------------------------------

model = keras.models.load_model(MODEL_PATH)

# --------------------------------------------------
# 4. Compile the model
# --------------------------------------------------

model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    ),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"]
)

# --------------------------------------------------
# 5. Define callbacks
# --------------------------------------------------

early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=1,
    restore_best_weights=True
)

model_checkpoint = keras.callbacks.ModelCheckpoint(
    filepath="models/truthlens_distilbert_best.keras",
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

# --------------------------------------------------
# 6. Train the model
# --------------------------------------------------

history = model.fit(
    x={
        "token_ids": train_input_ids,
        "padding_mask": train_attention_masks
    },
    y=train_labels,

    validation_data=(
        {
            "token_ids": val_input_ids,
            "padding_mask": val_attention_masks
        },
        val_labels
    ),

    batch_size=BATCH_SIZE,
    epochs=EPOCHS,

    callbacks=[
        early_stopping,
        model_checkpoint
    ],

    verbose=1
)

# --------------------------------------------------
# 7. Save the final model
# --------------------------------------------------

model.save(
    "models/truthlens_distilbert.keras"
)

# --------------------------------------------------
# 8. Save training history
# --------------------------------------------------

history_df = keras.utils.PyDataset

import pandas as pd

history_df = pd.DataFrame(history.history)

history_df.to_csv(
    "models/distilbert_training_history.csv",
    index=False
)

print("\nDistilBERT training completed.")
print("Best model:", "models/truthlens_distilbert_best.keras")
print("Final model:", "models/truthlens_distilbert.keras")
print("Training history saved.")

# cat > build_ann_model.py <<'PY'
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# --------------------------------------------------
# 1. Project configuration
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

MAX_TOKENS = 20_000
SEQUENCE_LENGTH = 300
EMBEDDING_DIM = 64


# --------------------------------------------------
# 2. Build the ANN architecture
# --------------------------------------------------

model = keras.Sequential(
    [
        keras.Input(
            shape=(SEQUENCE_LENGTH,),
            dtype="int32",
            name="token_ids"
        ),

        layers.Embedding(
            input_dim=MAX_TOKENS,
            output_dim=EMBEDDING_DIM,
            mask_zero=True,
            name="word_embedding"
        ),

        layers.GlobalAveragePooling1D(
            name="global_average_pooling"
        ),

        layers.Dense(
            64,
            activation="relu",
            name="hidden_layer"
        ),

        layers.Dropout(
            0.3,
            name="dropout"
        ),

        layers.Dense(
            1,
            activation="sigmoid",
            name="fake_real_output"
        )
    ],
    name="truthlens_ann"
)


# --------------------------------------------------
# 3. Configure training
# --------------------------------------------------

model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# --------------------------------------------------
# 4. Display the architecture
# --------------------------------------------------

print("\nTRUTHLENS ANN ARCHITECTURE\n")
model.summary()

print("\nTotal parameters:", model.count_params())


# --------------------------------------------------
# 5. Save the initial, untrained model
# --------------------------------------------------

model_path = MODEL_DIR / "truthlens_ann_untrained.keras"

model.save(model_path)

print("\nModel architecture created successfully.")
print("Saved to:", model_path)
print("Status: UNTRAINED")
print("Output mapping: 0 = Fake, 1 = Real")

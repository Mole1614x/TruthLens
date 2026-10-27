from pathlib import Path

import keras
import keras_hub

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

PRESET = "distil_bert_base_en_uncased"
MAX_LENGTH = 256
NUM_CLASSES = 2

OUTPUT_DIR = Path("models")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 2. Load the pretrained DistilBERT backbone
# --------------------------------------------------

backbone = keras_hub.models.DistilBertBackbone.from_preset(
    PRESET
)

# --------------------------------------------------
# 3. Define model inputs
# --------------------------------------------------

input_ids = keras.Input(
    shape=(MAX_LENGTH,),
    dtype="int32",
    name="token_ids"
)

attention_mask = keras.Input(
    shape=(MAX_LENGTH,),
    dtype="int32",
    name="padding_mask"
)

# --------------------------------------------------
# 4. Pass inputs through DistilBERT
# --------------------------------------------------

backbone_outputs = backbone(
    {
        "token_ids": input_ids,
        "padding_mask": attention_mask
    }
)

# Select the first token's contextual representation.
# The backbone output is a tensor, not a dictionary.
cls_representation = backbone_outputs[:, 0, :]

# --------------------------------------------------
# 5. Add the classification head
# --------------------------------------------------

dropout = keras.layers.Dropout(0.2)(
    cls_representation
)

output = keras.layers.Dense(
    NUM_CLASSES,
    activation="softmax",
    name="fake_real_output"
)(dropout)

# --------------------------------------------------
# 6. Create the complete model
# --------------------------------------------------

model = keras.Model(
    inputs={
        "token_ids": input_ids,
        "padding_mask": attention_mask
    },
    outputs=output,
    name="truthlens_distilbert"
)

# --------------------------------------------------
# 7. Compile the model
# --------------------------------------------------

model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=2e-5
    ),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"]
)

# --------------------------------------------------
# 8. Display model information
# --------------------------------------------------

model.summary()

# --------------------------------------------------
# 9. Save the model
# --------------------------------------------------

model.save(
    OUTPUT_DIR / "truthlens_distilbert_untrained.keras"
)

print("\nDistilBERT model created successfully.")

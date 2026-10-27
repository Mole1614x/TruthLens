from pathlib import Path
from tensorflow import keras
from tensorflow.keras import layers

# Model configuration
VOCAB_SIZE = 20000
SEQUENCE_LENGTH = 300
EMBEDDING_DIM = 64
LSTM_UNITS = 64

# Build the BiLSTM architecture
model = keras.Sequential(
    [
        keras.Input(
            shape=(SEQUENCE_LENGTH,),
            dtype="int32",
            name="token_ids"
        ),

        layers.Embedding(
            input_dim=VOCAB_SIZE,
            output_dim=EMBEDDING_DIM,
            mask_zero=True,
            name="word_embedding"
        ),

        layers.Bidirectional(
            layers.LSTM(LSTM_UNITS),
            name="bilstm_layer"
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
    name="truthlens_bilstm"
)

# Configure the model for binary classification
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Display the architecture
model.summary()

# Save the untrained model
output_path = Path("models/truthlens_bilstm_untrained.keras")
output_path.parent.mkdir(parents=True, exist_ok=True)
model.save(output_path)

print("\nUntrained BiLSTM model saved to:", output_path)
print("Input shape:", model.input_shape)
print("Output shape:", model.output_shape)
print("Trainable parameters:", model.count_params())

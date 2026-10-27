import tensorflow as tf
from tensorflow import keras

# Load the trained LSTM model
model = keras.models.load_model(
    "models/truthlens_lstm.keras",
    compile=False
)

# Load the vectorizer fitted on training data
vectorizer_model = keras.models.load_model(
    "models/ann_text_vectorizer.keras",
    compile=False
)

# The saved vectorizer is wrapped in a Sequential model
text_vectorizer = vectorizer_model.layers[0]

print("TruthLens LSTM loaded successfully.")
print("Enter 'exit' to stop.")

while True:
    news_text = input("\nEnter news text: ").strip()

    if news_text.lower() == "exit":
        print("Exiting TruthLens.")
        break

    if not news_text:
        print("Please enter some news text.")
        continue

    # Convert text to token IDs
    token_ids = text_vectorizer(
        tf.constant([news_text])
    )

    # Predict the score
    score = float(
        model.predict(token_ids, verbose=0)[0][0]
    )

    # Convert score to a class
    if score >= 0.5:
        label = "Real"
    else:
        label = "Fake"

    print("\nPrediction:", label)
    print(f"Real score: {score:.4f}")
    print(f"Fake score: {1 - score:.4f}")
    print(
        "Note: This is a model prediction, "
        "not independent fact verification."
    )

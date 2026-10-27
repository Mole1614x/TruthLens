import numpy as np
import tensorflow as tf
from tensorflow import keras

MODEL_PATH = "models/truthlens_bilstm_best.keras"
VECTORIZER_PATH = "models/ann_text_vectorizer.keras"

# Load the trained model and existing vectorizer
model = keras.models.load_model(MODEL_PATH, compile=False)
vectorizer_model = keras.models.load_model(
    VECTORIZER_PATH,
    compile=False
)

# The saved vectorizer is wrapped in a Sequential model.
text_vectorizer = vectorizer_model.layers[0]

def predict_news(news_text):
    if not news_text.strip():
        raise ValueError("Please enter some news text.")

    # Convert the text into token IDs
    token_ids = text_vectorizer(
        tf.constant([news_text])
    )

    # Predict the probability of the Real class
    real_probability = float(
        model.predict(token_ids, verbose=0)[0][0]
    )

    # Apply the classification threshold
    predicted_label = (
        "Real" if real_probability >= 0.5 else "Fake"
    )

    return {
        "prediction": predicted_label,
        "real_probability": real_probability,
        "fake_probability": 1.0 - real_probability,
    }

print("TruthLens BiLSTM Prediction")
print("Enter 'exit' to quit.")

while True:
    news_text = input("\nEnter news text: ").strip()

    if news_text.lower() == "exit":
        print("Exiting TruthLens.")
        break

    try:
        result = predict_news(news_text)

        print("Predicted label:", result["prediction"])
        print(
            f"Real-class score: "
            f"{result['real_probability']:.4f}"
        )
        print(
            f"Fake-class score: "
            f"{result['fake_probability']:.4f}"
        )

    except ValueError as error:
        print(error)

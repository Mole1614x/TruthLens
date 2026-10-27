import numpy as np
import tensorflow as tf
from pathlib import Path

# Paths to the saved model and text vectorizer
MODEL_PATH = Path("models/truthlens_ann.keras")
VECTORIZER_PATH = Path("models/ann_text_vectorizer.keras")

# Check that the required files exist
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")

if not VECTORIZER_PATH.exists():
    raise FileNotFoundError(f"Text vectorizer not found: {VECTORIZER_PATH}")

# Load the trained ANN and text vectorizer
print("Loading TruthLens ANN model...")
model = tf.keras.models.load_model(MODEL_PATH)

print("Loading text vectorizer...")
vectorizer = tf.keras.models.load_model(VECTORIZER_PATH)

print("\nTruthLens is ready to analyse news.")
print("Enter 'exit' to stop.\n")


def predict_news(news_text):
    """Predict whether a news article is Fake or Real."""

    if not news_text.strip():
        print("Please enter some news text.")
        return

    # Step 1: Convert raw text into token IDs
    text_vectorizer = vectorizer.layers[0]
    token_ids = text_vectorizer(tf.constant([news_text]))

    # Step 2: Get the ANN output score
    prediction_score = float(
        model.predict(token_ids, verbose=0)[0][0]
    )

    # Step 3: Convert the score into a predicted class
    if prediction_score >= 0.5:
        label = "Real"
    else:
        label = "Fake"

    # Display the prediction
    print("\n========== TRUTHLENS RESULT ==========")
    print(f"Prediction: {label}")
    print(f"Model score for Real: {prediction_score:.4f}")
    print(f"Model score for Fake: {1 - prediction_score:.4f}")
    print("======================================\n")


while True:
    news = input("Enter a news headline or article: ").strip()

    if news.lower() == "exit":
        print("TruthLens session ended.")
        break

    predict_news(news)

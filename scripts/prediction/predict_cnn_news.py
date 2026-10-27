import tensorflow as tf
from tensorflow import keras

MODEL_PATH = "models/truthlens_cnn_best.keras"
VECTORIZER_PATH = "models/ann_text_vectorizer.keras"

# Load the trained CNN model.
model = keras.models.load_model(MODEL_PATH)

# Load the vectorizer fitted on training data.
vectorizer_model = keras.models.load_model(VECTORIZER_PATH)
vectorizer = vectorizer_model.layers[0]

# Enter the news text to classify.
news_text = input("Enter news text: ").strip()

if not news_text:
    print("Please enter some news text.")
else:
    # Convert the text into token IDs.
    token_ids = vectorizer(tf.constant([news_text]))

    # Obtain the model's sigmoid output.
    prediction_score = float(
        model.predict(token_ids, verbose=0)[0][0]
    )

    # Convert the score into a class prediction.
    if prediction_score >= 0.5:
        predicted_class = "Real"
    else:
        predicted_class = "Fake"

    print("\n--- TruthLens CNN Prediction ---")
    print("Predicted class:", predicted_class)
    print(f"Real-class score: {prediction_score:.4f}")
    print(f"Fake-class score: {1 - prediction_score:.4f}")

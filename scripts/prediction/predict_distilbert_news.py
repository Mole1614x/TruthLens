import numpy as np
import tensorflow as tf
import keras_hub

from pathlib import Path


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

MODEL_PATH = Path("models/truthlens_distilbert.keras")

PRESET = "distil_bert_base_en_uncased"

MAX_LENGTH = 256

# Label mapping
LABELS = {
    0: "Fake",
    1: "Real"
}


# --------------------------------------------------
# 2. Check model file
# --------------------------------------------------

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Trained DistilBERT model not found: {MODEL_PATH}"
    )


# --------------------------------------------------
# 3. Load the tokenizer
# --------------------------------------------------

print("Loading DistilBERT tokenizer...")

tokenizer = keras_hub.models.DistilBertTokenizer.from_preset(
    PRESET
)

# Get special token IDs
cls_id = tokenizer.token_to_id("[CLS]")
sep_id = tokenizer.token_to_id("[SEP]")
pad_id = tokenizer.token_to_id("[PAD]")


# --------------------------------------------------
# 4. Load the trained model
# --------------------------------------------------

print("Loading trained TruthLens DistilBERT model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("\nTruthLens DistilBERT is ready.")
print("Enter 'exit' to stop.\n")


# --------------------------------------------------
# 5. Prepare input text
# --------------------------------------------------

def prepare_text(news_text):

    # Convert input into a string
    news_text = str(news_text)

    # Tokenize the text
    token_ids = tokenizer.tokenize(
        news_text
    ).numpy().tolist()

    # Reserve space for CLS and SEP
    token_ids = token_ids[:MAX_LENGTH - 2]

    # Add special tokens
    token_ids = (
        [cls_id]
        + token_ids
        + [sep_id]
    )

    # Create attention mask
    attention_mask = [1] * len(token_ids)

    # Calculate padding
    padding_length = MAX_LENGTH - len(token_ids)

    # Add padding tokens
    token_ids += [pad_id] * padding_length

    # Add zeros to attention mask
    attention_mask += [0] * padding_length

    # Convert to NumPy arrays
    input_ids = np.array(
        [token_ids],
        dtype=np.int32
    )

    attention_mask = np.array(
        [attention_mask],
        dtype=np.int32
    )

    return input_ids, attention_mask


# --------------------------------------------------
# 6. Predict news
# --------------------------------------------------

def predict_news(news_text):

    # Handle empty input
    if not news_text.strip():

        print("Please enter some news text.")
        return

    # Step 1: Prepare input
    input_ids, attention_mask = prepare_text(
        news_text
    )

    # Step 2: Get model probabilities
    prediction_scores = model.predict(
        {
            "token_ids": input_ids,
            "padding_mask": attention_mask
        },
        verbose=0
    )[0]

    # Step 3: Find predicted class
    predicted_class = int(
        np.argmax(prediction_scores)
    )

    predicted_label = LABELS[predicted_class]

    # Step 4: Get class scores
    fake_score = float(prediction_scores[0])
    real_score = float(prediction_scores[1])

    # Step 5: Display result
    print("\n========== TRUTHLENS RESULT ==========")

    print(f"Prediction: {predicted_label}")

    print(
        f"Model score for Fake: {fake_score:.4f}"
    )

    print(
        f"Model score for Real: {real_score:.4f}"
    )

    print("======================================\n")


# --------------------------------------------------
# 7. Interactive loop
# --------------------------------------------------

while True:

    news = input(
        "Enter a news headline or article: "
    ).strip()

    if news.lower() == "exit":

        print("TruthLens session ended.")
        break

    predict_news(news)
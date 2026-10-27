
import csv
from datetime import datetime
from pathlib import Path

import numpy as np
import keras
import keras_hub

from lime.lime_text import LimeTextExplainer


# -----------------------------------------
# 1. Configuration
# -----------------------------------------

MODEL_PATH = Path("models/truthlens_distilbert.keras")
REPORT_DIR = Path("reports")
REPORT_PATH = REPORT_DIR / "lime_explanations.csv"

MAX_LENGTH = 256
CLASS_NAMES = ["Fake News", "Real News"]


# -----------------------------------------
# 2. Load the trained model and tokenizer
# -----------------------------------------

print("Loading the TruthLens model...")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Trained model not found: {MODEL_PATH}\n"
        "Run this script from your TruthLens project directory."
    )

model = keras.models.load_model(MODEL_PATH)

tokenizer = keras_hub.models.DistilBertTokenizer.from_preset(
    "distil_bert_base_en_uncased"
)

explainer = LimeTextExplainer(
    class_names=CLASS_NAMES,
    random_state=42
)

print("Model and tokenizer loaded successfully.")


# -----------------------------------------
# 3. Preprocess text
# -----------------------------------------

def preprocess_text(text):
    """Convert text into token IDs and an attention mask."""

    token_ids = tokenizer.tokenize([text])[0]

    # Support either tensor-like output or a Python list.
    if hasattr(token_ids, "numpy"):
        token_ids = token_ids.numpy().tolist()
    else:
        token_ids = list(token_ids)

    # Reserve space for [CLS] and [SEP].
    token_ids = token_ids[:MAX_LENGTH - 2]

    cls_id = tokenizer.token_to_id("[CLS]")
    sep_id = tokenizer.token_to_id("[SEP]")
    pad_id = tokenizer.token_to_id("[PAD]")

    token_ids = [cls_id] + token_ids + [sep_id]

    padding_length = MAX_LENGTH - len(token_ids)

    padding_mask = (
        [1] * len(token_ids)
        + [0] * padding_length
    )

    token_ids = token_ids + [pad_id] * padding_length

    return token_ids, padding_mask


# -----------------------------------------
# 4. Predict class scores
# -----------------------------------------

def predict_probabilities(texts):
    """Return Fake News and Real News scores for each text."""

    all_token_ids = []
    all_padding_masks = []

    for text in texts:
        token_ids, padding_mask = preprocess_text(text)

        all_token_ids.append(token_ids)
        all_padding_masks.append(padding_mask)

    inputs = {
        "token_ids": np.array(
            all_token_ids, dtype=np.int32
        ),
        "padding_mask": np.array(
            all_padding_masks, dtype=np.int32
        ),
    }

    probabilities = model.predict(inputs, verbose=0)

    return np.asarray(probabilities)


# -----------------------------------------
# 5. Save LIME explanations to CSV
# -----------------------------------------

def save_explanation_report(
    article,
    predicted_label,
    probabilities,
    explanation
):
    """Append prediction details and LIME weights to a CSV."""

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    file_exists = (
        REPORT_PATH.exists()
        and REPORT_PATH.stat().st_size > 0
    )

    columns = [
        "timestamp",
        "article",
        "predicted_label",
        "fake_score",
        "real_score",
        "explained_class",
        "word",
        "weight",
    ]

    with REPORT_PATH.open(
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=columns
        )

        if not file_exists:
            writer.writeheader()

        timestamp = datetime.now().astimezone().isoformat()

        for label, class_name in [
            (0, "Fake News"),
            (1, "Real News"),
        ]:
            for word, weight in explanation.as_list(
                label=label
            ):
                writer.writerow({
                    "timestamp": timestamp,
                    "article": article,
                    "predicted_label": predicted_label,
                    "fake_score": float(probabilities[0]),
                    "real_score": float(probabilities[1]),
                    "explained_class": class_name,
                    "word": word,
                    "weight": float(weight),
                })

    return REPORT_PATH


# -----------------------------------------
# 6. Main program
# -----------------------------------------

def main():
    article = input(
        "\nEnter a news article to explain:\n"
    ).strip()

    if not article:
        print("Please enter some news text.")
        return

    print("\nAnalyzing article. Please wait...")

    # Get the original model prediction.
    probabilities = predict_probabilities([article])[0]
    predicted_class = int(np.argmax(probabilities))
    predicted_label = CLASS_NAMES[predicted_class]

    print("\n--- TruthLens Prediction ---")
    print("Predicted label:", predicted_label)
    print(f"Fake News score: {probabilities[0]:.4f}")
    print(f"Real News score: {probabilities[1]:.4f}")

    # Generate the local LIME explanation.
    print("\nGenerating LIME explanation...")

    explanation = explainer.explain_instance(
        article,
        predict_probabilities,
        labels=[0, 1],
        num_features=8,
        num_samples=100
    )

    print("\n--- Words influencing the Fake News explanation ---")

    for word, weight in explanation.as_list(label=0):
        print(f"{word}: {weight:+.6f}")

    print("\n--- Words influencing the Real News explanation ---")

    for word, weight in explanation.as_list(label=1):
        print(f"{word}: {weight:+.6f}")

    # Save prediction and explanation details.
    report_path = save_explanation_report(
        article,
        predicted_label,
        probabilities,
        explanation
    )

    print("\n--- Report Saved ---")
    print("Location:", report_path.resolve())

    print(
        "\nNote: TruthLens classifies text patterns; "
        "it does not independently verify factual claims."
    )


if __name__ == "__main__":
    main()

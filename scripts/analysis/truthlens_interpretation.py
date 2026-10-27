
import csv
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
import keras
import keras_hub

from lime.lime_text import LimeTextExplainer


# Project configuration
MODEL_PATH = Path("models/truthlens_distilbert.keras")
REPORT_DIR = Path("reports")
MAX_LENGTH = 256

CLASS_NAMES = ["Fake News", "Real News"]


class TruthLensInterpreter:

    def __init__(self):
        # Load the trained model only once.
        self.model = keras.models.load_model(MODEL_PATH)

        self.tokenizer = keras_hub.models.DistilBertTokenizer.from_preset(
            "distil_bert_base_en_uncased"
        )

        self.explainer = LimeTextExplainer(
            class_names=CLASS_NAMES,
            random_state=42
        )

    def preprocess_text(self, text):
        token_ids = self.tokenizer.tokenize([text])[0]

        if hasattr(token_ids, "numpy"):
            token_ids = token_ids.numpy().tolist()
        else:
            token_ids = list(token_ids)

        token_ids = token_ids[:MAX_LENGTH - 2]

        cls_id = self.tokenizer.token_to_id("[CLS]")
        sep_id = self.tokenizer.token_to_id("[SEP]")
        pad_id = self.tokenizer.token_to_id("[PAD]")

        token_ids = [cls_id] + token_ids + [sep_id]

        padding_length = MAX_LENGTH - len(token_ids)

        padding_mask = (
            [1] * len(token_ids)
            + [0] * padding_length
        )

        token_ids += [pad_id] * padding_length

        return token_ids, padding_mask

    def predict_probabilities(self, texts):
        all_token_ids = []
        all_padding_masks = []

        for text in texts:
            token_ids, padding_mask = self.preprocess_text(text)

            all_token_ids.append(token_ids)
            all_padding_masks.append(padding_mask)

        inputs = {
            "token_ids": np.array(all_token_ids, dtype=np.int32),
            "padding_mask": np.array(
                all_padding_masks, dtype=np.int32
            ),
        }

        return self.model.predict(inputs, verbose=0)

    def analyze(self, article, num_samples=100):
        if not article or not article.strip():
            raise ValueError("Please enter some news text.")

        probabilities = self.predict_probabilities([article])[0]
        predicted_class = int(np.argmax(probabilities))

        explanation = self.explainer.explain_instance(
            article,
            self.predict_probabilities,
            labels=[0, 1],
            num_features=8,
            num_samples=num_samples
        )

        return {
            "article": article,
            "predicted_label": CLASS_NAMES[predicted_class],
            "fake_score": float(probabilities[0]),
            "real_score": float(probabilities[1]),
            "fake_explanation": explanation.as_list(label=0),
            "real_explanation": explanation.as_list(label=1),
        }


def save_explanation_report(result):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    report_path = REPORT_DIR / "lime_explanations.csv"

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

    file_exists = report_path.exists() and report_path.stat().st_size > 0

    with report_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)

        if not file_exists:
            writer.writeheader()

        timestamp = datetime.now().astimezone().isoformat()

        for class_name, explanation_key in [
            ("Fake News", "fake_explanation"),
            ("Real News", "real_explanation"),
        ]:
            for word, weight in result[explanation_key]:
                writer.writerow({
                    "timestamp": timestamp,
                    "article": result["article"],
                    "predicted_label": result["predicted_label"],
                    "fake_score": result["fake_score"],
                    "real_score": result["real_score"],
                    "explained_class": class_name,
                    "word": word,
                    "weight": weight,
                })

    return report_path

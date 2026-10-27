"""
Evaluate all six saved TruthLens models on the same held-out test split.

Run from the TruthLens project root:
    python evaluate_all_models.py

Labels: 0 = Fake, 1 = Real.
This script loads existing models only; it does not train or tune models.
"""

from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import keras_hub

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(".")
TEST_PATH = ROOT / "data/processed/splits/test.csv"
REPORTS_DIR = ROOT / "reports"
MODELS_DIR = ROOT / "models"

ANN_PATH = MODELS_DIR / "truthlens_ann.keras"
LSTM_PATH = MODELS_DIR / "truthlens_lstm.keras"
BILSTM_PATH = MODELS_DIR / "truthlens_bilstm_best.keras"
CNN_PATH = MODELS_DIR / "truthlens_cnn_best.keras"
DISTILBERT_PATH = MODELS_DIR / "truthlens_distilbert.keras"
VECTORIZER_PATH = MODELS_DIR / "ann_text_vectorizer.keras"
TFIDF_PATH = MODELS_DIR / "tfidf_vectorizer.pkl"
SVM_PATH = MODELS_DIR / "tuned_linear_svm.pkl"

DISTILBERT_PRESET = "distil_bert_base_en_uncased"
DISTILBERT_MAX_LENGTH = 256
BATCH_SIZE = 32
LABEL_NAMES = ["Fake", "Real"]


def require_files(paths):
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "These required files were not found:\n  - "
            + "\n  - ".join(missing)
            + "\nRun this script from your TruthLens project root."
        )


def load_test_data():
    if not TEST_PATH.exists():
        raise FileNotFoundError(f"Test split not found: {TEST_PATH}")

    data = pd.read_csv(TEST_PATH)
    required_columns = {"label", "combined_text"}
    missing = required_columns - set(data.columns)
    if missing:
        raise ValueError(f"Test CSV is missing columns: {sorted(missing)}")

    data["combined_text"] = data["combined_text"].fillna("").astype(str)
    data["label"] = data["label"].astype(int)

    invalid_labels = sorted(set(data["label"].unique()) - {0, 1})
    if invalid_labels:
        raise ValueError(f"Expected labels 0/1, found: {invalid_labels}")

    if data.empty:
        raise ValueError("The test split is empty.")

    return data


def predict_sequence_model(model, text_vectorizer, texts, model_name):
    """Predict for ANN/LSTM/BiLSTM/CNN using the saved shared vectorizer."""
    print(f"  Vectorizing and predicting with {model_name}...")
    predictions = []

    for start in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[start : start + BATCH_SIZE]
        token_ids = text_vectorizer(tf.constant(batch_texts))
        scores = np.asarray(model.predict(token_ids, verbose=0))

        # These four saved models produce a sigmoid score for class 1 (Real).
        scores = scores.reshape(-1)
        batch_predictions = (scores >= 0.5).astype(int)
        predictions.extend(batch_predictions.tolist())

    return np.asarray(predictions, dtype=int)


def prepare_distilbert_batch(texts, tokenizer, cls_id, sep_id, pad_id):
    """Match predict_distilbert_news.py preprocessing for a batch of texts."""
    token_id_rows = []
    mask_rows = []

    for text in texts:
        token_ids = tokenizer.tokenize(str(text)).numpy().tolist()
        token_ids = token_ids[: DISTILBERT_MAX_LENGTH - 2]
        token_ids = [cls_id] + token_ids + [sep_id]

        attention_mask = [1] * len(token_ids)
        padding_length = DISTILBERT_MAX_LENGTH - len(token_ids)

        token_ids += [pad_id] * padding_length
        attention_mask += [0] * padding_length

        token_id_rows.append(token_ids)
        mask_rows.append(attention_mask)

    return (
        np.asarray(token_id_rows, dtype=np.int32),
        np.asarray(mask_rows, dtype=np.int32),
    )


def predict_distilbert(model, texts):
    print("  Loading DistilBERT tokenizer...")
    tokenizer = keras_hub.models.DistilBertTokenizer.from_preset(
        DISTILBERT_PRESET
    )
    cls_id = tokenizer.token_to_id("[CLS]")
    sep_id = tokenizer.token_to_id("[SEP]")
    pad_id = tokenizer.token_to_id("[PAD]")

    predictions = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[start : start + BATCH_SIZE]
        input_ids, padding_mask = prepare_distilbert_batch(
            batch_texts, tokenizer, cls_id, sep_id, pad_id
        )
        scores = np.asarray(
            model.predict(
                {"token_ids": input_ids, "padding_mask": padding_mask},
                verbose=0,
            )
        )

        # The existing prediction script uses argmax over [Fake, Real] scores.
        if scores.ndim == 1:
            batch_predictions = (scores >= 0.5).astype(int)
        elif scores.shape[-1] == 1:
            batch_predictions = (scores.reshape(-1) >= 0.5).astype(int)
        else:
            batch_predictions = np.argmax(scores, axis=1).astype(int)

        predictions.extend(batch_predictions.tolist())

        completed = min(start + len(batch_texts), len(texts))
        print(f"    DistilBERT processed {completed:,}/{len(texts):,}")

    return np.asarray(predictions, dtype=int)


def evaluate_model(name, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    summary = {
        "model": name,
        "test_samples": int(len(y_true)),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_weighted": precision_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "recall_weighted": recall_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "f1_weighted": f1_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "correct": int(np.sum(y_true == y_pred)),
        "incorrect": int(np.sum(y_true != y_pred)),
        "true_fake": int(tn),
        "fake_predicted_real": int(fp),
        "real_predicted_fake": int(fn),
        "true_real": int(tp),
    }

    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=LABEL_NAMES,
        output_dict=True,
        zero_division=0,
    )
    return summary, report


def main():
    require_files(
        [
            ANN_PATH,
            LSTM_PATH,
            BILSTM_PATH,
            CNN_PATH,
            DISTILBERT_PATH,
            VECTORIZER_PATH,
            TFIDF_PATH,
            SVM_PATH,
            TEST_PATH,
        ]
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    data = load_test_data()
    texts = data["combined_text"].tolist()
    y_true = data["label"].to_numpy(dtype=int)

    print(f"\nTest rows: {len(data):,}")
    print(f"Label counts: {data['label'].value_counts().sort_index().to_dict()}")
    print("Labels: 0 = Fake, 1 = Real")
    print("\nLoading shared Keras text vectorizer...")
    vectorizer_model = tf.keras.models.load_model(
        VECTORIZER_PATH, compile=False
    )
    text_vectorizer = vectorizer_model.layers[0]

    model_paths = [
        ("ANN", ANN_PATH),
        ("LSTM", LSTM_PATH),
        ("BiLSTM", BILSTM_PATH),
        ("CNN", CNN_PATH),
    ]

    predictions_by_model = {}
    summaries = []
    class_reports = {}

    for name, model_path in model_paths:
        print(f"\nLoading {name}: {model_path}")
        model = tf.keras.models.load_model(model_path, compile=False)
        y_pred = predict_sequence_model(model, text_vectorizer, texts, name)
        predictions_by_model[name] = y_pred
        summary, report = evaluate_model(name, y_true, y_pred)
        summaries.append(summary)
        class_reports[name] = report
        del model

    print(f"\nLoading DistilBERT: {DISTILBERT_PATH}")
    distilbert_model = tf.keras.models.load_model(
        DISTILBERT_PATH, compile=False
    )
    y_pred = predict_distilbert(distilbert_model, texts)
    predictions_by_model["DistilBERT"] = y_pred
    summary, report = evaluate_model("DistilBERT", y_true, y_pred)
    summaries.append(summary)
    class_reports["DistilBERT"] = report
    del distilbert_model

    print("\nLoading TF-IDF vectorizer and tuned Linear SVM...")
    tfidf_vectorizer = joblib.load(TFIDF_PATH)
    svm_model = joblib.load(SVM_PATH)
    tfidf_features = tfidf_vectorizer.transform(texts)
    predictions_by_model["Linear SVM"] = svm_model.predict(
        tfidf_features
    ).astype(int)
    summary, report = evaluate_model(
        "Linear SVM", y_true, predictions_by_model["Linear SVM"]
    )
    summaries.append(summary)
    class_reports["Linear SVM"] = report

    summary_df = pd.DataFrame(summaries).sort_values(
        "f1_weighted", ascending=False
    )
    summary_path = REPORTS_DIR / "all_models_test_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    # Save each test row's true label and each model's prediction.
    prediction_df = pd.DataFrame()
    if "source_row_id" in data.columns:
        prediction_df["source_row_id"] = data["source_row_id"]
    if "title" in data.columns:
        prediction_df["title"] = data["title"].fillna("").astype(str)
    prediction_df["true_label"] = y_true
    prediction_df["true_name"] = np.where(y_true == 1, "Real", "Fake")

    for name, y_pred in predictions_by_model.items():
        safe_name = name.lower().replace(" ", "_")
        prediction_df[f"{safe_name}_prediction"] = y_pred
        prediction_df[f"{safe_name}_label"] = np.where(
            y_pred == 1, "Real", "Fake"
        )
        prediction_df[f"{safe_name}_correct"] = y_pred == y_true

    predictions_path = REPORTS_DIR / "all_models_test_predictions.csv"
    prediction_df.to_csv(predictions_path, index=False)

    report_path = REPORTS_DIR / "all_models_classification_reports.json"
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(class_reports, file, indent=2)

    print("\n" + "=" * 95)
    print("HELD-OUT TEST RESULTS (sorted by weighted F1)")
    print("=" * 95)
    display_columns = [
        "model",
        "accuracy",
        "precision_weighted",
        "recall_weighted",
        "f1_weighted",
        "correct",
        "incorrect",
    ]
    print(
        summary_df[display_columns]
        .to_string(
            index=False,
            formatters={
                "accuracy": "{:.4f}".format,
                "precision_weighted": "{:.4f}".format,
                "recall_weighted": "{:.4f}".format,
                "f1_weighted": "{:.4f}".format,
            },
        )
    )

    print("\nConfusion matrices (rows = actual [Fake, Real], columns = predicted [Fake, Real]):")
    for _, row in summary_df.iterrows():
        print(
            f"\n{row['model']}:\n"
            f"[[{int(row['true_fake'])}, {int(row['fake_predicted_real'])}],\n"
            f" [{int(row['real_predicted_fake'])}, {int(row['true_real'])}]]"
        )

    print("\nSaved reports:")
    print(f"  Summary:      {summary_path}")
    print(f"  Predictions:  {predictions_path}")
    print(f"  Class reports:{report_path}")
    print("\nEvaluation complete. No models were trained or tuned.")


if __name__ == "__main__":
    main()

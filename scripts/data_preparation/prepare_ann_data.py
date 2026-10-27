from pathlib import Path
import gc
import json

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import TextVectorization


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
SPLIT_DIR = PROJECT_DIR / "data" / "processed" / "splits"
OUTPUT_DIR = PROJECT_DIR / "data" / "processed" / "ann"
MODEL_DIR = PROJECT_DIR / "models"

TRAIN_FILE = SPLIT_DIR / "train.csv"
VALIDATION_FILE = SPLIT_DIR / "validation.csv"

MAX_TOKENS = 20_000
SEQUENCE_LENGTH = 300
BATCH_SIZE = 256

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Load only the columns we need
# --------------------------------------------------

def load_split(file_path):
    print(f"\nLoading: {file_path.name}")

    df = pd.read_csv(
        file_path,
        usecols=["combined_text", "label"]
    )

    df["combined_text"] = (
        df["combined_text"]
        .fillna("")
        .astype(str)
    )

    if df["label"].isna().any():
        raise ValueError(f"Missing labels found in {file_path.name}")

    labels = df["label"].to_numpy(dtype=np.int32)

    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Labels must be 0 (Fake) or 1 (Real).")

    texts = df["combined_text"].to_numpy(dtype=str)

    print("Number of articles:", len(df))
    print("Text column: combined_text")
    print("Label counts:", df["label"].value_counts().sort_index().to_dict())

    return texts, labels


# --------------------------------------------------
# 3. Load training data
# --------------------------------------------------

train_texts, y_train = load_split(TRAIN_FILE)


# --------------------------------------------------
# 4. Create the text vectorizer
# --------------------------------------------------

vectorizer = TextVectorization(
    max_tokens=MAX_TOKENS,
    output_mode="int",
    output_sequence_length=SEQUENCE_LENGTH,
    standardize="lower_and_strip_punctuation",
    split="whitespace"
)

# IMPORTANT:
# Learn the vocabulary from training text only.
train_text_dataset = tf.data.Dataset.from_tensor_slices(
    train_texts
).batch(BATCH_SIZE)

print("\nLearning vocabulary from training data...")
vectorizer.adapt(train_text_dataset)

vocabulary = vectorizer.get_vocabulary()

print("Vocabulary size:", len(vocabulary))
print("Sequence length:", SEQUENCE_LENGTH)


# --------------------------------------------------
# 5. Convert training text into token IDs
# --------------------------------------------------

print("\nVectorizing training data...")

X_train = vectorizer(
    tf.constant(train_texts),
    training=False
).numpy().astype(np.int32)

print("Training sequence shape:", X_train.shape)
print("Training label shape:", y_train.shape)

np.savez_compressed(
    OUTPUT_DIR / "train_sequences.npz",
    X=X_train,
    y=y_train
)

print("Saved train_sequences.npz")

del train_text_dataset, train_texts, X_train, y_train
gc.collect()


# --------------------------------------------------
# 6. Save the learned vectorizer and vocabulary
# --------------------------------------------------

vectorizer_model = keras.Sequential([
    keras.Input(shape=(), dtype=tf.string),
    vectorizer
])

vectorizer_model.save(
    MODEL_DIR / "ann_text_vectorizer.keras"
)

with open(OUTPUT_DIR / "ann_vocabulary.json", "w", encoding="utf-8") as f:
    json.dump(vocabulary, f, ensure_ascii=False)

print("Saved ANN text vectorizer and vocabulary.")


# --------------------------------------------------
# 7. Transform validation data
# --------------------------------------------------

# Reuse the SAME vectorizer.
# Do not adapt it on validation data.

validation_texts, y_validation = load_split(VALIDATION_FILE)

print("\nVectorizing validation data...")

X_validation = vectorizer(
    tf.constant(validation_texts),
    training=False
).numpy().astype(np.int32)

print("Validation sequence shape:", X_validation.shape)
print("Validation label shape:", y_validation.shape)

np.savez_compressed(
    OUTPUT_DIR / "validation_sequences.npz",
    X=X_validation,
    y=y_validation
)

print("Saved validation_sequences.npz")


# --------------------------------------------------
# 8. Final checks
# --------------------------------------------------

assert X_validation.shape[1] == SEQUENCE_LENGTH
assert len(X_validation) == len(y_validation)
assert len(vocabulary) <= MAX_TOKENS

print("\nPreprocessing completed successfully.")
print("Output directory:", OUTPUT_DIR)
print("Label mapping: 0 = Fake, 1 = Real")
print("The test split was not used.")

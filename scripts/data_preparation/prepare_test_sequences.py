from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

PROJECT_DIR = Path(__file__).resolve().parent

TEST_FILE = PROJECT_DIR / "data" / "processed" / "splits" / "test.csv"
VECTORIZER_FILE = PROJECT_DIR / "models" / "ann_text_vectorizer.keras"
OUTPUT_FILE = PROJECT_DIR / "data" / "processed" / "ann" / "test_sequences.npz"

print("Loading test split...")
df = pd.read_csv(TEST_FILE, usecols=["combined_text", "label"])

df["combined_text"] = df["combined_text"].fillna("").astype(str)

if df["label"].isna().any():
    raise ValueError("Test dataset contains missing labels.")

texts = df["combined_text"].to_numpy(dtype=str)
labels = df["label"].to_numpy(dtype=np.int32)

if not np.isin(labels, [0, 1]).all():
    raise ValueError("Labels must be 0 (Fake) or 1 (Real).")

print("Test articles:", len(texts))
print("Label counts:", dict(zip(*np.unique(labels, return_counts=True))))

print("\nLoading the existing training vectorizer...")
vectorizer_model = keras.models.load_model(VECTORIZER_FILE, compile=False)

print("Converting test text into sequences...")
X_test = vectorizer_model(tf.constant(texts), training=False).numpy()
X_test = X_test.astype(np.int32)

assert X_test.shape == (len(labels), 300)
assert len(X_test) == len(labels)

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

np.savez_compressed(
    OUTPUT_FILE,
    X=X_test,
    y=labels
)

print("\nTest preprocessing completed.")
print("Test sequence shape:", X_test.shape)
print("Test label shape:", labels.shape)
print("Saved to:", OUTPUT_FILE)
print("Training and validation files were not modified.")

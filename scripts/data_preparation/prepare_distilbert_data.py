from pathlib import Path

import numpy as np
import pandas as pd
import keras_hub

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

TRAIN_PATH = "data/processed/splits/train.csv"
VALIDATION_PATH = "data/processed/splits/validation.csv"

OUTPUT_DIR = Path("data/processed/distilbert")

MAX_LENGTH = 256

PRESET = "distil_bert_base_en_uncased"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 2. Load the datasets
# --------------------------------------------------

train_df = pd.read_csv(TRAIN_PATH)
validation_df = pd.read_csv(VALIDATION_PATH)

print("Training rows:", len(train_df))
print("Validation rows:", len(validation_df))

# --------------------------------------------------
# 3. Load the pretrained tokenizer
# --------------------------------------------------

tokenizer = keras_hub.models.DistilBertTokenizer.from_preset(
    PRESET
)

# Obtain the special token IDs from the tokenizer.
cls_id = tokenizer.token_to_id("[CLS]")
sep_id = tokenizer.token_to_id("[SEP]")
pad_id = tokenizer.token_to_id("[PAD]")

print("\nSpecial token IDs:")
print("CLS:", cls_id)
print("SEP:", sep_id)
print("PAD:", pad_id)

# --------------------------------------------------
# 4. Prepare one text example
# --------------------------------------------------

def prepare_text(text):
    # Handle missing text.
    if pd.isna(text):
        text = ""

    text = str(text)

    # Convert the text into token IDs.
    token_ids = tokenizer.tokenize(text).numpy().tolist()

    # Reserve two positions for special tokens.
    token_ids = token_ids[:MAX_LENGTH - 2]

    # Add CLS at the beginning and SEP at the end.
    token_ids = [cls_id] + token_ids + [sep_id]

    # Create an attention mask.
    attention_mask = [1] * len(token_ids)

    # Pad shorter sequences to MAX_LENGTH.
    padding_length = MAX_LENGTH - len(token_ids)

    token_ids += [pad_id] * padding_length
    attention_mask += [0] * padding_length

    return token_ids, attention_mask

# --------------------------------------------------
# 5. Process a complete dataset
# --------------------------------------------------

def prepare_dataset(dataframe, dataset_name):
    all_input_ids = []
    all_attention_masks = []

    texts = dataframe["combined_text"].fillna("").tolist()

    for index, text in enumerate(texts):
        input_ids, attention_mask = prepare_text(text)

        all_input_ids.append(input_ids)
        all_attention_masks.append(attention_mask)

        if (index + 1) % 5000 == 0:
            print(
                f"{dataset_name}: "
                f"Processed {index + 1}/{len(texts)} rows"
            )

    input_ids_array = np.array(
        all_input_ids,
        dtype=np.int32
    )

    attention_mask_array = np.array(
        all_attention_masks,
        dtype=np.int32
    )

    labels_array = dataframe["label"].to_numpy(
        dtype=np.int32
    )

    return (
        input_ids_array,
        attention_mask_array,
        labels_array
    )

# --------------------------------------------------
# 6. Prepare training data
# --------------------------------------------------

print("\nPreparing training data...")

train_input_ids, train_attention_masks, train_labels = (
    prepare_dataset(train_df, "Training")
)

# --------------------------------------------------
# 7. Prepare validation data
# --------------------------------------------------

print("\nPreparing validation data...")

val_input_ids, val_attention_masks, val_labels = (
    prepare_dataset(validation_df, "Validation")
)

# --------------------------------------------------
# 8. Save processed arrays
# --------------------------------------------------

np.savez_compressed(
    OUTPUT_DIR / "train.npz",
    input_ids=train_input_ids,
    attention_mask=train_attention_masks,
    labels=train_labels
)

np.savez_compressed(
    OUTPUT_DIR / "validation.npz",
    input_ids=val_input_ids,
    attention_mask=val_attention_masks,
    labels=val_labels
)

# --------------------------------------------------
# 9. Display shapes
# --------------------------------------------------

print("\n--- DistilBERT Data Preparation Complete ---")

print("Training input IDs:", train_input_ids.shape)
print("Training attention masks:", train_attention_masks.shape)
print("Training labels:", train_labels.shape)

print("\nValidation input IDs:", val_input_ids.shape)
print("Validation attention masks:", val_attention_masks.shape)
print("Validation labels:", val_labels.shape)

print("\nSaved files:")
print(OUTPUT_DIR / "train.npz")
print(OUTPUT_DIR / "validation.npz")

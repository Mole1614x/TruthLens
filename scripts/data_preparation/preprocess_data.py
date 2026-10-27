
from pathlib import Path

import pandas as pd


DATA_PATH = Path("data/raw/WELFake_Dataset.csv")
OUTPUT_PATH = Path("data/processed/truthlens_cleaned.csv")


def main():
    # 1. Load the original dataset.
    df = pd.read_csv(DATA_PATH)

    print("========== ORIGINAL DATASET ==========")
    print("Rows:", len(df))
    print("Columns:", df.columns.tolist())

    # 2. Preserve original row positions for traceability.
    df["source_row_id"] = range(len(df))

    # 3. Report missing values before cleaning.
    print("\n========== MISSING VALUES BEFORE CLEANING ==========")
    print(df[["title", "text", "label"]].isna().sum())

    # 4. Replace missing text fields with empty strings.
    # Keep rows if either the title or article text is available.
    df["title"] = df["title"].fillna("").astype(str).str.strip()
    df["text"] = df["text"].fillna("").astype(str).str.strip()

    # 5. Remove records that contain no usable text at all.
    df["combined_text"] = (
        df["title"] + " " + df["text"]
    ).str.strip()

    empty_text_mask = df["combined_text"].eq("")
    removed_rows = int(empty_text_mask.sum())

    df = df.loc[~empty_text_mask].copy()

    # 6. Keep only the columns needed for the next stages.
    # The original Unnamed: 0 column is not a predictive feature.
    df = df[
        ["source_row_id", "title", "text", "combined_text", "label"]
    ].copy()

    # 7. Create the processed-data directory if needed.
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # 8. Save the cleaned dataset.
    df.to_csv(OUTPUT_PATH, index=False)

    # 9. Report the results.
    print("\n========== CLEANING SUMMARY ==========")
    print("Original rows:", 72134)
    print("Rows removed for having no text:", removed_rows)
    print("Rows remaining:", len(df))
    print("Columns saved:", df.columns.tolist())

    print("\n========== LABEL DISTRIBUTION ==========")
    print(df["label"].value_counts().sort_index())

    print("\n========== MISSING VALUES AFTER CLEANING ==========")
    print(df[["title", "text", "combined_text", "label"]].isna().sum())

    print("\n========== TEXT LENGTH SUMMARY ==========")
    print(df["combined_text"].str.split().str.len().describe())

    print("\n========== SAMPLE CLEANED RECORD ==========")
    sample = df.iloc[0]

    print("Source row ID:", sample["source_row_id"])
    print("Label:", sample["label"])
    print("Title:", sample["title"][:150])
    print("Combined text preview:", sample["combined_text"][:300])

    print("\nSaved cleaned dataset to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()

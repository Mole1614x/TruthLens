
from pathlib import Path

import pandas as pd

DATA_PATH = Path("data/raw/WELFake_Dataset.csv")


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH.resolve()}"
        )

    df = pd.read_csv(DATA_PATH)

    print("\n========== BASIC INFORMATION ==========")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {df.columns.tolist()}")

    # Check the exported index column
    index_col = "Unnamed: 0"

    print("\n========== INDEX COLUMN CHECK ==========")
    if index_col in df.columns:
        index_values = df[index_col]

        print("Unique values:", index_values.nunique())
        print("Minimum:", index_values.min())
        print("Maximum:", index_values.max())
        print("Monotonically increasing:", index_values.is_monotonic_increasing)
        print(
            "Matches sequential index:",
            index_values.reset_index(drop=True).equals(
                pd.Series(range(len(df)), name=index_col)
            ),
        )

    # Missing values
    print("\n========== MISSING VALUES ==========")
    print(df.isnull().sum())

    print(
        "\nRows with both title and text missing:",
        df["title"].isna().mul(df["text"].isna()).sum(),
    )

    # Exact duplicates across all columns
    print("\n========== EXACT ROW DUPLICATES ==========")
    print(df.duplicated().sum())

    # Normalize whitespace for duplicate investigation.
    # We are NOT performing model preprocessing here.
    title = df["title"].fillna("").astype(str).str.strip()
    text = df["text"].fillna("").astype(str).str.strip()

    title_nonempty = title[title.ne("")]
    text_nonempty = text[text.ne("")]

    print("\n========== DUPLICATE HEADLINES ==========")
    print(
        "Repeated non-empty headlines:",
        title_nonempty.duplicated().sum(),
    )

    print("\n========== DUPLICATE ARTICLE TEXT ==========")
    print(
        "Repeated non-empty article bodies:",
        text_nonempty.duplicated().sum(),
    )

    # Check if identical article bodies have conflicting labels.
    audit = pd.DataFrame({
        "text": text,
        "label": df["label"],
    })

    audit = audit[audit["text"].ne("")]

    label_counts = audit.groupby("text")["label"].nunique()
    conflicting_texts = label_counts[label_counts > 1]

    print("\n========== CONFLICTING LABELS ==========")
    print(
        "Unique repeated article texts with conflicting labels:",
        len(conflicting_texts),
    )

    # Text length statistics
    print("\n========== TITLE LENGTHS (CHARACTERS) ==========")
    print(title[title.ne("")].str.len().describe(
        percentiles=[0.25, 0.50, 0.75, 0.95, 0.99]
    ))

    print("\n========== ARTICLE LENGTHS (CHARACTERS) ==========")
    print(text[text.ne("")].str.len().describe(
        percentiles=[0.25, 0.50, 0.75, 0.95, 0.99]
    ))

    # Word counts help us understand article length more directly.
    print("\n========== ARTICLE WORD COUNTS ==========")
    word_counts = text.str.split().str.len()

    print(word_counts.describe(
        percentiles=[0.25, 0.50, 0.75, 0.95, 0.99]
    ))

    # Label mapping confirmed from the dataset documentation.
    print("\n========== LABEL DISTRIBUTION ==========")
    print(df["label"].value_counts(dropna=False).sort_index())
    print("\n0 = Fake, 1 = Real")


if __name__ == "__main__":
    main()

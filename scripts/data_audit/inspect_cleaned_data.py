
from pathlib import Path

import pandas as pd


DATA_PATH = Path("data/processed/truthlens_cleaned.csv")


def normalize_text(series):
    """Normalize case and whitespace for comparison."""
    return (
        series.fillna("")
        .astype(str)
        .str.casefold()
        .str.split()
        .str.join(" ")
    )


def main():
    # keep_default_na=False preserves empty CSV fields as empty strings.
    df = pd.read_csv(DATA_PATH, keep_default_na=False)

    print("========== DATASET CHECK ==========")
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())

    print("\nMissing or empty values:")
    for column in ["title", "text", "combined_text", "label"]:
        print(f"{column}: {(df[column].astype(str).str.strip() == '').sum()}")

    print("\n========== TEXT QUALITY ==========")
    df["title_normalized"] = normalize_text(df["title"])
    df["text_normalized"] = normalize_text(df["text"])

    print("Empty titles:", int(df["title_normalized"].eq("").sum()))
    print("Empty article bodies:", int(df["text_normalized"].eq("").sum()))
    print(
        "Empty combined texts:",
        int(df["combined_text"].str.strip().eq("").sum()),
    )

    print("\n========== DUPLICATE CHECK ==========")
    print(
        "Rows with repeated normalized titles:",
        int(df["title_normalized"].duplicated(keep=False).sum()),
    )
    print(
        "Rows with repeated normalized article bodies:",
        int(
            df.loc[df["text_normalized"].ne(""), "text_normalized"]
            .duplicated(keep=False)
            .sum()
        ),
    )

    nonempty_bodies = df.loc[
        df["text_normalized"].ne(""), "text_normalized"
    ]
    body_counts = nonempty_bodies.value_counts()
    repeated_body_groups = body_counts[body_counts > 1]

    print("Repeated non-empty body groups:", len(repeated_body_groups))

    print("\n========== LABEL CHECK ==========")
    print(df["label"].value_counts().sort_index())
    print("\nLabel proportions:")
    print(df["label"].value_counts(normalize=True).sort_index().round(4))

    print("\n========== SAMPLE RECORDS ==========")
    for _, row in df.head(3).iterrows():
        print("\nSource row ID:", row["source_row_id"])
        print("Label:", row["label"])
        print("Title:", row["title"][:120] or "[EMPTY TITLE]")
        print("Body preview:", row["text"][:200] or "[EMPTY BODY]")


if __name__ == "__main__":
    main()

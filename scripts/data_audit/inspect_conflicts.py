
from pathlib import Path

import pandas as pd

DATA_PATH = Path("data/raw/WELFake_Dataset.csv")


def normalize_text(series):
    return (
        series.fillna("")
        .astype(str)
        .str.casefold()
        .str.split()
        .str.join(" ")
    )


def main():
    df = pd.read_csv(DATA_PATH)

    # Keep the original row index for tracing records.
    df["row_id"] = range(len(df))

    df["normalized_title"] = normalize_text(df["title"])

    valid = df[df["normalized_title"].ne("")]

    # Identify normalized headlines associated with multiple labels.
    label_counts = valid.groupby("normalized_title")["label"].nunique()
    conflicting_titles = label_counts[label_counts > 1].index

    print("Conflicting headline groups:", len(conflicting_titles))

    for number, title in enumerate(conflicting_titles, start=1):
        group = valid[valid["normalized_title"] == title]

        print(f"\n========== CONFLICT {number} ==========")
        print("Normalized headline:", title[:200])
        print("Labels:", group["label"].value_counts().to_dict())

        for _, row in group.iterrows():
            body = str(row["text"]) if pd.notna(row["text"]) else ""

            print(f"\nRow ID: {row['row_id']}")
            print(f"Label: {row['label']} (0 = Fake, 1 = Real)")
            print("Original headline:", str(row["title"])[:200])
            print("Article preview:", body[:250].replace("\n", " "))


if __name__ == "__main__":
    main()

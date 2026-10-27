
from pathlib import Path
from difflib import SequenceMatcher

import pandas as pd

DATA_PATH = Path("data/raw/WELFake_Dataset.csv")


def normalize_text(text):
    """Normalize case and whitespace for text comparison."""
    if pd.isna(text):
        return ""

    return " ".join(str(text).casefold().split())


def similarity_score(text_a, text_b):
    """Return a character-sequence similarity score between 0 and 1."""
    return SequenceMatcher(
        None,
        normalize_text(text_a),
        normalize_text(text_b),
        autojunk=False,
    ).ratio()


def main():
    df = pd.read_csv(DATA_PATH)

    # Preserve the original row positions for reference.
    df["row_id"] = range(len(df))

    # Normalize titles using the same approach as the earlier audit.
    df["normalized_title"] = df["title"].map(normalize_text)

    valid = df[df["normalized_title"].ne("")]

    label_counts = valid.groupby("normalized_title")["label"].nunique()
    conflicting_titles = label_counts[label_counts > 1].index

    print("Conflicting headline groups:", len(conflicting_titles))

    for number, title in enumerate(conflicting_titles, start=1):
        group = valid[valid["normalized_title"] == title]

        print(f"\n{'=' * 65}")
        print(f"CONFLICT {number}: {title[:150]}")

        rows = list(group.to_dict("records"))

        # Compare every pair of articles within this headline group.
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                row_a = rows[i]
                row_b = rows[j]

                body_a = row_a["text"]
                body_b = row_b["text"]

                score = similarity_score(body_a, body_b)

                print(
                    f"\nRows {row_a['row_id']} and {row_b['row_id']}"
                )
                print(
                    f"Labels: {row_a['label']} vs {row_b['label']}"
                )
                print(f"Body similarity: {score:.4f}")

                preview_a = normalize_text(body_a)[:160]
                preview_b = normalize_text(body_b)[:160]

                print(f"Article A: {preview_a}")
                print(f"Article B: {preview_b}")


if __name__ == "__main__":
    main()

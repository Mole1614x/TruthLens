
from pathlib import Path

import pandas as pd

DATA_PATH = Path("data/raw/WELFake_Dataset.csv")


def normalize_text(value):
    if pd.isna(value):
        return ""
    return " ".join(str(value).casefold().split())


def main():
    df = pd.read_csv(DATA_PATH)

    df["normalized_text"] = df["text"].map(normalize_text)
    df["body_length"] = df["normalized_text"].str.len()

    valid = df[df["normalized_text"].ne("")]

    body_groups = (
        valid.groupby("normalized_text")
        .agg(
            rows=("label", "size"),
            unique_labels=("label", "nunique"),
            body_length=("body_length", "first"),
            titles=("title", lambda s: list(s.dropna().astype(str))),
            labels=("label", lambda s: s.value_counts().to_dict()),
        )
        .reset_index()
    )

    repeated = body_groups[body_groups["rows"] > 1]
    repeated = repeated.sort_values("rows", ascending=False)

    print("========== REPEATED BODY AUDIT ==========")
    print("Non-empty article bodies:", len(valid))
    print("Unique normalized bodies:", valid["normalized_text"].nunique())
    print("Repeated body groups:", len(repeated))
    print("Rows in repeated body groups:", int(repeated["rows"].sum()))

    print("\n========== LARGEST REPEATED BODIES ==========")

    for _, group in repeated.head(15).iterrows():
        print("\n" + "-" * 60)
        print("Rows:", group["rows"])
        print("Body characters:", group["body_length"])
        print("Label counts:", group["labels"])
        print("Distinct labels:", group["unique_labels"])
        print("Body preview:", group["normalized_text"][:220])

        print("Example headlines:")
        for title in group["titles"][:5]:
            print(" -", title[:120])


if __name__ == "__main__":
    main()

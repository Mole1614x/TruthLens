
from pathlib import Path
import pandas as pd

DATA_PATH = Path("data/processed/splits")
OUTPUT_PATH = DATA_PATH / "overlapping_body_report.csv"


def normalize_text(series):
    return (
        series.fillna("")
        .astype(str)
        .str.casefold()
        .str.split()
        .str.join(" ")
    )


def main():
    frames = []

    # Load the three existing data splits.
    for split_name in ["train", "validation", "test"]:
        path = DATA_PATH / f"{split_name}.csv"

        df = pd.read_csv(path, keep_default_na=False)
        df["split"] = split_name
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)

    # Normalize article bodies to find exact text matches.
    df["normalized_body"] = normalize_text(df["text"])
    df = df.loc[df["normalized_body"].ne("")].copy()

    # Summarize each unique article body.
    body_summary = (
        df.groupby("normalized_body")
        .agg(
            row_count=("source_row_id", "size"),
            split_count=("split", "nunique"),
            splits=("split", lambda values: ", ".join(sorted(set(values)))),
            labels=(
                "label",
                lambda values: ", ".join(
                    map(str, sorted(set(map(int, values))))
                ),
            ),
        )
        .reset_index()
    )

    body_summary["body_length"] = (
        body_summary["normalized_body"].str.len()
    )

    # Keep bodies appearing in more than one split.
    overlapping = body_summary.loc[
        body_summary["split_count"] > 1
    ].copy()

    overlapping["preview"] = (
        overlapping["normalized_body"].str.slice(0, 350)
    )

    overlapping = overlapping.sort_values(
        ["body_length", "row_count"],
        ascending=[False, False],
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    overlapping.to_csv(OUTPUT_PATH, index=False)

    print("========== CROSS-SPLIT BODY AUDIT ==========")
    print("Unique non-empty article bodies:", len(body_summary))
    print("Bodies appearing across multiple splits:", len(overlapping))

    affected_rows = int(
        overlapping["row_count"].sum()
    )
    print("Total rows belonging to overlapping body groups:", affected_rows)

    print("\n========== OVERLAP BY BODY LENGTH ==========")
    if not overlapping.empty:
        length_bins = [-1, 100, 500, 1500, float("inf")]
        length_labels = [
            "0-100 characters",
            "101-500 characters",
            "501-1500 characters",
            "Over 1500 characters",
        ]

        print(
            pd.cut(
                overlapping["body_length"],
                bins=length_bins,
                labels=length_labels,
            ).value_counts(sort=False)
        )

    print("\n========== 15 LONGEST OVERLAPPING BODIES ==========")

    for _, row in overlapping.head(15).iterrows():
        print("\n" + "-" * 60)
        print("Body length:", row["body_length"])
        print("Rows:", row["row_count"])
        print("Splits:", row["splits"])
        print("Labels:", row["labels"])
        print("Preview:", row["preview"])

    print("\nFull report saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()


from pathlib import Path

import pandas as pd

DATA_PATH = Path("data/raw/WELFake_Dataset.csv")


def normalize_text(series):
    """Normalize case and whitespace for duplicate auditing."""
    return (
        series.fillna("")
        .astype(str)
        .str.casefold()
        .str.split()
        .str.join(" ")
    )


def report_duplicate_groups(df, column, normalized):
    """Report repeated-content groups and their label consistency."""
    valid = normalized[normalized.ne("")]

    group_sizes = valid.value_counts()
    repeated_groups = group_sizes[group_sizes > 1]

    print(f"\n========== {column.upper()} AUDIT ==========")
    print("Non-empty records:", len(valid))
    print("Unique non-empty normalized values:", normalized[normalized.ne("")].nunique())
    print("Repeated-content groups:", len(repeated_groups))
    print(
        "Records belonging to repeated groups:",
        int(repeated_groups.sum()),
    )
    print(
        "Repeated occurrences beyond the first:",
        int((repeated_groups - 1).sum()),
    )

    # Check label consistency within repeated-content groups.
    audit = pd.DataFrame({
        "content": normalized,
        "label": df["label"],
    })

    audit = audit[audit["content"].ne("")]
    label_counts = audit.groupby("content")["label"].nunique()
    conflicting = label_counts[label_counts > 1]

    print("Groups with conflicting labels:", len(conflicting))

    if len(repeated_groups):
        print("Largest group sizes:")
        print(repeated_groups.head(10).to_string())


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH.resolve()}"
        )

    df = pd.read_csv(DATA_PATH)

    print("========== LABEL MAPPING ==========")
    print("0 = Fake, 1 = Real")

    # Audit repeated headlines and article bodies separately.
    normalized_titles = normalize_text(df["title"])
    normalized_texts = normalize_text(df["text"])

    report_duplicate_groups(
        df, "headline", normalized_titles
    )

    report_duplicate_groups(
        df, "article body", normalized_texts
    )

    # Investigate whether label distribution varies by row order.
    print("\n========== LABEL DISTRIBUTION BY ROW-ORDER QUINTILE ==========")

    ordered = df[["label"]].copy()

    ordered["row_order_group"] = pd.qcut(
        range(len(ordered)),
        q=5,
        labels=["First 20%", "Second 20%", "Middle 20%",
                "Fourth 20%", "Last 20%"],
    )

    distribution = pd.crosstab(
        ordered["row_order_group"],
        ordered["label"],
        normalize="index",
    )

    distribution = distribution.rename(
        columns={0: "Fake proportion", 1: "Real proportion"}
    )

    print(distribution.round(4))

    print("\n========== MISSING TEXT COUNTS ==========")
    print("Missing headlines:", df["title"].isna().sum())
    print("Missing article bodies:", df["text"].isna().sum())

    print("\nAudit complete. Original dataset was not modified.")


if __name__ == "__main__":
    main()

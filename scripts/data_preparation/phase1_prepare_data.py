
from pathlib import Path
from collections import defaultdict

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


# ============================================================
# CONFIGURATION
# ============================================================

RAW_PATH = Path("data/raw/WELFake_Dataset.csv")
PROCESSED_DIR = Path("data/processed")
SPLIT_DIR = PROCESSED_DIR / "splits"

CLEANED_PATH = PROCESSED_DIR / "truthlens_cleaned.csv"
REPORT_PATH = SPLIT_DIR / "phase1_audit_report.txt"

RANDOM_STATE = 42
BODY_GROUP_MIN_CHARS = 1000


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(series):
    """Normalize case and whitespace without removing words."""
    return (
        series.fillna("")
        .astype(str)
        .str.casefold()
        .str.split()
        .str.join(" ")
    )


# ============================================================
# UNION-FIND FOR DUPLICATE GROUPS
# ============================================================

class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item):
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, first, second):
        root_a = self.find(first)
        root_b = self.find(second)

        if root_a == root_b:
            return

        if self.rank[root_a] < self.rank[root_b]:
            root_a, root_b = root_b, root_a

        self.parent[root_b] = root_a

        if self.rank[root_a] == self.rank[root_b]:
            self.rank[root_a] += 1


# ============================================================
# DATA CLEANING
# ============================================================

def load_and_clean():
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {RAW_PATH}"
        )

    df = pd.read_csv(RAW_PATH)

    required_columns = {"title", "text", "label"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    print("\n========== RAW DATASET ==========")
    print("Rows:", len(df))
    print("Columns:", df.columns.tolist())
    print("\nMissing values:")
    print(df[["title", "text", "label"]].isna().sum())

    # Keep a stable reference to each original row.
    df["source_row_id"] = range(len(df))

    # Handle missing titles and bodies.
    df["title"] = df["title"].fillna("").astype(str).str.strip()
    df["text"] = df["text"].fillna("").astype(str).str.strip()

    # Retain only records with some usable text.
    combined = (df["title"] + " " + df["text"]).str.strip()
    valid_mask = combined.ne("")

    removed_rows = int((~valid_mask).sum())
    df = df.loc[valid_mask].copy()

    df["combined_text"] = (
        df["title"] + " " + df["text"]
    ).str.strip()

    # Validate labels before creating splits.
    df["label"] = pd.to_numeric(df["label"], errors="raise").astype(int)

    observed_labels = set(df["label"].unique())
    if observed_labels != {0, 1}:
        raise ValueError(
            f"Expected binary labels 0 and 1; found {observed_labels}"
        )

    # Preserve the dataset's original label mapping:
    # 0 = Fake, 1 = Real.
    df = df[
        [
            "source_row_id",
            "title",
            "text",
            "combined_text",
            "label",
        ]
    ].copy()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEANED_PATH, index=False)

    print("\n========== CLEANING SUMMARY ==========")
    print("Rows removed for having no usable text:", removed_rows)
    print("Rows retained:", len(df))
    print("Empty titles retained:", int(df["title"].eq("").sum()))
    print("Empty bodies retained:", int(df["text"].eq("").sum()))
    print("Labels: 0 = Fake, 1 = Real")
    print("\nLabel counts:")
    print(df["label"].value_counts().sort_index())

    return df, removed_rows


# ============================================================
# BUILD DUPLICATE GROUPS
# ============================================================

def build_duplicate_groups(df):
    """
    Keep identical normalized titles together.

    Also keep identical normalized bodies of at least
    BODY_GROUP_MIN_CHARS characters together.

    Short repeated snippets are not used to connect records,
    because common snippets can join unrelated articles.
    """
    size = len(df)
    union_find = UnionFind(size)

    normalized_titles = normalize_text(df["title"])
    normalized_bodies = normalize_text(df["text"])

    first_title_row = {}
    first_body_row = {}

    for position in range(size):
        title = normalized_titles.iloc[position]
        body = normalized_bodies.iloc[position]

        # Identical non-empty headlines belong together.
        if title:
            if title in first_title_row:
                union_find.union(
                    position, first_title_row[title]
                )
            else:
                first_title_row[title] = position

        # Long, exactly repeated bodies belong together.
        if len(body) >= BODY_GROUP_MIN_CHARS:
            if body in first_body_row:
                union_find.union(
                    position, first_body_row[body]
                )
            else:
                first_body_row[body] = position

    groups = [
        union_find.find(position)
        for position in range(size)
    ]

    return groups, normalized_titles, normalized_bodies


# ============================================================
# CREATE TRAIN / VALIDATION / TEST SPLITS
# ============================================================

def create_splits(df):
    groups, normalized_titles, normalized_bodies = (
        build_duplicate_groups(df)
    )

    X = df["combined_text"]
    y = df["label"]

    splitter = StratifiedGroupKFold(
        n_splits=10,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    fold_number = [-1] * len(df)

    for fold, (_, held_out_positions) in enumerate(
        splitter.split(X, y, groups=groups)
    ):
        for position in held_out_positions:
            fold_number[position] = fold

    if any(fold < 0 for fold in fold_number):
        raise RuntimeError("Some records were not assigned to a fold.")

    # Eight folds for training, one for validation, one for testing.
    split_names = [
        "train" if fold < 8
        else "validation" if fold == 8
        else "test"
        for fold in fold_number
    ]

    result = df.copy()
    result["split"] = split_names
    result["normalized_title"] = normalized_titles
    result["normalized_body"] = normalized_bodies

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    for split_name in ["train", "validation", "test"]:
        subset = result.loc[result["split"] == split_name].copy()

        subset[
            [
                "source_row_id",
                "title",
                "text",
                "combined_text",
                "label",
            ]
        ].to_csv(
            SPLIT_DIR / f"{split_name}.csv",
            index=False,
        )

    result[
        ["source_row_id", "label", "split"]
    ].to_csv(
        SPLIT_DIR / "split_manifest.csv",
        index=False,
    )

    print("\n========== SPLIT SIZES ==========")

    for split_name in ["train", "validation", "test"]:
        subset = result.loc[result["split"] == split_name]
        print(
            f"{split_name.capitalize():10} "
            f"{len(subset):6} rows "
            f"({len(subset) / len(result):.2%})"
        )

    print("\n========== LABEL DISTRIBUTION BY SPLIT ==========")
    print(
        pd.crosstab(
            result["split"],
            result["label"],
            normalize="index",
        ).round(4)
    )

    return result


# ============================================================
# AUDIT SPLITS FOR DUPLICATE LEAKAGE
# ============================================================

def find_cross_split_overlaps(result, column):
    """Return normalized values that occur in multiple splits."""
    data = result.loc[result[column].ne("")]

    summary = (
        data.groupby(column)
        .agg(
            row_count=("source_row_id", "size"),
            split_count=("split", "nunique"),
            splits=("split", lambda values: ", ".join(sorted(set(values)))),
            labels=("label", lambda values: ", ".join(
                map(str, sorted(set(map(int, values))))
            )),
        )
        .reset_index()
    )

    return summary.loc[summary["split_count"] > 1].copy()


def audit_splits(result):
    title_overlaps = find_cross_split_overlaps(
        result, "normalized_title"
    )

    body_overlaps = find_cross_split_overlaps(
        result, "normalized_body"
    )

    # Short body overlaps are reported separately.
    if not body_overlaps.empty:
        body_overlaps["body_length"] = (
            body_overlaps["normalized_body"].str.len()
        )

        long_body_overlaps = body_overlaps.loc[
            body_overlaps["body_length"] >= BODY_GROUP_MIN_CHARS
        ].copy()

        short_body_overlaps = body_overlaps.loc[
            body_overlaps["body_length"] < BODY_GROUP_MIN_CHARS
        ].copy()
    else:
        long_body_overlaps = body_overlaps.copy()
        short_body_overlaps = body_overlaps.copy()

    # Identical titles and substantial exact bodies should not cross splits.
    checks_passed = (
        title_overlaps.empty
        and long_body_overlaps.empty
    )

    print("\n========== DUPLICATE LEAKAGE AUDIT ==========")
    print("Cross-split normalized headline groups:", len(title_overlaps))
    print(
        "Cross-split exact body groups (all lengths):",
        len(body_overlaps),
    )
    print(
        f"Cross-split exact body groups >= {BODY_GROUP_MIN_CHARS} characters:",
        len(long_body_overlaps),
    )
    print(
        f"Cross-split exact body groups < {BODY_GROUP_MIN_CHARS} characters:",
        len(short_body_overlaps),
    )

    if not title_overlaps.empty:
        print("\nWARNING: Headline overlap remains.")

    if not long_body_overlaps.empty:
        print("\nWARNING: Substantial exact body overlap remains.")

    if not short_body_overlaps.empty:
        print(
            "\nNote: Short repeated bodies remain for review; "
            "they may be snippets or boilerplate."
        )

    if checks_passed:
        print(
            "\nPASS: No exact normalized headline or substantial "
            "body group crosses splits."
        )
    else:
        print(
            "\nFAIL: Review the overlap report before trusting "
            "final test metrics."
        )

    # Save the detailed overlap report.
    overlap_report_path = SPLIT_DIR / "overlap_report.csv"

    if body_overlaps.empty:
        body_overlaps.to_csv(overlap_report_path, index=False)
    else:
        body_overlaps.to_csv(overlap_report_path, index=False)

    return {
        "title_overlap_groups": len(title_overlaps),
        "body_overlap_groups": len(body_overlaps),
        "long_body_overlap_groups": len(long_body_overlaps),
        "short_body_overlap_groups": len(short_body_overlaps),
        "checks_passed": checks_passed,
    }


# ============================================================
# SAVE PHASE 1 REPORT
# ============================================================

def save_report(df, removed_rows, audit):
    report_lines = [
        "TRUTHLENS - PHASE 1 DATA PREPARATION REPORT",
        "=" * 50,
        f"Original dataset: {RAW_PATH}",
        f"Original rows: {len(df) + removed_rows}",
        f"Rows removed for empty title and body: {removed_rows}",
        f"Rows retained: {len(df)}",
        "Label mapping: 0 = Fake, 1 = Real",
        f"Duplicate body grouping threshold: {BODY_GROUP_MIN_CHARS} characters",
        "",
        "SPLIT SIZES",
    ]

    for split_name in ["train", "validation", "test"]:
        split_path = SPLIT_DIR / f"{split_name}.csv"
        split_df = pd.read_csv(split_path, keep_default_na=False)
        report_lines.append(f"{split_name}: {len(split_df)} rows")

    report_lines.extend([
        "",
        "DUPLICATE AUDIT",
        f"Cross-split headline groups: {audit['title_overlap_groups']}",
        f"Cross-split exact body groups: {audit['body_overlap_groups']}",
        f"Cross-split substantial body groups: {audit['long_body_overlap_groups']}",
        f"Cross-split short body groups: {audit['short_body_overlap_groups']}",
        f"Headline and substantial-body checks passed: {audit['checks_passed']}",
        "",
        "Important: This audit checks exact normalized text only.",
        "It does not prove that paraphrased or near-duplicate articles are absent.",
        "Model performance must be evaluated on the held-out test split.",
    ])

    REPORT_PATH.write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    print("\n========== PHASE 1 REPORT ==========")
    print("Saved:", REPORT_PATH)
    print("Overlap details:", SPLIT_DIR / "overlap_report.csv")


# ============================================================
# MAIN
# ============================================================

def main():
    df, removed_rows = load_and_clean()
    result = create_splits(df)
    audit = audit_splits(result)
    save_report(df, removed_rows, audit)

    print("\n========== PHASE 1 SUMMARY ==========")

    if audit["checks_passed"]:
        print("Phase 1 split checks passed.")
        print("You can proceed to Phase 2: Text Preprocessing.")
    else:
        print(
            "Phase 1 generated the data and reports, but duplicate "
            "checks need review before final evaluation."
        )


if __name__ == "__main__":
    main()

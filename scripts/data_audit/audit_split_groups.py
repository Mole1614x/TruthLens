
from pathlib import Path

import pandas as pd

DATA_PATH = Path("data/raw/WELFake_Dataset.csv")


def normalize_text(value):
    """Normalize case and whitespace for duplicate detection."""
    if pd.isna(value):
        return ""

    return " ".join(str(value).casefold().split())


class UnionFind:
    """Track rows that belong to the same connected group."""

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


def connect_duplicates(df, column, union_find):
    """Connect rows sharing the same non-empty normalized text."""
    first_seen = {}

    for row_id, text in enumerate(df[column]):
        if not text:
            continue

        if text in first_seen:
            union_find.union(row_id, first_seen[text])
        else:
            first_seen[text] = row_id


def main():
    df = pd.read_csv(DATA_PATH)

    df["normalized_title"] = df["title"].map(normalize_text)
    df["normalized_text"] = df["text"].map(normalize_text)

    union_find = UnionFind(len(df))

    # Connect rows sharing an exact normalized title or body.
    connect_duplicates(df, "normalized_title", union_find)
    connect_duplicates(df, "normalized_text", union_find)

    df["group_id"] = [
        union_find.find(row_id)
        for row_id in range(len(df))
    ]

    group_sizes = df.groupby("group_id").size()
    repeated_groups = group_sizes[group_sizes > 1]

    label_counts = df.groupby("group_id")["label"].nunique()
    conflicting_groups = label_counts[label_counts > 1]

    print("========== SPLIT GROUP AUDIT ==========")
    print("Total rows:", len(df))
    print("Total connected groups:", group_sizes.size)
    print("Groups containing multiple rows:", len(repeated_groups))
    print(
        "Rows in multi-row groups:",
        int(repeated_groups.sum()),
    )
    print(
        "Largest group size:",
        int(group_sizes.max()),
    )
    print(
        "Groups with conflicting labels:",
        len(conflicting_groups),
    )

    print("\n========== LARGEST GROUPS ==========")

    for group_id, size in repeated_groups.sort_values(
        ascending=False
    ).head(10).items():
        group = df[df["group_id"] == group_id]

        print(f"\nGroup ID: {group_id}")
        print("Rows:", int(size))
        print("Labels:", group["label"].value_counts().to_dict())

        for _, row in group.head(3).iterrows():
            print(
                f"  Row {row.name} | Label {row['label']} | "
                f"Title: {row['title']!s:.120}"
            )

    print("\n========== CONFLICTING GROUPS ==========")

    for group_id in conflicting_groups.index[:20]:
        group = df[df["group_id"] == group_id]

        print(f"\nGroup ID: {group_id}")
        print("Rows:", len(group))
        print("Labels:", group["label"].value_counts().to_dict())

        for _, row in group.head(5).iterrows():
            print(
                f"  Row {row.name} | Label {row['label']} | "
                f"Title: {row['title']!s:.120}"
            )


if __name__ == "__main__":
    main()

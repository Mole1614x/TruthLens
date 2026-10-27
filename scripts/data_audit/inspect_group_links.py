
from pathlib import Path
from collections import defaultdict

import pandas as pd

DATA_PATH = Path("data/raw/WELFake_Dataset.csv")


def normalize_text(value):
    if pd.isna(value):
        return ""
    return " ".join(str(value).casefold().split())


def main():
    df = pd.read_csv(DATA_PATH)

    df["normalized_title"] = df["title"].map(normalize_text)
    df["normalized_text"] = df["text"].map(normalize_text)

    # Rebuild the exact-match connected groups.
    from audit_split_groups import UnionFind, connect_duplicates

    union_find = UnionFind(len(df))
    connect_duplicates(df, "normalized_title", union_find)
    connect_duplicates(df, "normalized_text", union_find)

    df["group_id"] = [
        union_find.find(i) for i in range(len(df))
    ]

    # Investigate the three largest groups from the previous output.
    target_groups = [50076, 50661, 43072]

    for group_id in target_groups:
        group = df[df["group_id"] == group_id]
        member_ids = set(group.index)

        print("\n" + "=" * 65)
        print(f"GROUP {group_id}: {len(group)} rows")

        for column, description in [
            ("normalized_title", "EXACT SHARED HEADLINES"),
            ("normalized_text", "EXACT SHARED ARTICLE BODIES"),
        ]:
            buckets = defaultdict(list)

            for row_id in member_ids:
                value = df.at[row_id, column]
                if value:
                    buckets[value].append(row_id)

            matches = [
                (value, ids)
                for value, ids in buckets.items()
                if len(ids) > 1
            ]

            print(f"\n{description}: {len(matches)} matching groups")

            for value, ids in matches[:10]:
                print(f"Rows: {ids[:15]}")
                print(f"Matching text preview: {value[:140]}")

        print("\nGROUP MEMBERS (first 10):")
        for row_id, row in group.head(10).iterrows():
            print(
                f"Row {row_id} | Label {row['label']} | "
                f"Title: {str(row['title'])[:100]}"
            )


if __name__ == "__main__":
    main()

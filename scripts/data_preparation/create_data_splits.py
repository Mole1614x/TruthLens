
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


DATA_PATH = Path("data/processed/truthlens_cleaned.csv")
OUTPUT_DIR = Path("data/processed/splits")


def normalize_text(series):
    return (
        series.fillna("")
        .astype(str)
        .str.casefold()
        .str.split()
        .str.join(" ")
    )


def create_groups(titles, bodies):
    """Keep records with repeated titles or bodies in the same group."""

    n = len(titles)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        root_a = find(a)
        root_b = find(b)

        if root_a != root_b:
            parent[root_b] = root_a

    seen_titles = {}
    seen_bodies = {}

    for i in range(n):
        title = titles.iloc[i]
        body = bodies.iloc[i]

        # Group records with the same non-empty headline.
        if title:
            if title in seen_titles:
                union(i, seen_titles[title])
            else:
                seen_titles[title] = i

        # Group records with the same non-empty article body.
        if body:
            if body in seen_bodies:
                union(i, seen_bodies[body])
            else:
                seen_bodies[body] = i

    # Assign one group ID to each connected group.
    return [find(i) for i in range(n)]


def main():
    # Step 1: Load the dataset.
    df = pd.read_csv(DATA_PATH, keep_default_na=False)

    print("Dataset loaded:", len(df), "records")

    # Step 2: Normalize headlines and article bodies.
    titles = normalize_text(df["title"])
    bodies = normalize_text(df["text"])

    # Step 3: Create groups using both titles and bodies.
    groups = create_groups(titles, bodies)

    print("Unique groups:", len(set(groups)))

    # Step 4: Separate features and labels.
    X = df["combined_text"].fillna("")
    y = df["label"].astype(int)

    # Step 5: Create 10 stratified group folds.
    splitter = StratifiedGroupKFold(
        n_splits=10,
        shuffle=True,
        random_state=42
    )

    df["fold"] = -1

    for fold, (_, test_index) in enumerate(
        splitter.split(X, y, groups)
    ):
        df.loc[df.index[test_index], "fold"] = fold

    # Step 6: Assign train, validation and test sets.
    df["split"] = "train"
    df.loc[df["fold"] == 8, "split"] = "validation"
    df.loc[df["fold"] == 9, "split"] = "test"

    # Step 7: Create output folder.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 8: Save the three datasets.
    for name in ["train", "validation", "test"]:
        data = df[df["split"] == name]

        data.drop(columns=["fold", "split"]).to_csv(
            OUTPUT_DIR / f"{name}.csv",
            index=False
        )

        print(name, "records:", len(data))

    # Step 9: Save the split manifest.
    df[["source_row_id", "label", "split"]].to_csv(
        OUTPUT_DIR / "split_manifest.csv",
        index=False
    )

    # Step 10: Check label distribution.
    print("\nLabel distribution:")
    print(
        pd.crosstab(
            df["split"],
            df["label"],
            normalize="index"
        ).round(3)
    )

    # Step 11: Verify that titles do not cross splits.
    title_data = pd.DataFrame({
        "title": titles,
        "split": df["split"]
    })

    title_data = title_data[title_data["title"] != ""]

    title_counts = title_data.groupby("title")["split"].nunique()

    print(
        "\nHeadline groups crossing splits:",
        (title_counts > 1).sum()
    )

    # Step 12: Verify that article bodies do not cross splits.
    body_data = pd.DataFrame({
        "body": bodies,
        "split": df["split"]
    })

    body_data = body_data[body_data["body"] != ""]

    body_counts = body_data.groupby("body")["split"].nunique()

    print(
        "Article bodies crossing splits:",
        (body_counts > 1).sum()
    )

    print("\nDataset splitting completed!")
    print("Files saved in:", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()

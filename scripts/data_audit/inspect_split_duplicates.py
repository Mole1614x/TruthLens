import pandas as pd

folder = "data/processed/splits"

train = pd.read_csv(f"{folder}/train.csv", keep_default_na=False)
validation = pd.read_csv(f"{folder}/validation.csv", keep_default_na=False)
test = pd.read_csv(f"{folder}/test.csv", keep_default_na=False)

train["split"] = "train"
validation["split"] = "validation"
test["split"] = "test"

df = pd.concat([train, validation, test], ignore_index=True)

# Normalize article bodies
df["body"] = (
    df["text"]
    .astype(str)
    .str.casefold()
    .str.split()
    .str.join(" ")
)

# Remove empty bodies
df = df[df["body"] != ""]

# Find bodies appearing in multiple splits
body_split_counts = df.groupby("body")["split"].nunique()
duplicate_bodies = body_split_counts[body_split_counts > 1]

print("Overlapping article bodies:", len(duplicate_bodies))

# Display a few examples
for body in duplicate_bodies.index[:5]:
    rows = df[df["body"] == body]

    print("\n" + "=" * 60)
    print("Article:", body[:300])
    print("Splits:", rows["split"].unique())
    print("Labels:", rows["label"].unique())

import pandas as pd
from pathlib import Path


INPUT_FILE = "9606.protein.links.detailed.v12.0.txt.gz"

OUTPUT_DATASET = "data/processed/ppi_dataset_hard.csv"
OUTPUT_POSITIVES = "data/processed/hard_positive_pairs.csv"
OUTPUT_NEGATIVES = "data/processed/hard_negative_pairs.csv"

RANDOM_SEED = 42

# Harder setup
POSITIVE_THRESHOLD = 800

# Instead of <= 200, use medium-confidence negatives
NEGATIVE_MIN_SCORE = 300
NEGATIVE_MAX_SCORE = 500

FEATURE_COLS = [
    "neighborhood",
    "fusion",
    "cooccurence",
    "coexpression",
    "experimental",
    "database",
    "textmining",
    "combined_score",
]


def clean_pairs(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["protein1"] != df["protein2"]].copy()

    p1 = df["protein1"].copy()
    p2 = df["protein2"].copy()

    df["protein1"] = p1.where(p1 <= p2, p2)
    df["protein2"] = p2.where(p1 <= p2, p1)

    df = df.drop_duplicates(subset=["protein1", "protein2"]).copy()

    return df


def main():
    Path("data/processed").mkdir(parents=True, exist_ok=True)

    input_path = Path(INPUT_FILE)
    if not input_path.exists():
        raise FileNotFoundError(f"Could not find input file: {INPUT_FILE}")

    print(f"Loading dataset from {INPUT_FILE} ...")
    df = pd.read_csv(INPUT_FILE, sep=r"\s+", compression="gzip")

    required_cols = {"protein1", "protein2", *FEATURE_COLS}
    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    print(f"Original rows: {len(df):,}")

    positives = df[df["combined_score"] >= POSITIVE_THRESHOLD].copy()

    negatives = df[
        (df["combined_score"] >= NEGATIVE_MIN_SCORE)
        & (df["combined_score"] <= NEGATIVE_MAX_SCORE)
    ].copy()

    print(f"Positive rows before cleaning: {len(positives):,}")
    print(f"Negative rows before cleaning: {len(negatives):,}")

    positives = clean_pairs(positives)
    negatives = clean_pairs(negatives)

    positives["label"] = 1
    negatives["label"] = 0

    print(f"Positive pairs after cleaning: {len(positives):,}")
    print(f"Negative pairs after cleaning: {len(negatives):,}")

    n_samples = min(len(positives), len(negatives))

    positives_balanced = positives.sample(
        n=n_samples,
        random_state=RANDOM_SEED,
    )

    negatives_balanced = negatives.sample(
        n=n_samples,
        random_state=RANDOM_SEED,
    )

    full_df = pd.concat(
        [positives_balanced, negatives_balanced],
        ignore_index=True,
    )

    full_df = full_df.sample(
        frac=1.0,
        random_state=RANDOM_SEED,
    ).reset_index(drop=True)

    positives_balanced.to_csv(OUTPUT_POSITIVES, index=False)
    negatives_balanced.to_csv(OUTPUT_NEGATIVES, index=False)
    full_df.to_csv(OUTPUT_DATASET, index=False)

    print("\nSaved files:")
    print(f"Hard positives: {OUTPUT_POSITIVES}")
    print(f"Hard negatives: {OUTPUT_NEGATIVES}")
    print(f"Hard balanced dataset: {OUTPUT_DATASET}")

    print("\nFinal label counts:")
    print(full_df["label"].value_counts())

    print("\nCombined score summary:")
    print(full_df.groupby("label")["combined_score"].describe())


if __name__ == "__main__":
    main()
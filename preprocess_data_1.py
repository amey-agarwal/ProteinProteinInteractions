import pandas as pd
from pathlib import Path


INPUT_FILE = "data/raw/9606.protein.links.detailed.v12.0.txt.gz"
OUTPUT_FILE = "data/processed/positive_pairs_clean.csv"
CONFIDENCE_THRESHOLD = 700


def canonicalize_pair(p1: str, p2: str) -> tuple[str, str]:
    """Sort a protein pair so (A, B) and (B, A) are treated the same."""
    return tuple(sorted((p1, p2)))


def load_and_preprocess(input_file: str, output_file: str, threshold: int = 700) -> pd.DataFrame:
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Could not find input file: {input_file}")

    print(f"Loading dataset from {input_file} ...")
    df = pd.read_csv(input_file, sep=r"\s+", compression="gzip")

    required_cols = {
        "protein1",
        "protein2",
        "neighborhood",
        "fusion",
        "cooccurence",
        "coexpression",
        "experimental",
        "database",
        "textmining",
        "combined_score",
    }

    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    print(f"Original rows: {len(df):,}")

    # Remove self-interactions
    df = df[df["protein1"] != df["protein2"]].copy()
    print(f"After removing self-pairs: {len(df):,}")

    # Filter by confidence threshold
    df = df[df["combined_score"] >= threshold].copy()
    print(f"After filtering combined_score >= {threshold}: {len(df):,}")

    # Canonicalize undirected pairs
    canon_pairs = df.apply(lambda row: canonicalize_pair(row["protein1"], row["protein2"]), axis=1)
    df[["protein1", "protein2"]] = pd.DataFrame(canon_pairs.tolist(), index=df.index)

    # Drop duplicate undirected pairs
    before_dupes = len(df)
    df = df.drop_duplicates(subset=["protein1", "protein2"]).copy()
    print(f"Removed duplicates: {before_dupes - len(df):,}")
    print(f"Final positive pairs: {len(df):,}")

    # Helpful stats
    unique_proteins = pd.unique(df[["protein1", "protein2"]].values.ravel("K"))
    print(f"Unique proteins: {len(unique_proteins):,}")

    print("\nCombined score summary:")
    print(df["combined_score"].describe())

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"\nSaved cleaned positives to: {output_file}")

    return df


if __name__ == "__main__":
    load_and_preprocess(INPUT_FILE, OUTPUT_FILE, CONFIDENCE_THRESHOLD)
import random
from pathlib import Path

import pandas as pd


INPUT_POSITIVES = "positive_pairs_clean.csv"
OUTPUT_DATASET = "ppi_dataset_with_negatives.csv"
RANDOM_SEED = 42


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


def canonicalize_pair(p1: str, p2: str) -> tuple[str, str]:
    return tuple(sorted((p1, p2)))


def build_positive_pair_set(df: pd.DataFrame) -> set[tuple[str, str]]:
    return set(zip(df["protein1"], df["protein2"]))


def generate_edge_swap_negatives(
    positives: pd.DataFrame,
    target_count: int,
    random_seed: int = 42,
) -> list[tuple[str, str]]:
    """
    Create negatives by selecting two positive edges:
    (a, b) and (c, d) -> propose (a, c) and (b, d)
    if they are not already known positives and are not self-pairs.
    """
    rng = random.Random(random_seed)
    pos_rows = positives[["protein1", "protein2"]].values.tolist()
    pos_set = build_positive_pair_set(positives)

    negatives = set()
    max_attempts = target_count * 50
    attempts = 0

    while len(negatives) < target_count and attempts < max_attempts:
        attempts += 1
        (a, b), (c, d) = rng.sample(pos_rows, 2)

        candidates = [
            canonicalize_pair(a, c),
            canonicalize_pair(b, d),
            canonicalize_pair(a, d),
            canonicalize_pair(b, c),
        ]
        rng.shuffle(candidates)

        for p1, p2 in candidates:
            if p1 == p2:
                continue
            if (p1, p2) in pos_set:
                continue
            if (p1, p2) in negatives:
                continue
            negatives.add((p1, p2))
            if len(negatives) >= target_count:
                break

    print(f"Generated {len(negatives):,} edge-swap negatives after {attempts:,} attempts.")
    return list(negatives)


def generate_random_negatives(
    positives: pd.DataFrame,
    existing_negatives: set[tuple[str, str]],
    target_count: int,
    random_seed: int = 42,
) -> list[tuple[str, str]]:
    """
    Fallback: randomly sample protein pairs not in positives or existing negatives.
    """
    rng = random.Random(random_seed)
    pos_set = build_positive_pair_set(positives)
    proteins = list(pd.unique(positives[["protein1", "protein2"]].values.ravel("K")))

    negatives = set(existing_negatives)
    new_negatives = set()

    max_attempts = target_count * 100
    attempts = 0

    while len(new_negatives) < target_count and attempts < max_attempts:
        attempts += 1
        p1, p2 = rng.sample(proteins, 2)
        pair = canonicalize_pair(p1, p2)

        if pair in pos_set or pair in negatives or pair in new_negatives:
            continue
        if pair[0] == pair[1]:
            continue

        new_negatives.add(pair)

    print(f"Generated {len(new_negatives):,} fallback random negatives after {attempts:,} attempts.")
    return list(new_negatives)


def build_negative_dataframe(negatives: list[tuple[str, str]]) -> pd.DataFrame:
    """
    Since STRING evidence is only present for known linked pairs, negatives created
    by sampling are assigned zero evidence features.
    """
    neg_df = pd.DataFrame(negatives, columns=["protein1", "protein2"])
    for col in FEATURE_COLS:
        neg_df[col] = 0
    return neg_df


def main():
    input_path = Path(INPUT_POSITIVES)
    if not input_path.exists():
        raise FileNotFoundError(
            f"Could not find {INPUT_POSITIVES}. Run preprocess_data.py first."
        )

    positives = pd.read_csv(INPUT_POSITIVES)

    required_cols = {"protein1", "protein2", *FEATURE_COLS}
    missing = required_cols - set(positives.columns)
    if missing:
        raise ValueError(f"Missing required columns in positives file: {missing}")

    positives = positives.copy()
    positives["label"] = 1

    target_negatives = len(positives)
    print(f"Target negative count: {target_negatives:,}")

    edge_swap_negatives = generate_edge_swap_negatives(
        positives,
        target_count=target_negatives,
        random_seed=RANDOM_SEED,
    )

    if len(edge_swap_negatives) < target_negatives:
        needed = target_negatives - len(edge_swap_negatives)
        fallback_negatives = generate_random_negatives(
            positives,
            existing_negatives=set(edge_swap_negatives),
            target_count=needed,
            random_seed=RANDOM_SEED,
        )
        all_negatives = edge_swap_negatives + fallback_negatives
    else:
        all_negatives = edge_swap_negatives[:target_negatives]

    negatives = build_negative_dataframe(all_negatives)
    negatives["label"] = 0

    full_df = pd.concat([positives, negatives], ignore_index=True)
    full_df = full_df.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)

    print(f"Positive samples: {len(positives):,}")
    print(f"Negative samples: {len(negatives):,}")
    print(f"Total dataset size: {len(full_df):,}")

    full_df.to_csv(OUTPUT_DATASET, index=False)
    print(f"Saved labeled dataset to: {OUTPUT_DATASET}")


if __name__ == "__main__":
    main()
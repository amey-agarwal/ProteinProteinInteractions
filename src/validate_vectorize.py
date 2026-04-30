"""Validate the vectorized build_pair_features against the original loop."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from run_ablation_experiments import (
    DATASET_PATH,
    EVIDENCE_FEATURES,
    NETWORK_EMBEDDINGS_PATH,
    SEQUENCE_EMBEDDINGS_PATH,
    build_embedding_matrix,
    build_pair_features,
    load_embeddings,
)


def build_pair_features_legacy(df, sequence_embeddings, network_embeddings, feature_mode):
    features = []
    labels = []
    missing_count = 0

    for _, row in df.iterrows():
        p1 = row["protein1"]
        p2 = row["protein2"]

        if feature_mode in {"sequence", "network", "combined", "sequence_network"}:
            if (
                p1 not in sequence_embeddings
                or p2 not in sequence_embeddings
                or p1 not in network_embeddings
                or p2 not in network_embeddings
            ):
                missing_count += 1
                continue

        parts = []

        if feature_mode in {"evidence", "combined"}:
            evidence = row[EVIDENCE_FEATURES].to_numpy(dtype=np.float32)
            parts.append(evidence)

        if feature_mode in {"sequence", "combined", "sequence_network"}:
            p1_seq = sequence_embeddings[p1]
            p2_seq = sequence_embeddings[p2]
            parts.append(np.abs(p1_seq - p2_seq))
            parts.append(p1_seq * p2_seq)

        if feature_mode in {"network", "combined", "sequence_network"}:
            p1_net = network_embeddings[p1]
            p2_net = network_embeddings[p2]
            parts.append(np.abs(p1_net - p2_net))
            parts.append(p1_net * p2_net)

        pair_features = np.concatenate(parts)
        features.append(pair_features)
        labels.append(row["label"])

    X = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    return X, y, missing_count


def main():
    print("Loading data...")
    df = pd.read_csv(DATASET_PATH).head(2000)

    seq_emb = load_embeddings(SEQUENCE_EMBEDDINGS_PATH)
    net_emb = load_embeddings(NETWORK_EMBEDDINGS_PATH)

    seq_lookup = build_embedding_matrix(seq_emb)
    net_lookup = build_embedding_matrix(net_emb)

    for mode in ["sequence", "network", "sequence_network"]:
        print(f"\n--- Validating {mode} ---")

        X_old, y_old, missing_old = build_pair_features_legacy(df, seq_emb, net_emb, mode)
        X_new, y_new = build_pair_features(df, seq_lookup, net_lookup, mode)

        assert X_old.shape == X_new.shape, f"Shape mismatch: {X_old.shape} vs {X_new.shape}"
        assert np.array_equal(y_old, y_new), "Labels differ"
        assert np.allclose(X_old, X_new, atol=1e-6), "Features differ"

        print(f"OK — shape={X_new.shape}, missing={missing_old}")

    print("\nAll modes match the legacy implementation.")


if __name__ == "__main__":
    main()

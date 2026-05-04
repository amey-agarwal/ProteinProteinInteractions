"""Sequence+network ablation with a 150K train cap to fit in memory.
Assembles final ablation_results.csv from all three saved curve files.
"""
import gc
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from run_ablation_experiments import (
    DATASET_PATH,
    NETWORK_EMBEDDINGS_PATH,
    RANDOM_SEED,
    RESULTS_OUTPUT,
    SEQUENCE_EMBEDDINGS_PATH,
    TEST_PROTEIN_FRACTION,
    build_embedding_matrix,
    build_pair_features,
    load_embeddings,
    protein_wise_split,
    sample_dataframe,
    train_model,
)

CURVES_DIR = Path("data/processed/curves")
SEQNET_TRAIN_CAP = 150_000
SEQNET_TEST_CAP = None


def metrics_from_npz(path: Path, mode: str) -> dict:
    data = np.load(path)
    y_test = data["y_test"]
    y_prob = data["y_prob"]
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "model": mode,
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "pr_auc": float(average_precision_score(y_test, y_prob)),
    }


def main():
    print("Loading data...")
    df = pd.read_csv(DATASET_PATH)

    seq_emb = load_embeddings(SEQUENCE_EMBEDDINGS_PATH)
    net_emb = load_embeddings(NETWORK_EMBEDDINGS_PATH)
    seq_lookup = build_embedding_matrix(seq_emb)
    net_lookup = build_embedding_matrix(net_emb)

    del seq_emb, net_emb
    gc.collect()

    train_df, test_df = protein_wise_split(df, TEST_PROTEIN_FRACTION, RANDOM_SEED)
    train_df = sample_dataframe(train_df, SEQNET_TRAIN_CAP, RANDOM_SEED)
    test_df = sample_dataframe(test_df, SEQNET_TEST_CAP, RANDOM_SEED)

    print(f"Train rows: {len(train_df):,}  (capped at {SEQNET_TRAIN_CAP:,})")
    print(f"Test rows:  {len(test_df):,}")

    print("\nBuilding sequence_network features...")
    X_train, y_train = build_pair_features(train_df, seq_lookup, net_lookup, "sequence_network")
    X_test, y_test = build_pair_features(test_df, seq_lookup, net_lookup, "sequence_network")

    del df, train_df, test_df, seq_lookup, net_lookup
    gc.collect()

    print(f"X_train: {X_train.shape} ({X_train.nbytes / 1e9:.2f} GB)")
    print(f"X_test:  {X_test.shape} ({X_test.nbytes / 1e9:.2f} GB)")

    print("\nTraining model...")
    model = train_model(X_train, y_train)

    del X_train, y_train
    gc.collect()

    print("\nEvaluating...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_prob)

    metrics = {
        "model": "sequence_network",
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "pr_auc": float(average_precision_score(y_test, y_prob)),
    }

    print(f"\nF1:      {metrics['f1']:.4f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"PR-AUC:  {metrics['pr_auc']:.4f}")

    CURVES_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        CURVES_DIR / "sequence_network.npz",
        fpr=fpr,
        tpr=tpr,
        precision=prec_curve,
        recall=rec_curve,
        roc_auc=metrics["roc_auc"],
        pr_auc=metrics["pr_auc"],
        y_test=y_test,
        y_prob=y_prob,
    )
    print(f"Saved curves to {CURVES_DIR / 'sequence_network.npz'}")

    print("\nBuilding combined results CSV...")
    rows = [
        metrics_from_npz(CURVES_DIR / "sequence.npz", "sequence"),
        metrics_from_npz(CURVES_DIR / "network.npz", "network"),
        metrics,
    ]
    results_df = pd.DataFrame(rows)
    results_df.to_csv(RESULTS_OUTPUT, index=False)

    print("\n" + "=" * 70)
    print("FINAL ABLATION RESULTS (full data; sequence_network capped at 150K)")
    print("=" * 70)
    print(results_df.to_string(index=False))
    print(f"\nSaved to {RESULTS_OUTPUT}")


if __name__ == "__main__":
    main()

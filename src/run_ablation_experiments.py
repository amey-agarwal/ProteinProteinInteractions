from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATASET_PATH = Path("data/processed/ppi_dataset_hard.csv")

SEQUENCE_EMBEDDINGS_PATH = Path("data/embeddings/9606.protein.sequence.embeddings.v12.0.h5")
NETWORK_EMBEDDINGS_PATH = Path("data/embeddings/9606.protein.network.embeddings.v12.0.h5")

RESULTS_OUTPUT = Path("data/processed/ablation_results.csv")
CURVES_DIR = Path("data/processed/curves")

RANDOM_SEED = 42
TEST_PROTEIN_FRACTION = 0.2

MAX_TRAIN_ROWS = None
MAX_TEST_ROWS = None

EVIDENCE_FEATURES = [
    "neighborhood",
    "fusion",
    "cooccurence",
    "coexpression",
    "experimental",
    "database",
    "textmining",
]


def load_embeddings(path: Path) -> dict[str, np.ndarray]:
    print(f"Loading embeddings from {path}...")

    with h5py.File(path, "r") as f:
        proteins = [
            p.decode("utf-8") if isinstance(p, bytes) else str(p)
            for p in f["proteins"][:]
        ]

        embeddings = f["embeddings"][:].astype(np.float32)

    return dict(zip(proteins, embeddings))


def build_embedding_matrix(
    embeddings: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, int]]:
    proteins = list(embeddings.keys())
    idx_map = {p: i for i, p in enumerate(proteins)}
    matrix = np.stack([embeddings[p] for p in proteins]).astype(np.float32)
    return matrix, idx_map


def protein_wise_split(
    df: pd.DataFrame,
    test_fraction: float = 0.2,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(random_seed)

    proteins = pd.unique(df[["protein1", "protein2"]].values.ravel("K"))
    proteins = np.array(proteins)
    rng.shuffle(proteins)

    n_test = max(1, int(len(proteins) * test_fraction))
    test_proteins = set(proteins[:n_test])

    is_p1_test = df["protein1"].isin(test_proteins)
    is_p2_test = df["protein2"].isin(test_proteins)

    test_mask = is_p1_test & is_p2_test
    train_mask = (~is_p1_test) & (~is_p2_test)

    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()

    print(f"Train rows before sampling: {len(train_df):,}")
    print(f"Test rows before sampling: {len(test_df):,}")
    print(f"Dropped mixed rows: {len(df) - len(train_df) - len(test_df):,}")

    return train_df, test_df


def sample_dataframe(df: pd.DataFrame, max_rows: int | None, random_seed: int) -> pd.DataFrame:
    if max_rows is None or len(df) <= max_rows:
        return df.reset_index(drop=True)

    return df.sample(n=max_rows, random_state=random_seed).reset_index(drop=True)


def build_pair_features(
    df: pd.DataFrame,
    sequence_lookup: tuple[np.ndarray, dict[str, int]],
    network_lookup: tuple[np.ndarray, dict[str, int]],
    feature_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    seq_matrix, seq_idx = sequence_lookup
    net_matrix, net_idx = network_lookup

    p1 = df["protein1"].to_numpy()
    p2 = df["protein2"].to_numpy()

    valid_mask = np.ones(len(df), dtype=bool)

    if feature_mode in {"sequence", "combined", "sequence_network"}:
        valid_mask &= np.fromiter((p in seq_idx for p in p1), dtype=bool, count=len(df))
        valid_mask &= np.fromiter((p in seq_idx for p in p2), dtype=bool, count=len(df))

    if feature_mode in {"network", "combined", "sequence_network"}:
        valid_mask &= np.fromiter((p in net_idx for p in p1), dtype=bool, count=len(df))
        valid_mask &= np.fromiter((p in net_idx for p in p2), dtype=bool, count=len(df))

    missing_count = int((~valid_mask).sum())
    df_valid = df.loc[valid_mask].reset_index(drop=True)
    p1_valid = df_valid["protein1"].to_numpy()
    p2_valid = df_valid["protein2"].to_numpy()

    parts: list[np.ndarray] = []

    if feature_mode in {"evidence", "combined"}:
        parts.append(df_valid[EVIDENCE_FEATURES].to_numpy(dtype=np.float32))

    if feature_mode in {"sequence", "combined", "sequence_network"}:
        p1_ix = np.fromiter((seq_idx[p] for p in p1_valid), dtype=np.int64, count=len(p1_valid))
        p2_ix = np.fromiter((seq_idx[p] for p in p2_valid), dtype=np.int64, count=len(p2_valid))
        p1_seq = seq_matrix[p1_ix]
        p2_seq = seq_matrix[p2_ix]
        parts.append(np.abs(p1_seq - p2_seq))
        parts.append(p1_seq * p2_seq)

    if feature_mode in {"network", "combined", "sequence_network"}:
        p1_ix = np.fromiter((net_idx[p] for p in p1_valid), dtype=np.int64, count=len(p1_valid))
        p2_ix = np.fromiter((net_idx[p] for p in p2_valid), dtype=np.int64, count=len(p2_valid))
        p1_net = net_matrix[p1_ix]
        p2_net = net_matrix[p2_ix]
        parts.append(np.abs(p1_net - p2_net))
        parts.append(p1_net * p2_net)

    if not parts:
        raise ValueError(f"No valid rows for feature_mode={feature_mode}")

    X = np.concatenate(parts, axis=1).astype(np.float32)
    y = df_valid["label"].to_numpy(dtype=np.int64)

    print(f"{feature_mode} feature matrix shape: {X.shape}")
    print(f"{feature_mode} missing/skipped pairs: {missing_count:,}")

    return X, y


def train_model(X_train: np.ndarray, y_train: np.ndarray) -> Pipeline:
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                MLPClassifier(
                    hidden_layer_sizes=(128, 64),
                    activation="relu",
                    solver="adam",
                    alpha=1e-4,
                    batch_size=256,
                    learning_rate_init=1e-3,
                    max_iter=30,
                    random_state=RANDOM_SEED,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=5,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)
    return model


def evaluate_model(
    model: Pipeline,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
) -> dict:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    results = {
        "model": model_name,
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "pr_auc": average_precision_score(y_test, y_prob),
    }

    print(f"\n=== {model_name.upper()} RESULTS ===")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1-score:  {results['f1']:.4f}")
    print(f"ROC-AUC:   {results['roc_auc']:.4f}")
    print(f"PR-AUC:    {results['pr_auc']:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4, zero_division=0))

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_prob)

    CURVES_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        CURVES_DIR / f"{model_name}.npz",
        fpr=fpr,
        tpr=tpr,
        precision=prec_curve,
        recall=rec_curve,
        roc_auc=results["roc_auc"],
        pr_auc=results["pr_auc"],
        y_test=y_test,
        y_prob=y_prob,
    )

    return results


def run_experiment(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    sequence_lookup: tuple[np.ndarray, dict[str, int]],
    network_lookup: tuple[np.ndarray, dict[str, int]],
    feature_mode: str,
) -> dict:
    print("\n" + "=" * 70)
    print(f"Running experiment: {feature_mode}")
    print("=" * 70)

    print("\nBuilding training features...")
    X_train, y_train = build_pair_features(
        train_df,
        sequence_lookup,
        network_lookup,
        feature_mode,
    )

    print("\nBuilding testing features...")
    X_test, y_test = build_pair_features(
        test_df,
        sequence_lookup,
        network_lookup,
        feature_mode,
    )

    print("\nTraining model...")
    model = train_model(X_train, y_train)

    return evaluate_model(model, X_test, y_test, feature_mode)


def main():
    RESULTS_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATASET_PATH)

    sequence_embeddings = load_embeddings(SEQUENCE_EMBEDDINGS_PATH)
    network_embeddings = load_embeddings(NETWORK_EMBEDDINGS_PATH)

    sequence_lookup = build_embedding_matrix(sequence_embeddings)
    network_lookup = build_embedding_matrix(network_embeddings)

    train_df, test_df = protein_wise_split(
        df,
        test_fraction=TEST_PROTEIN_FRACTION,
        random_seed=RANDOM_SEED,
    )

    train_df = sample_dataframe(train_df, MAX_TRAIN_ROWS, RANDOM_SEED)
    test_df = sample_dataframe(test_df, MAX_TEST_ROWS, RANDOM_SEED)

    print(f"Train rows after sampling: {len(train_df):,}")
    print(f"Test rows after sampling: {len(test_df):,}")

    feature_modes = [
        "sequence",
        "network",
        "sequence_network",
    ]

    all_results = []

    for mode in feature_modes:
        results = run_experiment(
            train_df=train_df,
            test_df=test_df,
            sequence_lookup=sequence_lookup,
            network_lookup=network_lookup,
            feature_mode=mode,
        )

        all_results.append(results)

    results_df = pd.DataFrame(all_results)

    print("\n" + "=" * 70)
    print("FINAL ABLATION RESULTS")
    print("=" * 70)
    print(results_df.to_string(index=False))

    results_df.to_csv(RESULTS_OUTPUT, index=False)
    print(f"\nSaved ablation results to: {RESULTS_OUTPUT}")


if __name__ == "__main__":
    main()

from pathlib import Path

import h5py
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATASET_PATH = Path("data/processed/ppi_dataset_hard.csv")

SEQUENCE_EMBEDDINGS_PATH = Path("data/embeddings/9606.protein.sequence.embeddings.v12.0.h5")
NETWORK_EMBEDDINGS_PATH = Path("data/embeddings/9606.protein.network.embeddings.v12.0.h5")

# MODEL_OUTPUT = Path("models/embedding_mlp_small.joblib")
MODEL_OUTPUT = Path("models/network_embedding_mlp.joblib")

RANDOM_SEED = 42
TEST_PROTEIN_FRACTION = 0.2

MAX_TRAIN_ROWS = 50_000
MAX_TEST_ROWS = 10_000

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


def sample_dataframe(df: pd.DataFrame, max_rows: int, random_seed: int) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df

    return df.sample(n=max_rows, random_state=random_seed).reset_index(drop=True)


def build_pair_features(
    df: pd.DataFrame,
    sequence_embeddings: dict[str, np.ndarray],
    network_embeddings: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    features = []
    labels = []

    missing_count = 0

    for _, row in df.iterrows():
        p1 = row["protein1"]
        p2 = row["protein2"]

        if (
            p1 not in sequence_embeddings
            or p2 not in sequence_embeddings
            or p1 not in network_embeddings
            or p2 not in network_embeddings
        ):
            missing_count += 1
            continue

        # p1_seq = sequence_embeddings[p1]
        # p2_seq = sequence_embeddings[p2]

        p1_net = network_embeddings[p1]
        p2_net = network_embeddings[p2]

        # evidence = row[EVIDENCE_FEATURES].to_numpy(dtype=np.float32)

        # pair_features = np.concatenate(
        #     [
        #         evidence,
        #         np.abs(p1_seq - p2_seq),
        #         p1_seq * p2_seq,
        #         np.abs(p1_net - p2_net),
        #         p1_net * p2_net,
        #     ]
        # )

        p1_net = network_embeddings[p1]
        p2_net = network_embeddings[p2]

        pair_features = np.concatenate(
            [
                np.abs(p1_net - p2_net),
                p1_net * p2_net,
            ]
        )

        features.append(pair_features)
        labels.append(row["label"])

    if not features:
        raise ValueError("No valid protein pairs had matching embeddings.")

    X = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)

    print(f"Feature matrix shape: {X.shape}")
    print(f"Missing/skipped pairs: {missing_count:,}")

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
                    verbose=True,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)
    return model


def evaluate_model(model: Pipeline, X_test: np.ndarray, y_test: np.ndarray) -> None:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\n=== Small Embedding Model Results ===")
    print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"F1-score:  {f1_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(y_test, y_prob):.4f}")
    print(f"PR-AUC:    {average_precision_score(y_test, y_prob):.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4, zero_division=0))


def main():
    MODEL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATASET_PATH)

    sequence_embeddings = load_embeddings(SEQUENCE_EMBEDDINGS_PATH)
    network_embeddings = load_embeddings(NETWORK_EMBEDDINGS_PATH)

    train_df, test_df = protein_wise_split(
        df,
        test_fraction=TEST_PROTEIN_FRACTION,
        random_seed=RANDOM_SEED,
    )

    train_df = sample_dataframe(train_df, MAX_TRAIN_ROWS, RANDOM_SEED)
    test_df = sample_dataframe(test_df, MAX_TEST_ROWS, RANDOM_SEED)

    print(f"Train rows after sampling: {len(train_df):,}")
    print(f"Test rows after sampling: {len(test_df):,}")

    print("\nBuilding training features...")
    X_train, y_train = build_pair_features(
        train_df,
        sequence_embeddings,
        network_embeddings,
    )

    print("\nBuilding testing features...")
    X_test, y_test = build_pair_features(
        test_df,
        sequence_embeddings,
        network_embeddings,
    )

    print("\nTraining small embedding model...")
    model = train_model(X_train, y_train)

    evaluate_model(model, X_test, y_test)

    joblib.dump(model, MODEL_OUTPUT)
    print(f"\nSaved model to: {MODEL_OUTPUT}")


if __name__ == "__main__":
    main()
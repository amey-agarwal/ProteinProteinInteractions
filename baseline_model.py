from pathlib import Path

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


INPUT_DATASET = "ppi_dataset_with_negatives.csv"
MODEL_OUTPUT = "baseline_mlp.joblib"
RANDOM_SEED = 42
TEST_PROTEIN_FRACTION = 0.2

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


def protein_wise_split(
    df: pd.DataFrame,
    test_fraction: float = 0.2,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Protein-wise split:
    - choose a subset of proteins for test
    - test set contains only pairs where both proteins are in the test protein set
    - train set contains only pairs where both proteins are outside the test protein set
    - mixed pairs are dropped to avoid leakage
    """
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

    dropped = len(df) - len(train_df) - len(test_df)

    print(f"Total rows: {len(df):,}")
    print(f"Train rows: {len(train_df):,}")
    print(f"Test rows: {len(test_df):,}")
    print(f"Dropped mixed rows: {dropped:,}")

    if len(train_df) == 0 or len(test_df) == 0:
        raise ValueError(
            "Protein-wise split produced an empty train or test set. "
            "Try adjusting the confidence threshold or test fraction."
        )

    return train_df, test_df


def train_baseline_model(train_df: pd.DataFrame) -> Pipeline:
    X_train = train_df[FEATURE_COLS]
    y_train = train_df["label"]

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                MLPClassifier(
                    hidden_layer_sizes=(64, 32),
                    activation="relu",
                    solver="adam",
                    alpha=1e-4,
                    batch_size=256,
                    learning_rate_init=1e-3,
                    max_iter=50,
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


def evaluate_model(model: Pipeline, test_df: pd.DataFrame) -> None:
    X_test = test_df[FEATURE_COLS]
    y_test = test_df["label"]

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)

    print("\n=== Baseline Model Results ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print(f"PR-AUC:    {pr_auc:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4, zero_division=0))


def main():
    input_path = Path(INPUT_DATASET)
    if not input_path.exists():
        raise FileNotFoundError(
            f"Could not find {INPUT_DATASET}. Run preprocess_data.py and negative_sampling.py first."
        )

    df = pd.read_csv(INPUT_DATASET)

    required_cols = {"protein1", "protein2", "label", *FEATURE_COLS}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in dataset: {missing}")

    train_df, test_df = protein_wise_split(
        df,
        test_fraction=TEST_PROTEIN_FRACTION,
        random_seed=RANDOM_SEED,
    )

    print("\nTraining baseline MLP model...")
    model = train_baseline_model(train_df)

    evaluate_model(model, test_df)

    joblib.dump(model, MODEL_OUTPUT)
    print(f"\nSaved trained model to: {MODEL_OUTPUT}")


if __name__ == "__main__":
    main()
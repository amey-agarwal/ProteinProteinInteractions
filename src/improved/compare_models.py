from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
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
from xgboost import XGBClassifier

DATASET_PATH = Path("data/processed/ppi_dataset_hard.csv")
NETWORK_EMBEDDINGS_PATH = Path("data/embeddings/9606.protein.network.embeddings.v12.0.h5")
RESULTS_OUTPUT = Path("data/processed/model_comparison_results.csv")
PLOTS_DIR = Path("plots")
RANDOM_SEED = 42
TEST_PROTEIN_FRACTION = 0.2
MAX_TRAIN_ROWS = 50_000
MAX_TEST_ROWS = 10_000


def load_embeddings(path):
    with h5py.File(path, "r") as f:
        proteins = [p.decode("utf-8") if isinstance(p, bytes) else str(p) for p in f["proteins"][:]]
        embeddings = f["embeddings"][:].astype(np.float32)
    return dict(zip(proteins, embeddings))


def protein_wise_split(df):
    rng = np.random.default_rng(RANDOM_SEED)
    proteins = np.array(pd.unique(df[["protein1", "protein2"]].values.ravel("K")))
    rng.shuffle(proteins)
    n_test = max(1, int(len(proteins) * TEST_PROTEIN_FRACTION))
    test_proteins = set(proteins[:n_test])
    is_p1_test = df["protein1"].isin(test_proteins)
    is_p2_test = df["protein2"].isin(test_proteins)
    train_df = df[(~is_p1_test) & (~is_p2_test)].copy()
    test_df = df[is_p1_test & is_p2_test].copy()
    return train_df, test_df


def build_features(df, network_embeddings):
    features, labels = [], []
    for _, row in df.iterrows():
        p1, p2 = row["protein1"], row["protein2"]
        if p1 not in network_embeddings or p2 not in network_embeddings:
            continue
        e1 = network_embeddings[p1]
        e2 = network_embeddings[p2]
        features.append(np.concatenate([np.abs(e1 - e2), e1 * e2]))
        labels.append(row["label"])
    return np.array(features, dtype=np.float32), np.array(labels, dtype=np.int64)


def get_models():
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)),
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100, random_state=RANDOM_SEED, eval_metric="logloss", verbosity=0
        ),
        "MLP": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(
                hidden_layer_sizes=(128, 64),
                max_iter=30,
                random_state=RANDOM_SEED,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=5,
            )),
        ]),
    }


def evaluate(model, X_test, y_test, name):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return {
        "model": name,
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "pr_auc": average_precision_score(y_test, y_prob),
    }, y_pred, y_prob


def plot_roc_curves(outputs, y_test):
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, y_prob in outputs:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Models")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "roc_curves.png", dpi=150)
    plt.close(fig)
    print("Saved plots/roc_curves.png")


def plot_pr_curves(outputs, y_test):
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, y_prob in outputs:
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)
        ax.plot(recall, precision, label=f"{name} (PR-AUC={pr_auc:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves — All Models")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "pr_curves.png", dpi=150)
    plt.close(fig)
    print("Saved plots/pr_curves.png")


def plot_confusion_matrices(outputs, y_test):
    n = len(outputs)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    for ax, (name, y_pred) in zip(axes, outputs):
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=["Negative", "Positive"])
        disp.plot(ax=ax, colorbar=False)
        ax.set_title(name)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "confusion_matrices.png", dpi=150)
    plt.close(fig)
    print("Saved plots/confusion_matrices.png")


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATASET_PATH)
    network_embeddings = load_embeddings(NETWORK_EMBEDDINGS_PATH)

    train_df, test_df = protein_wise_split(df)

    if len(train_df) > MAX_TRAIN_ROWS:
        train_df = train_df.sample(n=MAX_TRAIN_ROWS, random_state=RANDOM_SEED).reset_index(drop=True)
    if len(test_df) > MAX_TEST_ROWS:
        test_df = test_df.sample(n=MAX_TEST_ROWS, random_state=RANDOM_SEED).reset_index(drop=True)

    print(f"Train: {len(train_df):,}  |  Test: {len(test_df):,}")

    X_train, y_train = build_features(train_df, network_embeddings)
    X_test, y_test = build_features(test_df, network_embeddings)

    print(f"Feature shape: {X_train.shape}")

    models = get_models()
    all_results = []
    roc_outputs = []
    pr_outputs = []
    cm_outputs = []

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        result, y_pred, y_prob = evaluate(model, X_test, y_test, name)
        all_results.append(result)
        roc_outputs.append((name, y_prob))
        pr_outputs.append((name, y_prob))
        cm_outputs.append((name, y_pred))
        print(f"  Precision={result['precision']:.4f}  Recall={result['recall']:.4f}  "
              f"F1={result['f1']:.4f}  ROC-AUC={result['roc_auc']:.4f}  PR-AUC={result['pr_auc']:.4f}")

    results_df = pd.DataFrame(all_results)
    print("\n" + "=" * 70)
    print(results_df.to_string(index=False))
    results_df.to_csv(RESULTS_OUTPUT, index=False)
    print(f"\nSaved results to {RESULTS_OUTPUT}")

    plot_roc_curves(roc_outputs, y_test)
    plot_pr_curves(pr_outputs, y_test)
    plot_confusion_matrices(cm_outputs, y_test)


if __name__ == "__main__":
    main()

from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from xgboost import XGBClassifier

DATASET_PATH = Path("data/processed/ppi_dataset_hard.csv")
NETWORK_EMBEDDINGS_PATH = Path("data/embeddings/9606.protein.network.embeddings.v12.0.h5")
PLOTS_DIR = Path("plots")
RANDOM_SEED = 42
TEST_PROTEIN_FRACTION = 0.2
MAX_TRAIN_ROWS = 50_000
MAX_TEST_ROWS = 10_000
SHAP_SAMPLE = 500


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


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATASET_PATH)
    network_embeddings = load_embeddings(NETWORK_EMBEDDINGS_PATH)

    train_df, test_df = protein_wise_split(df)

    if len(train_df) > MAX_TRAIN_ROWS:
        train_df = train_df.sample(n=MAX_TRAIN_ROWS, random_state=RANDOM_SEED).reset_index(drop=True)
    if len(test_df) > MAX_TEST_ROWS:
        test_df = test_df.sample(n=MAX_TEST_ROWS, random_state=RANDOM_SEED).reset_index(drop=True)

    X_train, y_train = build_features(train_df, network_embeddings)
    X_test, y_test = build_features(test_df, network_embeddings)

    print("Training XGBoost for SHAP analysis...")
    model = XGBClassifier(
        n_estimators=100, random_state=RANDOM_SEED, eval_metric="logloss", verbosity=0
    )
    model.fit(X_train, y_train)

    n_emb = X_train.shape[1] // 2
    feature_names = (
        [f"abs_diff_{i}" for i in range(n_emb)] +
        [f"product_{i}" for i in range(n_emb)]
    )

    X_sample = X_test[:SHAP_SAMPLE]

    print(f"Computing SHAP values on {SHAP_SAMPLE} test samples...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved plots/shap_summary.png")

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[::-1][:20]
    top_df = pd.DataFrame({
        "feature": [feature_names[i] for i in top_indices],
        "mean_abs_shap": mean_abs_shap[top_indices],
    })
    top_df.to_csv(PLOTS_DIR / "shap_top_features.csv", index=False)
    print("Saved plots/shap_top_features.csv")

    print("\nTop 10 most important features:")
    print(top_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()

from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

DATASET_PATH = Path("data/processed/ppi_dataset_hard.csv")
SEQUENCE_EMBEDDINGS_PATH = Path("data/embeddings/9606.protein.sequence.embeddings.v12.0.h5")
NETWORK_EMBEDDINGS_PATH = Path("data/embeddings/9606.protein.network.embeddings.v12.0.h5")
PROTEIN_INFO_PATH = Path("data/raw/9606.protein.info.v12.0.txt")
PLOTS_DIR = Path("plots")
CASE_STUDY_OUTPUT = Path("data/processed/case_study_examples.csv")
RANDOM_SEED = 42
TEST_PROTEIN_FRACTION = 0.2
MAX_TRAIN_ROWS = 200_000
MAX_TEST_ROWS = 30_000
NET_PAIR_DIM = 1024
SEQ_PAIR_DIM = 2048
BATCH_SIZE = 512
MAX_EPOCHS = 100
NUM_EXAMPLES = 10


class PPINet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net_tower = nn.Sequential(
            nn.Linear(NET_PAIR_DIM, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.seq_tower = nn.Sequential(
            nn.Linear(SEQ_PAIR_DIM, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.head = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x_net = x[:, :NET_PAIR_DIM]
        x_seq = x[:, NET_PAIR_DIM:]
        h_net = self.net_tower(x_net)
        h_seq = self.seq_tower(x_seq)
        return self.head(torch.cat([h_net, h_seq], dim=1)).squeeze(-1)


def load_embeddings(path):
    with h5py.File(path, "r") as f:
        proteins = [p.decode("utf-8") if isinstance(p, bytes) else str(p) for p in f["proteins"][:]]
        embeddings = f["embeddings"][:].astype(np.float32)
    return dict(zip(proteins, embeddings))


def load_protein_names():
    df = pd.read_csv(PROTEIN_INFO_PATH, sep="\t")
    return dict(zip(df["#string_protein_id"], df["preferred_name"]))


def protein_wise_split(df):
    rng = np.random.default_rng(RANDOM_SEED)
    proteins = np.array(pd.unique(df[["protein1", "protein2"]].values.ravel("K")))
    rng.shuffle(proteins)
    n_test = max(1, int(len(proteins) * TEST_PROTEIN_FRACTION))
    test_proteins = set(proteins[:n_test])
    is_p1_test = df["protein1"].isin(test_proteins)
    is_p2_test = df["protein2"].isin(test_proteins)
    return df[(~is_p1_test) & (~is_p2_test)].copy(), df[is_p1_test & is_p2_test].copy()


def build_features_with_metadata(df, seq_emb, net_emb):
    valid = df[
        df["protein1"].isin(net_emb) & df["protein2"].isin(net_emb) &
        df["protein1"].isin(seq_emb) & df["protein2"].isin(seq_emb)
    ].copy()
    p1s, p2s = valid["protein1"].values, valid["protein2"].values
    n1 = np.array([net_emb[p] for p in p1s], dtype=np.float32)
    n2 = np.array([net_emb[p] for p in p2s], dtype=np.float32)
    s1 = np.array([seq_emb[p] for p in p1s], dtype=np.float32)
    s2 = np.array([seq_emb[p] for p in p2s], dtype=np.float32)
    X = np.concatenate([np.abs(n1 - n2), n1 * n2, np.abs(s1 - s2), s1 * s2], axis=1)
    y = valid["label"].values.astype(np.float32)
    scores = valid["combined_score"].values if "combined_score" in valid.columns else np.zeros(len(valid))
    pairs = list(zip(p1s, p2s, scores))
    return X, y, pairs


def train_model(X_train, y_train, device):
    val_n = int(len(X_train) * 0.1)
    idx = np.random.RandomState(RANDOM_SEED).permutation(len(X_train))
    val_idx, tr_idx = idx[:val_n], idx[val_n:]

    mean = X_train[tr_idx].mean(axis=0)
    std = X_train[tr_idx].std(axis=0) + 1e-8
    X_tr = (X_train[tr_idx] - mean) / std
    X_vl = (X_train[val_idx] - mean) / std

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_tr), torch.tensor(y_train[tr_idx])),
        batch_size=BATCH_SIZE, shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_vl), torch.tensor(y_train[val_idx])),
        batch_size=BATCH_SIZE,
    )

    model = PPINet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                v_loss += criterion(model(xb), yb).item() * len(xb)
        v_loss /= len(val_loader.dataset)
        scheduler.step(v_loss)

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= 15:
                break

    model.load_state_dict(best_state)
    return model, mean, std


def predict(model, X, mean, std, device):
    model.eval()
    X_norm = (X - mean) / std
    with torch.no_grad():
        logits = model(torch.tensor(X_norm, dtype=torch.float32).to(device))
        probs = torch.sigmoid(logits).cpu().numpy()
    return probs


def build_case_table(pairs, y_true, y_prob, protein_names, category, n=NUM_EXAMPLES):
    rows = []
    for i, (p1, p2, score) in enumerate(pairs):
        name1 = protein_names.get(p1, p1)
        name2 = protein_names.get(p2, p2)
        rows.append({
            "protein1_id": p1,
            "protein1_name": name1,
            "protein2_id": p2,
            "protein2_name": name2,
            "true_label": int(y_true[i]),
            "predicted_prob": round(float(y_prob[i]), 4),
            "predicted_label": int(y_prob[i] >= 0.5),
            "combined_score": int(score),
            "category": category,
        })
    df = pd.DataFrame(rows)
    df["confidence"] = (df["predicted_prob"] - 0.5).abs()
    return df


def main():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    CASE_STUDY_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATASET_PATH)
    seq_emb = load_embeddings(SEQUENCE_EMBEDDINGS_PATH)
    net_emb = load_embeddings(NETWORK_EMBEDDINGS_PATH)
    protein_names = load_protein_names()

    train_df, test_df = protein_wise_split(df)

    if len(train_df) > MAX_TRAIN_ROWS:
        train_df = train_df.sample(n=MAX_TRAIN_ROWS, random_state=RANDOM_SEED).reset_index(drop=True)
    if len(test_df) > MAX_TEST_ROWS:
        test_df = test_df.sample(n=MAX_TEST_ROWS, random_state=RANDOM_SEED).reset_index(drop=True)

    X_train, y_train, _ = build_features_with_metadata(train_df, seq_emb, net_emb)
    X_test, y_test, test_pairs = build_features_with_metadata(test_df, seq_emb, net_emb)

    print("Training model for case study...")
    model, mean, std = train_model(X_train, y_train, device)
    y_prob = predict(model, X_test, mean, std, device)
    y_pred = (y_prob >= 0.5).astype(int)
    y_true = y_test.astype(int)

    print(f"Test F1={f1_score(y_true, y_pred):.4f}  ROC-AUC={roc_auc_score(y_true, y_prob):.4f}")

    full_table = build_case_table(test_pairs, y_true, y_prob, protein_names, "all")

    correct = full_table[full_table["true_label"] == full_table["predicted_label"]]
    wrong = full_table[full_table["true_label"] != full_table["predicted_label"]]

    tp = correct[correct["true_label"] == 1].nlargest(NUM_EXAMPLES, "confidence")
    tn = correct[correct["true_label"] == 0].nlargest(NUM_EXAMPLES, "confidence")
    fp = wrong[wrong["predicted_label"] == 1].nlargest(NUM_EXAMPLES, "confidence")
    fn = wrong[wrong["predicted_label"] == 0].nlargest(NUM_EXAMPLES, "confidence")

    tp["category"] = "True Positive (high confidence)"
    tn["category"] = "True Negative (high confidence)"
    fp["category"] = "False Positive (high confidence)"
    fn["category"] = "False Negative (high confidence)"

    examples = pd.concat([tp, tn, fp, fn], ignore_index=True)
    display_cols = [
        "category", "protein1_name", "protein2_name",
        "true_label", "predicted_prob", "combined_score",
    ]
    examples[display_cols].to_csv(CASE_STUDY_OUTPUT, index=False)
    print(f"\nSaved {len(examples)} case study examples to {CASE_STUDY_OUTPUT}")

    for cat in examples["category"].unique():
        subset = examples[examples["category"] == cat]
        print(f"\n--- {cat} ---")
        for _, row in subset.iterrows():
            print(f"  {row['protein1_name']:>10s} — {row['protein2_name']:<10s}  "
                  f"prob={row['predicted_prob']:.3f}  score={row['combined_score']}")

    error_df = full_table.copy()
    error_df["correct"] = error_df["true_label"] == error_df["predicted_label"]

    bins = [0, 400, 500, 600, 700, 800, 900, 1000]
    error_df["score_bin"] = pd.cut(error_df["combined_score"], bins=bins)
    acc_by_score = error_df.groupby("score_bin", observed=True)["correct"].mean()

    fig, ax = plt.subplots(figsize=(8, 5))
    acc_by_score.plot(kind="bar", ax=ax, color="#4C72B0")
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Combined Score Range")
    ax.set_title("Model Accuracy by STRING Combined Score")
    ax.set_ylim(0, 1)
    for i, v in enumerate(acc_by_score.values):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "accuracy_by_score.png", dpi=150)
    plt.close(fig)
    print("\nSaved plots/accuracy_by_score.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    correct_probs = full_table[full_table["true_label"] == full_table["predicted_label"]]["predicted_prob"]
    wrong_probs = full_table[full_table["true_label"] != full_table["predicted_label"]]["predicted_prob"]
    ax.hist(correct_probs, bins=50, alpha=0.6, label=f"Correct ({len(correct_probs)})", color="#55A868")
    ax.hist(wrong_probs, bins=50, alpha=0.6, label=f"Wrong ({len(wrong_probs)})", color="#C44E52")
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Count")
    ax.set_title("Prediction Confidence Distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "confidence_distribution.png", dpi=150)
    plt.close(fig)
    print("Saved plots/confidence_distribution.png")


if __name__ == "__main__":
    main()

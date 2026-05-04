"""Two-Tower MLP with cross-tower multi-head self-attention fusion."""
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset

DATASET_PATH = Path("data/processed/ppi_dataset_hard.csv")
SEQUENCE_EMBEDDINGS_PATH = Path("data/embeddings/9606.protein.sequence.embeddings.v12.0.h5")
NETWORK_EMBEDDINGS_PATH = Path("data/embeddings/9606.protein.network.embeddings.v12.0.h5")
PROTEIN_INFO_PATH = Path("data/raw/9606.protein.info.v12.0.txt")
MODEL_OUTPUT = Path("models/attention_mlp.pt")
REPORT_OUTPUT = Path("data/processed/attention_analysis.md")
PLOT_OUTPUT = Path("plots/attention_weights.png")

RANDOM_SEED = 42
TEST_PROTEIN_FRACTION = 0.2
MAX_TRAIN_ROWS = 200_000
MAX_TEST_ROWS = 30_000
BATCH_SIZE = 512
MAX_EPOCHS = 100
LR = 5e-4
WEIGHT_DECAY = 0.01
PATIENCE = 15
VAL_FRACTION = 0.1
NET_PAIR_DIM = 1024
SEQ_PAIR_DIM = 2048
TOWER_DIM = 128
NUM_HEADS = 4


class AttentionPPINet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net_tower = nn.Sequential(
            nn.Linear(NET_PAIR_DIM, TOWER_DIM),
            nn.BatchNorm1d(TOWER_DIM),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.seq_tower = nn.Sequential(
            nn.Linear(SEQ_PAIR_DIM, TOWER_DIM),
            nn.BatchNorm1d(TOWER_DIM),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=TOWER_DIM,
            num_heads=NUM_HEADS,
            dropout=0.1,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(TOWER_DIM)
        self.head = nn.Sequential(
            nn.Linear(TOWER_DIM * 2, TOWER_DIM),
            nn.BatchNorm1d(TOWER_DIM),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(TOWER_DIM, 1),
        )

    def forward(self, x, return_attention=False):
        x_net = x[:, :NET_PAIR_DIM]
        x_seq = x[:, NET_PAIR_DIM:]

        h_net = self.net_tower(x_net)
        h_seq = self.seq_tower(x_seq)

        tokens = torch.stack([h_net, h_seq], dim=1)
        attn_out, attn_weights = self.attention(tokens, tokens, tokens)
        tokens = self.attn_norm(tokens + attn_out)
        h = tokens.reshape(tokens.size(0), -1)

        logits = self.head(h).squeeze(-1)

        if return_attention:
            return logits, attn_weights
        return logits


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


def build_features(df, seq_emb, net_emb):
    valid = df[
        df["protein1"].isin(net_emb) & df["protein2"].isin(net_emb) &
        df["protein1"].isin(seq_emb) & df["protein2"].isin(seq_emb)
    ]
    p1s, p2s = valid["protein1"].values, valid["protein2"].values
    n1 = np.array([net_emb[p] for p in p1s], dtype=np.float32)
    n2 = np.array([net_emb[p] for p in p2s], dtype=np.float32)
    s1 = np.array([seq_emb[p] for p in p1s], dtype=np.float32)
    s2 = np.array([seq_emb[p] for p in p2s], dtype=np.float32)
    X = np.concatenate([np.abs(n1 - n2), n1 * n2, np.abs(s1 - s2), s1 * s2], axis=1)
    y = valid["label"].values.astype(np.float32)
    pairs = valid[["protein1", "protein2"]].reset_index(drop=True)
    return X, y, pairs


def normalize_features(X_train, X_val, X_test):
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    return (X_train - mean) / std, (X_val - mean) / std, (X_test - mean) / std


def train_model(model, train_loader, val_loader, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * len(X_batch)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                val_loss += criterion(model(X_batch), y_batch).item() * len(X_batch)
        val_loss /= len(val_loader.dataset)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}  train={train_loss:.4f}  val={val_loss:.4f}  lr={current_lr:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_state)
    return model


def evaluate(model, X_test, y_test, device):
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X_test, dtype=torch.float32).to(device))
        y_prob = torch.sigmoid(logits).cpu().numpy()

    y_pred = (y_prob >= 0.5).astype(int)
    y_true = y_test.astype(int)

    results = {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
    }

    print("\n=== Attention MLP Results ===")
    for k, v in results.items():
        print(f"  {k:>12s}: {v:.4f}")
    print("\n" + classification_report(y_true, y_pred, digits=4, zero_division=0))
    return results, y_prob, y_pred


def extract_attention_weights(model, X_test, device, batch_size=1024):
    model.eval()
    all_weights = []
    X_tensor = torch.tensor(X_test, dtype=torch.float32)
    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            batch = X_tensor[i:i + batch_size].to(device)
            _, attn_w = model(batch, return_attention=True)
            all_weights.append(attn_w.cpu().numpy())
    return np.concatenate(all_weights, axis=0)


def analyze_attention(attn_weights, y_true, y_pred, y_prob, test_pairs, info_df):
    correct = y_pred == y_true
    categories = {
        "All predictions": np.ones(len(y_true), dtype=bool),
        "Correct positives": correct & (y_true == 1),
        "Correct negatives": correct & (y_true == 0),
        "False positives": ~correct & (y_true == 0),
        "False negatives": ~correct & (y_true == 1),
        "High confidence (>0.9)": y_prob > 0.9,
        "Low confidence (0.4-0.6)": (y_prob >= 0.4) & (y_prob <= 0.6),
    }

    labels = ["net→net", "net→seq", "seq→net", "seq→seq"]

    lines = ["## Attention Weight Analysis\n"]
    lines.append("The 2x2 attention matrix shows how each tower queries the other:")
    lines.append("- **net→net**: network tower attending to itself")
    lines.append("- **net→seq**: network tower attending to sequence")
    lines.append("- **seq→net**: sequence tower attending to network")
    lines.append("- **seq→seq**: sequence tower attending to itself\n")

    lines.append("| Category | Count | net→net | net→seq | seq→net | seq→seq |")
    lines.append("|---|---|---|---|---|---|")

    category_means = {}
    for name, mask in categories.items():
        if mask.sum() == 0:
            continue
        w = attn_weights[mask]
        mean_w = w.mean(axis=0)  # (2, 2)
        category_means[name] = mean_w
        lines.append(
            f"| {name} | {mask.sum():,} | "
            f"{mean_w[0,0]:.4f} | {mean_w[0,1]:.4f} | "
            f"{mean_w[1,0]:.4f} | {mean_w[1,1]:.4f} |"
        )

    all_w = attn_weights.mean(axis=0)
    net_self = all_w[0, 0]
    net_cross = all_w[0, 1]
    seq_self = all_w[1, 1]
    seq_cross = all_w[1, 0]
    net_reliance = (net_self + seq_cross) / 2
    seq_reliance = (seq_self + net_cross) / 2

    lines.append(f"\n**Overall tower reliance:**")
    lines.append(f"- Network tower receives {net_reliance:.1%} of attention")
    lines.append(f"- Sequence tower receives {seq_reliance:.1%} of attention")

    if "Correct positives" in category_means and "False negatives" in category_means:
        cp = category_means["Correct positives"]
        fn = category_means["False negatives"]
        lines.append(f"\n**Correct positives vs false negatives:**")
        lines.append(f"- Correct positives: net→seq = {cp[0,1]:.4f}, seq→net = {cp[1,0]:.4f}")
        lines.append(f"- False negatives: net→seq = {fn[0,1]:.4f}, seq→net = {fn[1,0]:.4f}")
        cross_diff = (cp[0, 1] + cp[1, 0]) / 2 - (fn[0, 1] + fn[1, 0]) / 2
        if cross_diff > 0:
            lines.append(f"- Cross-tower attention is {abs(cross_diff):.4f} higher for correct positives")
        else:
            lines.append(f"- Cross-tower attention is {abs(cross_diff):.4f} lower for correct positives")

    return "\n".join(lines), category_means


def plot_attention(category_means, output_path):
    categories = [k for k in category_means if k in [
        "All predictions", "Correct positives", "Correct negatives",
        "False positives", "False negatives"
    ]]

    fig, axes = plt.subplots(1, len(categories), figsize=(4 * len(categories), 3.5))
    if len(categories) == 1:
        axes = [axes]

    token_labels = ["Network", "Sequence"]

    for ax, name in zip(axes, categories):
        w = category_means[name]
        im = ax.imshow(w, cmap="YlOrRd", vmin=0, vmax=1)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(token_labels, fontsize=9)
        ax.set_yticklabels(token_labels, fontsize=9)
        ax.set_xlabel("Key", fontsize=9)
        ax.set_ylabel("Query", fontsize=9)
        ax.set_title(name, fontsize=10, fontweight="bold")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{w[i, j]:.3f}", ha="center", va="center",
                        fontsize=11, fontweight="bold",
                        color="white" if w[i, j] > 0.6 else "black")

    plt.suptitle("Attention Weights: How Towers Attend to Each Other", fontsize=12, y=1.02)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved attention plot to {output_path}")


def main():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    MODEL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    print("\n[1/6] Loading data...")
    df = pd.read_csv(DATASET_PATH)
    seq_emb = load_embeddings(SEQUENCE_EMBEDDINGS_PATH)
    net_emb = load_embeddings(NETWORK_EMBEDDINGS_PATH)

    train_df, test_df = protein_wise_split(df)
    if len(train_df) > MAX_TRAIN_ROWS:
        train_df = train_df.sample(n=MAX_TRAIN_ROWS, random_state=RANDOM_SEED).reset_index(drop=True)
    if len(test_df) > MAX_TEST_ROWS:
        test_df = test_df.sample(n=MAX_TEST_ROWS, random_state=RANDOM_SEED).reset_index(drop=True)
    print(f"  Train: {len(train_df):,}  |  Test: {len(test_df):,}")

    print("[2/6] Building features...")
    X_train_full, y_train_full, _ = build_features(train_df, seq_emb, net_emb)
    X_test, y_test, test_pairs = build_features(test_df, seq_emb, net_emb)

    n_val = int(len(X_train_full) * VAL_FRACTION)
    indices = np.random.permutation(len(X_train_full))
    val_idx, train_idx = indices[:n_val], indices[n_val:]
    X_train, y_train = X_train_full[train_idx], y_train_full[train_idx]
    X_val, y_val = X_train_full[val_idx], y_train_full[val_idx]
    X_train, X_val, X_test = normalize_features(X_train, X_val, X_test)

    print(f"  Feature dim: {X_train.shape[1]}  |  Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

    print("[3/6] Training attention model...")
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
        batch_size=BATCH_SIZE, shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val), torch.tensor(y_val)),
        batch_size=BATCH_SIZE,
    )

    model = AttentionPPINet().to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {param_count:,}")

    model = train_model(model, train_loader, val_loader, device)

    print("[4/6] Evaluating...")
    results, y_prob, y_pred = evaluate(model, X_test, y_test, device)

    print("[5/6] Extracting attention weights...")
    attn_weights = extract_attention_weights(model, X_test, device)
    print(f"  Attention weights shape: {attn_weights.shape}")

    info_df = pd.read_csv(PROTEIN_INFO_PATH, sep="\t")
    info_df = info_df.rename(columns={"#string_protein_id": "protein_id"})

    analysis_md, category_means = analyze_attention(
        attn_weights, y_test.astype(int), y_pred, y_prob, test_pairs, info_df
    )

    print("[6/6] Generating plots and report...")
    plot_attention(category_means, PLOT_OUTPUT)

    two_tower_results = {"f1": 0.805, "roc_auc": 0.896, "pr_auc": 0.902}
    comparison = f"""## Performance Comparison

| Model | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|
| Two-Tower (no attention) | {two_tower_results['f1']:.3f} | {two_tower_results['roc_auc']:.3f} | {two_tower_results['pr_auc']:.3f} |
| **Two-Tower + Attention** | **{results['f1']:.3f}** | **{results['roc_auc']:.3f}** | **{results['pr_auc']:.3f}** |
"""

    report = f"""# Attention Analysis — Two-Tower MLP with Cross-Tower Attention

{comparison}

{analysis_md}

## Architecture

```
net_tower:  {NET_PAIR_DIM} → {TOWER_DIM}  (BN + ReLU + Dropout 0.3)
seq_tower:  {SEQ_PAIR_DIM} → {TOWER_DIM}  (BN + ReLU + Dropout 0.3)
attention:  {NUM_HEADS}-head self-attention over 2 tokens of dim {TOWER_DIM}
            + residual connection + LayerNorm
head:       {TOWER_DIM * 2} → {TOWER_DIM} → 1  (BN + ReLU + Dropout 0.2)
```

Parameters: {param_count:,}
"""

    REPORT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUTPUT.write_text(report)
    print(f"  Report saved to {REPORT_OUTPUT}")

    torch.save(model.state_dict(), MODEL_OUTPUT)
    print(f"  Model saved to {MODEL_OUTPUT}")


if __name__ == "__main__":
    main()

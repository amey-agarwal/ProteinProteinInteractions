from pathlib import Path

import h5py
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
MODEL_OUTPUT = Path("models/improved_mlp.pt")
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
        h = torch.cat([h_net, h_seq], dim=1)
        return self.head(h).squeeze(-1)


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
    features, labels = [], []
    for _, row in df.iterrows():
        p1, p2 = row["protein1"], row["protein2"]
        if p1 not in net_emb or p2 not in net_emb or p1 not in seq_emb or p2 not in seq_emb:
            continue
        n1, n2 = net_emb[p1], net_emb[p2]
        s1, s2 = seq_emb[p1], seq_emb[p2]
        pair = np.concatenate([
            np.abs(n1 - n2), n1 * n2,
            np.abs(s1 - s2), s1 * s2,
        ])
        features.append(pair)
        labels.append(row["label"])
    return np.array(features, dtype=np.float32), np.array(labels, dtype=np.float32)


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
            print(f"Epoch {epoch+1:3d}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  lr={current_lr:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
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

    print("\n=== Improved MLP Results ===")
    for k, v in results.items():
        print(f"{k:>12s}: {v:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, digits=4, zero_division=0))
    return results


def main():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    MODEL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATASET_PATH)
    seq_emb = load_embeddings(SEQUENCE_EMBEDDINGS_PATH)
    net_emb = load_embeddings(NETWORK_EMBEDDINGS_PATH)

    train_df, test_df = protein_wise_split(df)

    if len(train_df) > MAX_TRAIN_ROWS:
        train_df = train_df.sample(n=MAX_TRAIN_ROWS, random_state=RANDOM_SEED).reset_index(drop=True)
    if len(test_df) > MAX_TEST_ROWS:
        test_df = test_df.sample(n=MAX_TEST_ROWS, random_state=RANDOM_SEED).reset_index(drop=True)

    print(f"Train: {len(train_df):,}  |  Test: {len(test_df):,}")

    X_train_full, y_train_full = build_features(train_df, seq_emb, net_emb)
    X_test, y_test = build_features(test_df, seq_emb, net_emb)

    n_val = int(len(X_train_full) * VAL_FRACTION)
    indices = np.random.permutation(len(X_train_full))
    val_idx, train_idx = indices[:n_val], indices[n_val:]

    X_train, y_train = X_train_full[train_idx], y_train_full[train_idx]
    X_val, y_val = X_train_full[val_idx], y_train_full[val_idx]

    X_train, X_val, X_test = normalize_features(X_train, X_val, X_test)

    print(f"Feature dim: {X_train.shape[1]}  |  Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
        batch_size=BATCH_SIZE, shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val), torch.tensor(y_val)),
        batch_size=BATCH_SIZE,
    )

    model = PPINet().to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    model = train_model(model, train_loader, val_loader, device)
    evaluate(model, X_test, y_test, device)

    torch.save(model.state_dict(), MODEL_OUTPUT)
    print(f"\nSaved model to {MODEL_OUTPUT}")


if __name__ == "__main__":
    main()

from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier

DATASET_PATH = Path("data/processed/ppi_dataset_hard.csv")
SEQUENCE_EMBEDDINGS_PATH = Path("data/embeddings/9606.protein.sequence.embeddings.v12.0.h5")
NETWORK_EMBEDDINGS_PATH = Path("data/embeddings/9606.protein.network.embeddings.v12.0.h5")
RESULTS_OUTPUT = Path("data/processed/model_comparison_results.csv")
PLOTS_DIR = Path("plots")
RANDOM_SEED = 42
TEST_PROTEIN_FRACTION = 0.2
MAX_TRAIN_ROWS = 200_000
MAX_TEST_ROWS = 30_000
NET_PAIR_DIM = 1024
SEQ_PAIR_DIM = 2048
BATCH_SIZE = 512
MAX_EPOCHS = 100
TT_LR = 5e-4
TT_WEIGHT_DECAY = 0.01
TT_PATIENCE = 15


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


def protein_wise_split(df):
    rng = np.random.default_rng(RANDOM_SEED)
    proteins = np.array(pd.unique(df[["protein1", "protein2"]].values.ravel("K")))
    rng.shuffle(proteins)
    n_test = max(1, int(len(proteins) * TEST_PROTEIN_FRACTION))
    test_proteins = set(proteins[:n_test])
    is_p1_test = df["protein1"].isin(test_proteins)
    is_p2_test = df["protein2"].isin(test_proteins)
    return df[(~is_p1_test) & (~is_p2_test)].copy(), df[is_p1_test & is_p2_test].copy()


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
    y = valid["label"].values.astype(np.int64)
    return X, y


def get_sklearn_models():
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)),
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100, random_state=RANDOM_SEED, eval_metric="logloss", verbosity=0,
        ),
        "MLP (sklearn)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(
                hidden_layer_sizes=(128, 64), max_iter=30, random_state=RANDOM_SEED,
                early_stopping=True, validation_fraction=0.1, n_iter_no_change=5,
            )),
        ]),
    }


def train_two_tower(X_train, y_train, device):
    val_n = int(len(X_train) * 0.1)
    idx = np.random.RandomState(RANDOM_SEED).permutation(len(X_train))
    val_idx, tr_idx = idx[:val_n], idx[val_n:]

    mean = X_train[tr_idx].mean(axis=0)
    std = X_train[tr_idx].std(axis=0) + 1e-8
    X_tr = (X_train[tr_idx] - mean) / std
    X_vl = (X_train[val_idx] - mean) / std
    y_tr = y_train[tr_idx].astype(np.float32)
    y_vl = y_train[val_idx].astype(np.float32)

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)),
        batch_size=BATCH_SIZE, shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_vl), torch.tensor(y_vl)),
        batch_size=BATCH_SIZE,
    )

    model = PPINet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=TT_LR, weight_decay=TT_WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        t_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            t_loss += loss.item() * len(xb)
        t_loss /= len(train_loader.dataset)

        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                v_loss += criterion(model(xb), yb).item() * len(xb)
        v_loss /= len(val_loader.dataset)
        scheduler.step(v_loss)

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"  TT Epoch {epoch+1:3d}  train={t_loss:.4f}  val={v_loss:.4f}")

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= TT_PATIENCE:
                print(f"  TT early stop at epoch {epoch+1}")
                break

    model.load_state_dict(best_state)
    return model, mean, std


def predict_two_tower(model, X_test, mean, std, device):
    model.eval()
    X_norm = (X_test - mean) / std
    with torch.no_grad():
        logits = model(torch.tensor(X_norm, dtype=torch.float32).to(device))
        y_prob = torch.sigmoid(logits).cpu().numpy()
    return (y_prob >= 0.5).astype(int), y_prob


def eval_metrics(y_test, y_pred, y_prob, name):
    return {
        "model": name,
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "pr_auc": average_precision_score(y_test, y_prob),
    }


def plot_roc_curves(outputs, y_test):
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, y_prob in outputs:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Models")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "roc_curves.png", dpi=150)
    plt.close(fig)
    print("Saved plots/roc_curves.png")


def plot_pr_curves(outputs, y_test):
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, y_prob in outputs:
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)
        ax.plot(rec, prec, label=f"{name} (PR-AUC={pr_auc:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves — All Models")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "pr_curves.png", dpi=150)
    plt.close(fig)
    print("Saved plots/pr_curves.png")


def plot_confusion_matrices(outputs, y_test):
    n = len(outputs)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, (name, y_pred) in zip(axes, outputs):
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=["Neg", "Pos"])
        disp.plot(ax=ax, colorbar=False)
        ax.set_title(name, fontsize=10)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "confusion_matrices.png", dpi=150)
    plt.close(fig)
    print("Saved plots/confusion_matrices.png")


def plot_metric_bars(results_df):
    metrics = ["f1", "roc_auc", "pr_auc"]
    labels = ["F1-Score", "ROC-AUC", "PR-AUC"]
    models = results_df["model"].tolist()

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    x = np.arange(len(models))

    for ax, metric, label in zip(axes, metrics, labels):
        values = results_df[metric].values
        bars = ax.bar(x, values, color=["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"])
        ax.set_ylabel(label)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=30, ha="right", fontsize=8)
        ax.set_ylim(min(values) - 0.05, max(values) + 0.03)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", fontsize=8)

    fig.suptitle("Model Comparison — Key Metrics", fontsize=13)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "metric_comparison.png", dpi=150)
    plt.close(fig)
    print("Saved plots/metric_comparison.png")


def main():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATASET_PATH)
    seq_emb = load_embeddings(SEQUENCE_EMBEDDINGS_PATH)
    net_emb = load_embeddings(NETWORK_EMBEDDINGS_PATH)

    train_df, test_df = protein_wise_split(df)

    if len(train_df) > MAX_TRAIN_ROWS:
        train_df = train_df.sample(n=MAX_TRAIN_ROWS, random_state=RANDOM_SEED).reset_index(drop=True)
    if len(test_df) > MAX_TEST_ROWS:
        test_df = test_df.sample(n=MAX_TEST_ROWS, random_state=RANDOM_SEED).reset_index(drop=True)

    print(f"Train: {len(train_df):,}  |  Test: {len(test_df):,}")

    X_train, y_train = build_features(train_df, seq_emb, net_emb)
    X_test, y_test = build_features(test_df, seq_emb, net_emb)
    print(f"Feature shape: {X_train.shape}")

    all_results = []
    roc_out = []
    cm_out = []

    for name, model in get_sklearn_models().items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        r = eval_metrics(y_test, y_pred, y_prob, name)
        all_results.append(r)
        roc_out.append((name, y_prob))
        cm_out.append((name, y_pred))
        print(f"  F1={r['f1']:.4f}  ROC-AUC={r['roc_auc']:.4f}  PR-AUC={r['pr_auc']:.4f}")

    print("\nTraining Two-Tower MLP...")
    tt_model, mean, std = train_two_tower(X_train, y_train, device)
    tt_pred, tt_prob = predict_two_tower(tt_model, X_test, mean, std, device)
    r = eval_metrics(y_test, tt_pred, tt_prob, "Two-Tower MLP")
    all_results.append(r)
    roc_out.append(("Two-Tower MLP", tt_prob))
    cm_out.append(("Two-Tower MLP", tt_pred))
    print(f"  F1={r['f1']:.4f}  ROC-AUC={r['roc_auc']:.4f}  PR-AUC={r['pr_auc']:.4f}")

    results_df = pd.DataFrame(all_results)
    print("\n" + "=" * 70)
    print(results_df.to_string(index=False))
    results_df.to_csv(RESULTS_OUTPUT, index=False)
    print(f"\nSaved results to {RESULTS_OUTPUT}")

    plot_roc_curves(roc_out, y_test)
    plot_pr_curves(roc_out, y_test)
    plot_confusion_matrices(cm_out, y_test)
    plot_metric_bars(results_df)


if __name__ == "__main__":
    main()

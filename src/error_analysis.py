"""Error analysis of the network-only MLP on the full dataset."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

from run_ablation_experiments import (
    DATASET_PATH,
    NETWORK_EMBEDDINGS_PATH,
    RANDOM_SEED,
    SEQUENCE_EMBEDDINGS_PATH,
    TEST_PROTEIN_FRACTION,
    build_embedding_matrix,
    build_pair_features,
    load_embeddings,
    protein_wise_split,
    sample_dataframe,
    train_model,
)

PROTEIN_INFO_PATH = Path("data/raw/9606.protein.info.v12.0.txt")
REPORT_OUTPUT = Path("data/processed/error_analysis.md")
CSV_OUTPUT = Path("data/processed/error_details.csv")

MAX_TRAIN_ROWS = None
MAX_TEST_ROWS = None


def load_protein_info() -> pd.DataFrame:
    info = pd.read_csv(PROTEIN_INFO_PATH, sep="\t")
    info = info.rename(columns={"#string_protein_id": "protein_id"})
    return info


def cosine_similarity_pairs(net_emb: dict, p1_ids: list, p2_ids: list) -> np.ndarray:
    sims = []
    for p1, p2 in zip(p1_ids, p2_ids):
        key1 = p1.replace("9606.", "") if p1.startswith("9606.") else p1
        key2 = p2.replace("9606.", "") if p2.startswith("9606.") else p2
        e1 = net_emb.get(p1) if p1 in net_emb else net_emb.get(key1)
        e2 = net_emb.get(p2) if p2 in net_emb else net_emb.get(key2)
        if e1 is not None and e2 is not None:
            norm1 = np.linalg.norm(e1)
            norm2 = np.linalg.norm(e2)
            if norm1 > 0 and norm2 > 0:
                sims.append(float(np.dot(e1, e2) / (norm1 * norm2)))
            else:
                sims.append(np.nan)
        else:
            sims.append(np.nan)
    return np.array(sims)


def analyze_confidence_bins(y_true: np.ndarray, y_prob: np.ndarray) -> str:
    bins = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5),
            (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.001)]
    y_pred = (y_prob >= 0.5).astype(int)
    correct = y_pred == y_true

    lines = ["| Confidence Bin | Total | Errors | Error Rate |",
             "|---|---|---|---|"]
    for lo, hi in bins:
        mask = (y_prob >= lo) & (y_prob < hi)
        total = mask.sum()
        if total == 0:
            continue
        errors = (~correct[mask]).sum()
        rate = errors / total
        label = f"{lo:.1f}–{min(hi, 1.0):.1f}"
        lines.append(f"| {label} | {total:,} | {errors:,} | {rate:.3f} |")
    return "\n".join(lines)


def analyze_frequent_errors(test_df: pd.DataFrame, id_to_name: dict) -> str:
    errors = test_df[~test_df["correct"]]
    protein_counts: dict[str, int] = {}
    for _, row in errors.iterrows():
        for pid in [row["protein1"], row["protein2"]]:
            protein_counts[pid] = protein_counts.get(pid, 0) + 1

    sorted_proteins = sorted(protein_counts.items(), key=lambda x: x[1], reverse=True)[:15]

    lines = ["| Protein | Name | Error Count |",
             "|---|---|---|"]
    for pid, count in sorted_proteins:
        name = id_to_name.get(pid, pid)
        short_id = pid.split(".")[-1] if "." in pid else pid
        lines.append(f"| {short_id} | {name} | {count} |")
    return "\n".join(lines)


def analyze_protein_size(test_df: pd.DataFrame, id_to_size: dict) -> str:
    sizes = []
    for _, row in test_df.iterrows():
        s1 = id_to_size.get(row["protein1"], np.nan)
        s2 = id_to_size.get(row["protein2"], np.nan)
        sizes.append(np.nanmean([s1, s2]))
    test_df = test_df.copy()
    test_df["avg_size"] = sizes

    bins = [0, 200, 400, 600, 800, 1000, float("inf")]
    labels = ["<200", "200-400", "400-600", "600-800", "800-1000", ">1000"]

    lines = ["| Size Bin (aa) | Total | Errors | Error Rate |",
             "|---|---|---|---|"]
    for i in range(len(labels)):
        mask = (test_df["avg_size"] >= bins[i]) & (test_df["avg_size"] < bins[i + 1])
        total = mask.sum()
        if total == 0:
            continue
        errors = (~test_df.loc[mask, "correct"]).sum()
        rate = errors / total
        lines.append(f"| {labels[i]} | {total:,} | {errors:,} | {rate:.3f} |")
    return "\n".join(lines)


def analyze_cosine_similarity(test_df: pd.DataFrame) -> str:
    correct = test_df[test_df["correct"]]
    incorrect = test_df[~test_df["correct"]]

    groups = {
        "Correct Positives": correct[correct["y_true"] == 1],
        "Correct Negatives": correct[correct["y_true"] == 0],
        "False Positives": incorrect[incorrect["y_true"] == 0],
        "False Negatives": incorrect[incorrect["y_true"] == 1],
    }

    lines = ["| Category | Count | Mean Cosine Sim | Std |",
             "|---|---|---|---|"]
    for label, subset in groups.items():
        sims = subset["cosine_sim"].dropna()
        if len(sims) == 0:
            continue
        lines.append(f"| {label} | {len(sims):,} | {sims.mean():.4f} | {sims.std():.4f} |")
    return "\n".join(lines)


def analyze_keyword_enrichment(test_df: pd.DataFrame, id_to_anno: dict) -> str:
    keywords = ["ribosom", "kinase", "receptor", "transcription", "histone",
                "ubiquitin", "proteasome", "chaperone", "channel", "transport",
                "mitochond", "nuclear", "membrane", "signaling", "enzyme",
                "binding", "cytoplasm", "apoptosis", "DNA", "RNA"]

    errors = test_df[~test_df["correct"]]
    correct = test_df[test_df["correct"]]

    def keyword_rate(subset, kw):
        count = 0
        for _, row in subset.iterrows():
            a1 = str(id_to_anno.get(row["protein1"], "")).lower()
            a2 = str(id_to_anno.get(row["protein2"], "")).lower()
            if kw.lower() in a1 or kw.lower() in a2:
                count += 1
        return count / len(subset) if len(subset) > 0 else 0

    lines = ["| Keyword | Rate in Errors | Rate in Correct | Enrichment |",
             "|---|---|---|---|"]
    results = []
    for kw in keywords:
        err_rate = keyword_rate(errors, kw)
        cor_rate = keyword_rate(correct, kw)
        enrichment = (err_rate / cor_rate) if cor_rate > 0 else float("inf")
        results.append((kw, err_rate, cor_rate, enrichment))

    results.sort(key=lambda x: x[3], reverse=True)
    for kw, err_rate, cor_rate, enrichment in results:
        lines.append(f"| {kw} | {err_rate:.4f} | {cor_rate:.4f} | {enrichment:.2f}x |")
    return "\n".join(lines)


def main():
    print("=" * 65)
    print("Error Analysis — Network-Only MLP")
    print("=" * 65)

    print("\n[1/6] Loading data and embeddings...")
    df = pd.read_csv(DATASET_PATH)
    net_emb = load_embeddings(NETWORK_EMBEDDINGS_PATH)
    seq_emb = load_embeddings(SEQUENCE_EMBEDDINGS_PATH)
    seq_lookup = build_embedding_matrix(seq_emb)
    net_lookup = build_embedding_matrix(net_emb)
    info = load_protein_info()

    id_to_name = dict(zip(info["protein_id"], info["preferred_name"]))
    id_to_anno = dict(zip(info["protein_id"], info["annotation"]))
    id_to_size = dict(zip(info["protein_id"], info["protein_size"]))

    print("[2/6] Splitting data and building features...")
    train_df, test_df = protein_wise_split(df, TEST_PROTEIN_FRACTION, RANDOM_SEED)
    train_df = sample_dataframe(train_df, MAX_TRAIN_ROWS, RANDOM_SEED)
    test_df = sample_dataframe(test_df, MAX_TEST_ROWS, RANDOM_SEED)

    X_train, y_train = build_pair_features(train_df, seq_lookup, net_lookup, "network")
    X_test, y_test = build_pair_features(test_df, seq_lookup, net_lookup, "network")

    print(f"   Train: {len(y_train):,} rows | Test: {len(y_test):,} rows")

    print("[3/6] Training model...")
    model = train_model(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    test_df = test_df.reset_index(drop=True).iloc[:len(y_test)].copy()
    test_df["y_true"] = y_test
    test_df["y_prob"] = y_prob
    test_df["y_pred"] = y_pred
    test_df["correct"] = y_pred == y_test
    test_df["error_type"] = "correct"
    test_df.loc[(test_df["y_true"] == 1) & (test_df["y_pred"] == 0), "error_type"] = "false_negative"
    test_df.loc[(test_df["y_true"] == 0) & (test_df["y_pred"] == 1), "error_type"] = "false_positive"

    print("[4/6] Computing embedding cosine similarities...")
    test_df["cosine_sim"] = cosine_similarity_pairs(
        net_emb, test_df["protein1"].tolist(), test_df["protein2"].tolist()
    )

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    total_errors = fp + fn
    total = len(y_test)
    accuracy = (tp + tn) / total
    report = classification_report(y_test, y_pred, target_names=["non-interact", "interact"])

    print("[5/6] Running analyses...")

    confidence_table = analyze_confidence_bins(y_test, y_prob)
    frequent_errors_table = analyze_frequent_errors(test_df, id_to_name)
    size_table = analyze_protein_size(test_df, id_to_size)
    cosine_table = analyze_cosine_similarity(test_df)
    keyword_table = analyze_keyword_enrichment(test_df, id_to_anno)

    fp_df = test_df[test_df["error_type"] == "false_positive"].sort_values("y_prob", ascending=False).head(10)
    fp_lines = ["| Protein A | Protein B | Pred Prob | Cosine Sim |",
                "|---|---|---|---|"]
    for _, row in fp_df.iterrows():
        a = id_to_name.get(row["protein1"], row["protein1"])
        b = id_to_name.get(row["protein2"], row["protein2"])
        fp_lines.append(f"| {a} | {b} | {row['y_prob']:.3f} | {row['cosine_sim']:.4f} |")
    fp_table = "\n".join(fp_lines)

    fn_df = test_df[test_df["error_type"] == "false_negative"].sort_values("y_prob", ascending=True).head(10)
    fn_lines = ["| Protein A | Protein B | Pred Prob | Cosine Sim |",
                "|---|---|---|---|"]
    for _, row in fn_df.iterrows():
        a = id_to_name.get(row["protein1"], row["protein1"])
        b = id_to_name.get(row["protein2"], row["protein2"])
        fn_lines.append(f"| {a} | {b} | {row['y_prob']:.3f} | {row['cosine_sim']:.4f} |")
    fn_table = "\n".join(fn_lines)

    print("[6/6] Writing report...")

    report_md = f"""# Error Analysis — Network-Only MLP

## 1. Overall Performance

- **Total test pairs:** {total:,}
- **Accuracy:** {accuracy:.4f}
- **Total errors:** {total_errors:,} ({total_errors/total*100:.1f}%)
- **False Positives:** {fp:,} (non-interacting predicted as interacting)
- **False Negatives:** {fn:,} (interacting predicted as non-interacting)

```
{report}
```

**Confusion Matrix:**

|  | Pred Non-Interact | Pred Interact |
|---|---|---|
| **Actual Non-Interact** | {tn:,} | {fp:,} |
| **Actual Interact** | {fn:,} | {tp:,} |

## 2. Error Rate by Confidence Bin

How often the model is wrong at different confidence levels. Errors near 0.5 are
expected (uncertain predictions). Errors near 0.0 or 1.0 are confident mistakes.

{confidence_table}

## 3. Cosine Similarity — Correct vs Incorrect Predictions

Embedding cosine similarity between protein pairs, split by prediction outcome.
If false positives have high cosine similarity, the model is fooled by proteins
that are close in embedding space but don't actually interact.

{cosine_table}

## 4. Top 10 Most Confident False Positives

Non-interacting pairs the model is most sure are interacting:

{fp_table}

## 5. Top 10 Most Confident False Negatives

Interacting pairs the model is most sure are NOT interacting:

{fn_table}

## 6. Most Frequently Misclassified Proteins

Proteins that appear most often across all misclassified pairs:

{frequent_errors_table}

## 7. Error Rate by Protein Size

Average amino acid length of the pair vs error rate:

{size_table}

## 8. Functional Keyword Enrichment in Errors

Which functional categories are over/under-represented in errors compared to
correct predictions. Enrichment > 1.0 means the keyword appears more often in
errors; < 1.0 means it appears less.

{keyword_table}
"""

    REPORT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUTPUT.write_text(report_md)

    out_cols = ["protein1", "protein2", "y_true", "y_prob", "y_pred", "correct",
                "error_type", "cosine_sim"]
    test_df[out_cols].to_csv(CSV_OUTPUT, index=False)

    print(f"\nReport saved to: {REPORT_OUTPUT}")
    print(f"Details saved to: {CSV_OUTPUT}")
    print(f"\nSummary: {total_errors:,} errors out of {total:,} test pairs ({total_errors/total*100:.1f}%)")
    print(f"  False Positives: {fp:,} | False Negatives: {fn:,}")


if __name__ == "__main__":
    main()

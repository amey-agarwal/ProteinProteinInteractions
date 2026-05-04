"""Pick illustrative protein pairs from the test set and build a case-study table.

Trains the network-only model (the project's best mode) on the full train split,
predicts on the test split, and selects:
- Two confident correct positives  (label=1, high prob)
- One confident correct negative   (label=0, low prob)
- One confident wrong prediction   (high-confidence miss, either direction)
"""
from pathlib import Path

import numpy as np
import pandas as pd

from run_ablation_experiments import (
    DATASET_PATH,
    MAX_TEST_ROWS,
    MAX_TRAIN_ROWS,
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
OUTPUT_PATH = Path("data/processed/case_study.md")


def load_protein_info() -> pd.DataFrame:
    info = pd.read_csv(PROTEIN_INFO_PATH, sep="\t")
    info = info.rename(columns={"#string_protein_id": "protein_id"})
    return info


def short_annotation(text: str, max_len: int = 140) -> str:
    if not isinstance(text, str):
        return ""
    text = text.split(";")[0].strip()
    if len(text) > max_len:
        text = text[: max_len - 1].rsplit(" ", 1)[0] + "…"
    return text


def format_table(rows: list[dict]) -> str:
    headers = ["Protein A", "Protein B", "True label", "Pred prob", "Correct?", "Notes"]
    md = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        md.append(
            "| {a} | {b} | {label} | {prob:.3f} | {ok} | {note} |".format(
                a=r["a"],
                b=r["b"],
                label="interact" if r["label"] == 1 else "no",
                prob=r["prob"],
                ok="✓" if r["correct"] else "✗",
                note=r["note"],
            )
        )
    return "\n".join(md)


def pick_pairs(test_df: pd.DataFrame, y_true: np.ndarray, y_prob: np.ndarray, info: pd.DataFrame) -> list[dict]:
    test_df = test_df.reset_index(drop=True)
    test_df = test_df.iloc[: len(y_true)].copy()
    test_df["y_true"] = y_true
    test_df["y_prob"] = y_prob
    test_df["pred"] = (y_prob >= 0.5).astype(int)
    test_df["correct"] = test_df["pred"] == test_df["y_true"]

    id_to_name = dict(zip(info["protein_id"], info["preferred_name"]))
    id_to_anno = dict(zip(info["protein_id"], info["annotation"]))

    selections: list[tuple[str, pd.Series]] = []

    correct_pos = test_df[(test_df["y_true"] == 1) & test_df["correct"]].sort_values("y_prob", ascending=False)
    if len(correct_pos):
        selections.append(("top correct positive", correct_pos.iloc[0]))
    if len(correct_pos) > 1:
        selections.append(("strong correct positive", correct_pos.iloc[1]))

    correct_neg = test_df[(test_df["y_true"] == 0) & test_df["correct"]].sort_values("y_prob", ascending=True)
    if len(correct_neg):
        selections.append(("top correct negative", correct_neg.iloc[0]))

    wrong = test_df[~test_df["correct"]].copy()
    wrong["margin"] = (wrong["y_prob"] - 0.5).abs()
    wrong = wrong.sort_values("margin", ascending=False)
    if len(wrong):
        selections.append(("most confident miss", wrong.iloc[0]))

    rows: list[dict] = []
    for note, row in selections:
        a_id = row["protein1"]
        b_id = row["protein2"]
        a_name = id_to_name.get(a_id, a_id)
        b_name = id_to_name.get(b_id, b_id)
        a_anno = short_annotation(id_to_anno.get(a_id, ""))
        rows.append(
            {
                "a": f"{a_name} ({a_id.split('.')[-1]})",
                "b": f"{b_name} ({b_id.split('.')[-1]})",
                "label": int(row["y_true"]),
                "prob": float(row["y_prob"]),
                "correct": bool(row["correct"]),
                "note": f"{note} — A: {a_anno}",
            }
        )
    return rows


def main():
    print("Loading data...")
    df = pd.read_csv(DATASET_PATH)

    seq_emb = load_embeddings(SEQUENCE_EMBEDDINGS_PATH)
    net_emb = load_embeddings(NETWORK_EMBEDDINGS_PATH)
    seq_lookup = build_embedding_matrix(seq_emb)
    net_lookup = build_embedding_matrix(net_emb)

    train_df, test_df = protein_wise_split(df, TEST_PROTEIN_FRACTION, RANDOM_SEED)
    train_df = sample_dataframe(train_df, MAX_TRAIN_ROWS, RANDOM_SEED)
    test_df = sample_dataframe(test_df, MAX_TEST_ROWS, RANDOM_SEED)

    print("Building features (network mode)...")
    X_train, y_train = build_pair_features(train_df, seq_lookup, net_lookup, "network")
    X_test, y_test = build_pair_features(test_df, seq_lookup, net_lookup, "network")

    print("Training network-only model...")
    model = train_model(X_train, y_train)

    print("Predicting on test set...")
    y_prob = model.predict_proba(X_test)[:, 1]

    print("Loading protein info and selecting pairs...")
    info = load_protein_info()
    rows = pick_pairs(test_df, y_test, y_prob, info)

    table_md = format_table(rows)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(table_md + "\n")

    print("\n" + table_md)
    print(f"\nSaved case-study table to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

# Protein-Protein Interaction Prediction — Project Plan

## Project Overview

Build an AI model that predicts whether two proteins interact inside the human body, using data from the STRING database (v12.0, Homo sapiens). The project uses pretrained protein embeddings (sequence and network) as features and compares multiple ML models.

**Team:** Rishi Reddy Bandi, Nithil Balaji, Amey Agarwal, Sai Srinivas Velpuri, Vishal Maradana

## Data

- **Source:** STRING database, human proteins only (taxon 9606)
- **Main dataset:** `data/processed/ppi_dataset_hard.csv` (~324K rows)
- **Labeling:** Positive = combined_score >= 800, Negative = combined_score 300–500
- **Embeddings:**
  - Sequence embeddings (1024-dim) — from protein amino acid sequences
  - Network embeddings (512-dim) — from protein interaction graph
- **Split:** Protein-wise 80/20 (prevents leakage from shared proteins across train/test)

## Problems Solved

1. **Data leakage** — Early models used evidence-based features that directly compute the combined_score label. Removed these; focused on independent embeddings.
2. **Easy negatives** — Random/edge-swapped negatives gave unrealistically high metrics. Switched to medium-confidence STRING pairs (300–500) as harder negatives.
3. **Memory limits** — Full sequence+network feature matrix (~209K x 3072) exceeds laptop RAM. Capped sequence_network mode at 150K train rows.

## Completed Work

### 1. Ablation Study (Rishi)

Compares which embedding type contributes most. Run on full dataset (~209K train rows).

| Feature Set | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|
| Sequence only | 0.634 | 0.739 | 0.751 |
| **Network only** | **0.792** | **0.886** | **0.895** |
| Sequence + Network (150K cap) | 0.784 | 0.870 | 0.883 |

**Finding:** Network embeddings alone outperform sequence and even the combined set. Sequence embeddings encode structural/evolutionary similarity but proteins can be structurally unrelated yet interact.

Previous 50K results for reference:

| Feature Set | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|
| Sequence only | 0.626 | 0.724 | 0.723 |
| Network only | 0.773 | 0.864 | 0.872 |
| Sequence + Network | 0.770 | 0.865 | 0.871 |

**Files:** `src/run_ablation_experiments.py`, `src/run_seqnet_capped.py`

### 2. Model Comparison (50K train)

| Model | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|
| Logistic Regression | 0.766 | 0.856 | 0.857 |
| Random Forest | 0.716 | 0.832 | 0.838 |
| XGBoost | 0.752 | 0.847 | 0.854 |
| MLP (sklearn) | 0.767 | 0.858 | 0.865 |

**Files:** `src/improved/compare_models.py`

### 3. Improved MLP — Three Rounds of Iteration

**Round 1 — PyTorch MLP (50K train, network only)**
- Switched from sklearn to PyTorch for dropout, batch norm, proper training
- Architecture: 512 → 256 → 128 → 1
- Added average embedding as feature (total 1536 dims)
- Result: F1=0.797, ROC-AUC=0.888
- Issue: Overfitting (953K params on 45K samples)

**Round 2 — More data + regularization (200K train, sequence + network)**
- 200K train / 30K test, weight decay 1e-3, dropout 0.4/0.3/0.2
- Features: diff + product for both embeddings = 3072 dims
- Result: F1=0.796, ROC-AUC=0.887 — no gain despite 4x more data
- Issue: Flat MLP treats all 3072 features equally

**Round 3 — Two-Tower Fusion (200K train, sequence + network)**
- Separate towers for network (1024→128) and sequence (2048→128) before fusion
- Fusion head: 256 → 128 → 1
- ~430K params (down from 1.7M), weight decay 0.01, gradient clipping
- Result: **F1=0.805, ROC-AUC=0.896, PR-AUC=0.902**
- Trained 61 epochs, train-val gap well controlled

**Files:** `src/improved/train_improved_mlp.py`

### 4. Explainability & XAI Report (Amey)

Comprehensive explainability analysis generating an HTML report with 9 sections:

1. **Model ablation** — confirms network > sequence
2. **Embedding geometry** — network embeddings have 0.2236 cosine separation between interacting/non-interacting pairs vs 0.0564 for sequence (explains why network wins)
3. **Feature-component ablation** — element-wise product (F1=0.732) > abs-diff (F1=0.684); both together best (F1=0.768)
4. **Prototype pairs** — most confident predictions
5. **SHAP feature importance** — KernelSHAP on 300 test pairs, identifies top embedding dimensions
6. **PCA variance** — dimensionality analysis of embedding spaces
7. **Biology Q&A** — interprets what model learned (pathway membership, complex membership, hub vs peripheral proteins)
8. **Neighbourhood overlap** — k=10 nearest neighbour overlap between interacting pairs
9. **Limitations**

**Files:** `src/xai_ppi_analysis.py`, `data/processed/xai_report.html`

### 5. SHAP on XGBoost

Separate SHAP analysis focused specifically on XGBoost model.

**Files:** `src/improved/explain_model.py`

### 6. Case Studies

Two implementations identifying illustrative predictions:
- Correct positives: ribosomal proteins RPS3A–RPS24 and RPL8–RPL4 (prob=1.000)
- Correct negative: PRAMEF1–KCNV1 (prob=0.000)
- Most confident miss: **H2BS1–H2BC15** (histone H2B variants, labeled non-interacting but predicted prob=1.000) — worth a presentation slide

**Files:** `src/case_study.py`, `src/improved/case_study.py`

### 7. Visualizations

- ROC and PR curves for all ablation modes
- Confusion matrices for all models
- Metric comparison bar charts
- SHAP summary plots

**Files:** `src/plot_curves.py`, output in `figures/` and `plots/`

### 8. Error Analysis (DONE)

Systematic analysis of 2,376 errors across 12,672 test pairs (18.8% error rate) on the
network-only model trained on full data (~209K rows).

Key findings:
- **2:1 false negative to false positive ratio** (1,576 FN vs 800 FP) — model more likely to miss real interactions than hallucinate them
- **Histone false positives are likely label errors** — H2BS1–H2BC15, H3C6–H3Y2, etc. predicted with prob=1.000 and cosine sim 0.7–0.95. These variants physically assemble in nucleosome complexes; STRING's medium combined_score likely underestimates them
- **Hub proteins (EGFR, CDH1, AKT1) have the most errors** — expected since their embeddings average across many diverse interaction partners
- **Cross-module interactions are the main failure mode** — false negatives like SECISBP2 + ribosomal proteins have low cosine sim (~0.19); network embeddings can't bridge different graph neighborhoods
- **Ribosomal/proteasome proteins have lowest error rates** (0.68x/0.64x enrichment) — dense, well-characterized complexes are easy
- **Kinases/receptors/enzymes have highest error rates** (1.24x/1.14x/1.25x) — diverse interaction patterns are harder
- **Confidence calibration is reasonable** — 4.3% error rate at >0.9 confidence, 49% near the 0.5 boundary

**Files:** `src/error_analysis.py`, `data/processed/error_analysis.md`, `data/processed/error_details.csv`

## How to Run

```bash
sbatch job.sh
```

The pipeline runs 10 steps end-to-end (see `job.sh`). Latest output: `pipeline_output.txt`.

## File Index

| File | Purpose |
|---|---|
| `src/preprocess_data_3.py` | Build hard-labeled dataset from STRING |
| `src/run_ablation_experiments.py` | Ablation: sequence vs network vs combined (full data) |
| `src/run_seqnet_capped.py` | Memory-safe sequence+network run (150K cap) |
| `src/plot_curves.py` | ROC/PR curve plots from saved .npz files |
| `src/case_study.py` | Case study with protein annotations |
| `src/error_analysis.py` | Systematic error analysis on full-data network model |
| `src/compare_models.py` | Benchmark LogReg, RF, XGBoost, MLP, Two-Tower |
| `src/train_two_tower_mlp.py` | Two-Tower PyTorch MLP (best model) |
| `src/train_attention_mlp.py` | Attention MLP with cross-tower attention |
| `src/explain_model.py` | SHAP on XGBoost |
| `src/xai_ppi_analysis.py` | Full XAI analysis → HTML report |

# Protein-Protein Interaction Prediction

Predicting protein-protein interactions using network and sequence embeddings from STRING.

## Setup

```bash
pip install -r requirements.txt
```

The following data files are expected before running:

```
data/raw/9606.protein.info.v12.0.txt
data/embeddings/9606.protein.sequence.embeddings.v12.0.h5
data/embeddings/9606.protein.network.embeddings.v12.0.h5
9606.protein.links.detailed.v12.0.txt.gz          # only needed for preprocessing
```

## Running

### Option 1: Slurm (runs everything)

```bash
sbatch job.sh
```

### Option 2: Manual (run from project root)

```bash
export PYTHONPATH="src:$PYTHONPATH"
```

| Step | Command | Output |
|------|---------|--------|
| 0. Preprocessing | `python src/preprocess_data_3.py` | `data/processed/ppi_dataset_hard.csv` |
| 1. Ablation study | `python src/run_ablation_experiments.py` | `data/processed/curves/sequence.npz`, `network.npz` |
| 2. Seq+Net capped run | `python src/run_seqnet_capped.py` | `data/processed/ablation_results.csv`, `data/processed/curves/sequence_network.npz` |
| 3. Plot curves | `python src/plot_curves.py` | `plots/roc_curves.png`, `plots/pr_curves.png`, `plots/roc_pr_combined.png` |
| 4. Case study | `python src/case_study.py` | `data/processed/case_study.md` |
| 5. Error analysis | `python src/error_analysis.py` | `data/processed/error_analysis.md`, `data/processed/error_details.csv` |
| 6. Model comparison | `python src/compare_models.py` | `data/processed/model_comparison_results.csv`, `plots/confusion_matrices.png`, `plots/metric_comparison.png` |
| 7. Two-Tower MLP | `python src/train_two_tower_mlp.py` | `models/improved_mlp.pt` |
| 8. Attention MLP | `python src/train_attention_mlp.py` | `models/attention_mlp.pt`, `data/processed/attention_analysis.md`, `plots/attention_weights.png` |
| 9. SHAP explainability | `python src/explain_model.py` | `plots/shap_summary.png`, `plots/shap_top_features.csv` |
| 10. XAI report | `python src/xai_ppi_analysis.py` | `data/processed/xai_report.html` |

Steps 1-5 must run in order. Steps 6-10 are independent.

See `PPI_Pipeline-5248585.out` for the final pipeline results.

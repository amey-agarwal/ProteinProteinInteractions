#!/bin/bash
#SBATCH -J PPI_Full_Pipeline            # Job name
#SBATCH -N 1 --ntasks-per-node=1        # 1 node, 1 task
#SBATCH --cpus-per-task=8               # 8 CPU cores for data loading
#SBATCH --mem=64G                       # 64GB total memory
#SBATCH --gres=gpu:v100:1               # Request 1 NVIDIA V100 GPU (18 nodes, shortest queue)
#SBATCH -t 180                          # 3 hours wall time
#SBATCH -o PPI_Pipeline-%j.out          # Output file
#SBATCH -e PPI_Pipeline-%j.err          # Error file

# ══════════════════════════════════════════════════════════════
# PPI Full Pipeline — runs ALL training, analysis, and plotting
# ══════════════════════════════════════════════════════════════

set -e  # Exit on first error
PYTHON="${PYTHON:-python}"

# ── Load modules ──────────────────────────────────────────────
module purge
module load pytorch/2.1.0
module load cuda/12.1.1

# Install any missing packages into user space (silent if already installed)
"$PYTHON" -m pip install --user scikit-learn h5py matplotlib "xgboost<3" shap 2>/dev/null || true

# ── Navigate to project (ALL scripts run from here) ───────────
PROJECT_DIR="$HOME/Desktop/ProteinProteinInteractions"
cd "$PROJECT_DIR" || { echo "ERROR: Project dir not found at $PROJECT_DIR"; exit 1; }

# Add src/ to Python path so imports like "from run_ablation_experiments import ..."
# work even when running from the project root
export PYTHONPATH="$PROJECT_DIR/src:$PYTHONPATH"

# ── Print job info ────────────────────────────────────────────
echo "══════════════════════════════════════════════════════════"
echo "PPI Full Pipeline"
echo "══════════════════════════════════════════════════════════"
echo "Job ID:       $SLURM_JOB_ID"
echo "Node:         $SLURM_NODELIST"
echo "CPUs:         $SLURM_CPUS_PER_TASK"
echo "GPUs:         $CUDA_VISIBLE_DEVICES"
echo "Working dir:  $(pwd)"
echo "PYTHONPATH:   $PYTHONPATH"
echo "Start time:   $(date)"
echo "══════════════════════════════════════════════════════════"

"$PYTHON" -c "
import torch
print(f'PyTorch {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

# ══════════════════════════════════════════════════════════════
# PHASE 1: Core ablation (generates .npz curves + base results)
# ══════════════════════════════════════════════════════════════
echo ""
echo "━━━ [1/10] Ablation Study (full data, sequence + network) ━━━"
"$PYTHON" src/run_ablation_experiments.py
echo "✓ Ablation complete"

# ══════════════════════════════════════════════════════════════
# PHASE 2: Depends on ablation .npz curves
# ══════════════════════════════════════════════════════════════
echo ""
echo "━━━ [2/10] Sequence+Network Capped Run (150K) ━━━"
"$PYTHON" src/run_seqnet_capped.py
echo "✓ Seqnet capped complete"

echo ""
echo "━━━ [3/10] Plot ROC/PR Curves ━━━"
"$PYTHON" src/plot_curves.py
echo "✓ Curves plotted"

echo ""
echo "━━━ [4/10] Case Study ━━━"
"$PYTHON" src/case_study.py
echo "✓ Case study complete"

echo ""
echo "━━━ [5/10] Error Analysis (full data) ━━━"
"$PYTHON" src/error_analysis.py
echo "✓ Error analysis complete"

# ══════════════════════════════════════════════════════════════
# PHASE 3: Independent scripts
# ══════════════════════════════════════════════════════════════
echo ""
echo "━━━ [6/10] Model Comparison (LogReg, RF, XGBoost, MLP) ━━━"
"$PYTHON" src/improved/compare_models.py
echo "✓ Model comparison complete"

echo ""
echo "━━━ [7/10] Two-Tower MLP Training (PyTorch, GPU) ━━━"
"$PYTHON" src/improved/train_improved_mlp.py
echo "✓ Two-Tower MLP complete"

echo ""
echo "━━━ [8/10] Attention MLP Training (PyTorch, GPU) ━━━"
"$PYTHON" src/improved/train_attention_mlp.py
echo "✓ Attention MLP complete"

echo ""
echo "━━━ [9/10] SHAP Explainability (XGBoost) ━━━"
"$PYTHON" src/improved/explain_model.py
echo "✓ SHAP analysis complete"

echo ""
echo "━━━ [10/10] XAI Full Report ━━━"
"$PYTHON" src/xai_ppi_analysis.py
echo "✓ XAI report complete"

# ══════════════════════════════════════════════════════════════
# DONE
# ══════════════════════════════════════════════════════════════
echo ""
echo "══════════════════════════════════════════════════════════"
echo "ALL DONE"
echo "End time: $(date)"
echo "══════════════════════════════════════════════════════════"
echo ""
echo "Outputs:"
echo "  data/processed/ablation_results.csv"
echo "  data/processed/ablation_results_50k.csv"
echo "  data/processed/model_comparison_results.csv"
echo "  data/processed/error_analysis.md"
echo "  data/processed/error_details.csv"
echo "  data/processed/case_study.md"
echo "  data/processed/attention_analysis.md"
echo "  data/processed/xai_report.html"
echo "  models/improved_mlp.pt"
echo "  models/attention_mlp.pt"
echo "  figures/roc_curves.png"
echo "  figures/pr_curves.png"
echo "  plots/attention_weights.png"
echo "  plots/confusion_matrices.png"
echo "  plots/shap_summary.png"

#!/bin/bash
set -euo pipefail

# Evaluate the progressive FCVR (Full-Covariance Variational Router) model for Granite
# on OBQA, with FCVR on the last 5 MoE layers (27,28,29,30,31).
#
# This reconstructs the EXACT training-time stack (see prepare_model_fcvr in
# evaluate_fcvr.py):
#   1. Load the Stage 1 KVQ LoRA adapter          -> ./adapters/granite-obqa
#   2. Load the Stage 2a MAP routers into ALL 32 layers
#                                                 -> ./router_weights/base/granite_obqa
#   3. Swap layers 27-31 to FCVR (mean_base seeded from the MAP routers) and load their
#      trained variational nets                    -> ./router_weights/fcvr/fcvr-granite-obqa
# (The MAP dir and FCVR dir are derived from the shortcodes inside the adapter helpers,
#  identically to fcvr-tuning.py, so no explicit weight paths are passed.)
#
# It then produces the ID calibration report (ACC/NLL/ECE/MCE) and the OOD detection
# report with AUROC/AUPRC for: answer_entropy, gate_ent_all, gate_ent_susceptible, and
# FCVR's Inf-Logit-Var (trace of the posterior covariance = ||L||_F^2).
#
# Prerequisites (already produced for granite/obqa on quail-1):
#   - Stage 1 KVQ adapter:       ./adapters/granite-obqa
#   - Stage 2a MAP router wts:   ./router_weights/base/granite_obqa/layer_0..31_weights.pt
#   - Stage 2b FCVR router wts:  ./router_weights/fcvr/fcvr-granite-obqa/layer_27..31_weights.pt
#
# Run from the repo root inside the moe_env conda env, e.g.:
#   bash evaluate_fcvr_5_layers.sh
# For overnight, detach it:
#   nohup bash evaluate_fcvr_5_layers.sh &

# --- Log everything to a timestamped file while still printing to console ---
mkdir -p logs
LOG_FILE="logs/evaluate-fcvr-5-layers-granite-obqa-$(date +%Y%m%d-%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Logging to ${LOG_FILE}"

# evaluate_fcvr.py lives at the repo root (its imports are relative to the repo root),
# so make sure we're being run from there.
if [ ! -f "evaluate_fcvr.py" ]; then
    echo "ERROR: evaluate_fcvr.py not found. Run this script from the repo root." >&2
    exit 1
fi

# --- Parameters ---
MODEL_SHORTCODE="granite"
DATASET_SHORTCODE="obqa"
SEED=42
BATCH_SIZE=8
NUM_SAMPLES=35              # S=35 MC samples at eval (paper protocol)
SWAP_LAYERS="27 28 29 30 31"

BASE_ADAPTER_PATH="./adapters/${MODEL_SHORTCODE}-${DATASET_SHORTCODE}"
RUN_NAME="fcvr-${MODEL_SHORTCODE}-${DATASET_SHORTCODE}"

mkdir -p ./results/fcvr

echo "===================================================="
echo "FCVR evaluation (last-5 layers: ${SWAP_LAYERS})"
echo "Model: ${MODEL_SHORTCODE} | Dataset: ${DATASET_SHORTCODE}"
echo "batch=${BATCH_SIZE} num_samples=${NUM_SAMPLES} seed=${SEED}"
echo "===================================================="

# ====================================================
# Task 1: In-Distribution (ID) Calibration
# ====================================================
echo "----------------------------------------------------"
echo "Task 1: ID Calibration on ${DATASET_SHORTCODE}"
echo "----------------------------------------------------"

python evaluate_fcvr.py \
    --method "fcvr" \
    --task "id_calibration" \
    --model_shortcode "$MODEL_SHORTCODE" \
    --dataset_shortcode "$DATASET_SHORTCODE" \
    --kvq_adapter_path "$BASE_ADAPTER_PATH" \
    --output_json_path "./results/fcvr/id_calib_${RUN_NAME}_layers-27-28-29-30-31.json" \
    --swap_layers $SWAP_LAYERS \
    --num_samples "$NUM_SAMPLES" \
    --batch_size "$BATCH_SIZE" \
    --seed "$SEED"

# ====================================================
# Task 2: Out-of-Distribution (OOD) Detection (ID anchor: obqa)
# ====================================================
echo "----------------------------------------------------"
echo "Task 2: OOD Detection (ID anchor: obqa)"
echo "----------------------------------------------------"

python evaluate_fcvr.py \
    --method "fcvr" \
    --task "ood_detection" \
    --model_shortcode "$MODEL_SHORTCODE" \
    --dataset_shortcode "$DATASET_SHORTCODE" \
    --kvq_adapter_path "$BASE_ADAPTER_PATH" \
    --output_json_path "./results/fcvr/ood_detect_${RUN_NAME}_layers-27-28-29-30-31.json" \
    --swap_layers $SWAP_LAYERS \
    --num_samples "$NUM_SAMPLES" \
    --batch_size "$BATCH_SIZE" \
    --seed "$SEED"

echo "===================================================="
echo "FCVR evaluation complete."
echo "  ID calib:   ./results/fcvr/id_calib_${RUN_NAME}_layers-27-28-29-30-31.json"
echo "  OOD detect: ./results/fcvr/ood_detect_${RUN_NAME}_layers-27-28-29-30-31.json"
echo "  Log:        ${LOG_FILE}"
echo "===================================================="

#!/bin/bash
set -euo pipefail

# FCVR evaluation that MIRRORS Albus' original evaluate script
# (scripts/python/evaluate.py) and adds the Inf-Logit-Var metric.
#
# Mirrored logic (via evaluate_fcvr_original.py):
#   1. Load the Stage 1 KVQ LoRA adapter                 -> ./adapters/granite-obqa
#   2. Albus' original NON-FAITHFUL swap: replace layers 27-31 with FCVR routers whose
#      mean_base is seeded from the untuned Granite router (NOT the MAP router), then
#   3. Load the trained FCVR weights for those layers    -> ./router_weights/fcvr/fcvr-granite-obqa
#   4. ID calibration report (ACC/NLL/ECE/MCE) and OOD detection over Albus' original
#      OOD set {arc_c, medmcqa_med, mmlu_law, sciq}, scoring TWO signals:
#        - answer_entropy   (predictive entropy over the answer distribution, his signal)
#        - inf_log_var      (FCVR trace of posterior covariance = ||L||_F^2, the addition)
#
# NOTE: unlike evaluate_fcvr_5_layers.sh, this deliberately does NOT do the faithful MAP
# reconstruction — it mirrors Albus' original prepare_model exactly, so layers 0-26 keep
# the untuned Granite routers and FCVR mean_base is Granite-seeded. Use this for a
# like-for-like comparison against the original pipeline, not for paper-faithful numbers.
#
# Prerequisites (already produced for granite/obqa on quail-1):
#   - Stage 1 KVQ adapter:       ./adapters/granite-obqa
#   - Stage 2b FCVR router wts:  ./router_weights/fcvr/fcvr-granite-obqa/layer_27..31_weights.pt
#
# Run from the repo root inside the moe_env conda env, e.g.:
#   bash evaluate_fcvr_original.sh
# For overnight, detach it:
#   nohup bash evaluate_fcvr_original.sh &

# --- Log everything to a timestamped file while still printing to console ---
mkdir -p logs
LOG_FILE="logs/evaluate-fcvr-original-granite-obqa-$(date +%Y%m%d-%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Logging to ${LOG_FILE}"

if [ ! -f "evaluate_fcvr_original.py" ]; then
    echo "ERROR: evaluate_fcvr_original.py not found. Run this script from the repo root." >&2
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
# Albus' original prepare_model loads FCVR layer weights from this dir (per layer).
ROUTER_WEIGHTS_PATH="./router_weights/fcvr/${RUN_NAME}"

mkdir -p ./results/fcvr

echo "===================================================="
echo "FCVR eval (Albus' original logic + Inf-Logit-Var)"
echo "Model: ${MODEL_SHORTCODE} | Dataset: ${DATASET_SHORTCODE} | Layers: ${SWAP_LAYERS}"
echo "batch=${BATCH_SIZE} num_samples=${NUM_SAMPLES} seed=${SEED}"
echo "===================================================="

# ====================================================
# Task 1: In-Distribution (ID) Calibration
# ====================================================
echo "----------------------------------------------------"
echo "Task 1: ID Calibration on ${DATASET_SHORTCODE}"
echo "----------------------------------------------------"

python evaluate_fcvr_original.py \
    --method "fcvr" \
    --task "id_calibration" \
    --model_shortcode "$MODEL_SHORTCODE" \
    --dataset_shortcode "$DATASET_SHORTCODE" \
    --kvq_adapter_path "$BASE_ADAPTER_PATH" \
    --router_weights_path "$ROUTER_WEIGHTS_PATH" \
    --output_json_path "./results/fcvr/orig_id_calib_${RUN_NAME}_layers-27-28-29-30-31.json" \
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

python evaluate_fcvr_original.py \
    --method "fcvr" \
    --task "ood_detection" \
    --model_shortcode "$MODEL_SHORTCODE" \
    --dataset_shortcode "$DATASET_SHORTCODE" \
    --kvq_adapter_path "$BASE_ADAPTER_PATH" \
    --router_weights_path "$ROUTER_WEIGHTS_PATH" \
    --output_json_path "./results/fcvr/orig_ood_detect_${RUN_NAME}_layers-27-28-29-30-31.json" \
    --swap_layers $SWAP_LAYERS \
    --num_samples "$NUM_SAMPLES" \
    --batch_size "$BATCH_SIZE" \
    --seed "$SEED"

echo "===================================================="
echo "FCVR (original-logic) evaluation complete."
echo "  ID calib:   ./results/fcvr/orig_id_calib_${RUN_NAME}_layers-27-28-29-30-31.json"
echo "  OOD detect: ./results/fcvr/orig_ood_detect_${RUN_NAME}_layers-27-28-29-30-31.json"
echo "  Log:        ${LOG_FILE}"
echo "===================================================="

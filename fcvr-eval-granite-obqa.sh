#!/bin/bash
# ============================================================================
# FCVR (VGLR-FC) evaluation -- Granite-MoE / OBQA, PAPER-FAITHFUL.
#
# Evaluates the pretrained-prior FCVR runs trained by fcvr-tuning-granite-obqa.sh,
# one per beta in the paper's VGLR grid {0.01, 0.1}. For each run:
#   - id_calibration  -> ACC / NLL / ECE / MCE
#   - ood_detection   -> answer_entropy + inf_log_var (tr posterior cov = ||L||_F^2)
#
# Each run's results go to their own JSON files (tagged by suffix) -- nothing
# overwrites anything. Can be run standalone to re-evaluate without retraining
# (it is also invoked at the end of the training script).
#
# Prereq: Stage-1 adapter ./adapters/granite-obqa, and per beta the FCVR weights
# in ./router_weights/fcvr/fcvr-granite-obqa-pretrained-prior-beta<b>/ .
# (pretrained-prior does NOT need the Stage-2a MAP router weights.)
# ============================================================================

set -eo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# --- Optional: activate conda env (uncomment if your tmux shell hasn't) ---
# source "$(conda info --base)/etc/profile.d/conda.sh"
# conda activate moe_env

echo "Repo root: $REPO_ROOT"
echo "Python:    $(which python)"

# --- Common parameters (must match training) ---
MODEL_SHORTCODE="granite"
DATASET_SHORTCODE="obqa"
SEED=42
BATCH_SIZE=8
NUM_SAMPLES=35
PRIOR_SOURCE="pretrained"
LAYERS=(5 6 7 8 19 20 28 29 30 31)
BETAS=(0.01 0.1)

BASE_ADAPTER_PATH="./adapters/${MODEL_SHORTCODE}-${DATASET_SHORTCODE}"
RESULTS_DIR="./results/fcvr"

if [ ! -d "$BASE_ADAPTER_PATH" ]; then
    echo "ERROR: Stage-1 adapter not found at $BASE_ADAPTER_PATH" >&2
    exit 1
fi
mkdir -p "$RESULTS_DIR"

LAYER_TAG=$(printf '%s-' "${LAYERS[@]}"); LAYER_TAG=${LAYER_TAG%-}

for BETA in "${BETAS[@]}"; do
    SUFFIX="pretrained-prior-beta${BETA}"
    FCVR_WEIGHTS_DIR="./router_weights/fcvr/fcvr-${MODEL_SHORTCODE}-${DATASET_SHORTCODE}-${SUFFIX}"

    echo ""
    echo "===================================================="
    echo "EVAL beta=${BETA}  suffix=${SUFFIX}"
    echo "Layers: ${LAYERS[*]} | MC samples: ${NUM_SAMPLES}"
    echo "===================================================="

    # Prereq: FCVR weights for each layer in this run's dir.
    for i in "${LAYERS[@]}"; do
        if [ ! -f "${FCVR_WEIGHTS_DIR}/layer_${i}_weights.pt" ]; then
            echo "ERROR: missing FCVR weights ${FCVR_WEIGHTS_DIR}/layer_${i}_weights.pt" >&2
            echo "       Train beta=${BETA} first." >&2
            exit 1
        fi
    done

    # --- Task 1: ID calibration ---
    echo "---- beta=${BETA}: Task 1 ID calibration ----"
    python evaluate_fcvr.py \
        --task "id_calibration" \
        --model_shortcode "$MODEL_SHORTCODE" \
        --dataset_shortcode "$DATASET_SHORTCODE" \
        --kvq_adapter_path "$BASE_ADAPTER_PATH" \
        --swap_layers "${LAYERS[@]}" \
        --run_suffix "$SUFFIX" \
        --prior_source "$PRIOR_SOURCE" \
        --output_json_path "${RESULTS_DIR}/id_calib_fcvr-${MODEL_SHORTCODE}-${DATASET_SHORTCODE}-${SUFFIX}_layers-${LAYER_TAG}.json" \
        --num_samples "$NUM_SAMPLES" \
        --batch_size "$BATCH_SIZE" \
        --seed "$SEED"

    # --- Task 2: OOD detection (answer_entropy + inf_log_var) ---
    echo "---- beta=${BETA}: Task 2 OOD detection ----"
    python evaluate_fcvr.py \
        --task "ood_detection" \
        --model_shortcode "$MODEL_SHORTCODE" \
        --dataset_shortcode "$DATASET_SHORTCODE" \
        --kvq_adapter_path "$BASE_ADAPTER_PATH" \
        --swap_layers "${LAYERS[@]}" \
        --run_suffix "$SUFFIX" \
        --prior_source "$PRIOR_SOURCE" \
        --output_json_path "${RESULTS_DIR}/ood_detect_fcvr-${MODEL_SHORTCODE}-${DATASET_SHORTCODE}-${SUFFIX}_layers-${LAYER_TAG}.json" \
        --num_samples "$NUM_SAMPLES" \
        --batch_size "$BATCH_SIZE" \
        --seed "$SEED"
done

echo ""
echo "===================================================="
echo "FCVR evaluation complete for beta in {${BETAS[*]}}."
echo "Results in ${RESULTS_DIR}/  (files tagged -pretrained-prior-beta<b>)"
echo "===================================================="

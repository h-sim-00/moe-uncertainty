#!/bin/bash
# ============================================================================
# FCVR (VGLR-FC) evaluation -- Granite-MoE on OBQA, faithful reconstruction.
#
# Loads fine-tuned MAP routers into all 32 layers, swaps the 10 trained layers
# to FCVR (mean_base seeded from MAP), and extracts two OoD signals:
#   - answer_entropy : entropy of the predictive softmax over {A,B,C,D}
#   - inf_log_var    : tr(posterior cov) = ||L||_F^2 of the FCVR Cholesky factor
#
# Plain bash for a tmux session over ssh (NO SLURM). Runs from the repo root
# and drives evaluate_fcvr.py (also at the repo root).
#
# Prereq: Stage-1 adapter ./adapters/granite-obqa, all 32 MAP router weights in
# ./router_weights/base/granite_obqa/, and FCVR weights for each layer below in
# ./router_weights/fcvr/fcvr-granite-obqa/ (produced by fcvr-tuning-granite-obqa.sh).
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

# --- Parameters (must match the training run) ---
MODEL_SHORTCODE="granite"
DATASET_SHORTCODE="obqa"
SEED=42
BATCH_SIZE=8
NUM_SAMPLES=35

# FCVR layers (must match fcvr-tuning-granite-obqa.sh)
LAYERS=(5 6 7 8 19 20 28 29 30 31)

# Must match the training run's RUN_SUFFIX so we load the right weights dir
# (and never touch the progressive run's fcvr-granite-obqa/ weights).
RUN_SUFFIX="susceptible-nonprog"

BASE_ADAPTER_PATH="./adapters/${MODEL_SHORTCODE}-${DATASET_SHORTCODE}"
MAP_WEIGHTS_DIR="./router_weights/base/${MODEL_SHORTCODE}_${DATASET_SHORTCODE}"
FCVR_WEIGHTS_DIR="./router_weights/fcvr/fcvr-${MODEL_SHORTCODE}-${DATASET_SHORTCODE}-${RUN_SUFFIX}"
RESULTS_DIR="./results/fcvr"

# --- Prerequisite checks (fail fast) ---
if [ ! -d "$BASE_ADAPTER_PATH" ]; then
    echo "ERROR: Stage-1 adapter not found at $BASE_ADAPTER_PATH" >&2
    exit 1
fi
for i in $(seq 0 31); do
    if [ ! -f "${MAP_WEIGHTS_DIR}/layer_${i}_weights.pt" ]; then
        echo "ERROR: missing MAP router weights ${MAP_WEIGHTS_DIR}/layer_${i}_weights.pt" >&2
        echo "       Faithful reconstruction loads all 32 MAP routers." >&2
        exit 1
    fi
done
for i in "${LAYERS[@]}"; do
    if [ ! -f "${FCVR_WEIGHTS_DIR}/layer_${i}_weights.pt" ]; then
        echo "ERROR: missing FCVR weights ${FCVR_WEIGHTS_DIR}/layer_${i}_weights.pt" >&2
        echo "       Run fcvr-tuning-granite-obqa.sh first." >&2
        exit 1
    fi
done

mkdir -p "$RESULTS_DIR"

LAYER_TAG=$(printf '%s-' "${LAYERS[@]}"); LAYER_TAG=${LAYER_TAG%-}

echo "===================================================="
echo "FCVR evaluation | ${MODEL_SHORTCODE} | ${DATASET_SHORTCODE}"
echo "Layers: ${LAYERS[*]} | MC samples: ${NUM_SAMPLES}"
echo "===================================================="

# --- Task 1: ID calibration (ACC / NLL / ECE / MCE) ---
echo "---- Task 1: ID calibration ----"
python evaluate_fcvr.py \
    --task "id_calibration" \
    --model_shortcode "$MODEL_SHORTCODE" \
    --dataset_shortcode "$DATASET_SHORTCODE" \
    --kvq_adapter_path "$BASE_ADAPTER_PATH" \
    --swap_layers "${LAYERS[@]}" \
    --run_suffix "$RUN_SUFFIX" \
    --output_json_path "${RESULTS_DIR}/id_calib_fcvr-${MODEL_SHORTCODE}-${DATASET_SHORTCODE}-${RUN_SUFFIX}_layers-${LAYER_TAG}.json" \
    --num_samples "$NUM_SAMPLES" \
    --batch_size "$BATCH_SIZE" \
    --seed "$SEED"

# --- Task 2: OOD detection (answer_entropy + inf_log_var) ---
echo "---- Task 2: OOD detection ----"
python evaluate_fcvr.py \
    --task "ood_detection" \
    --model_shortcode "$MODEL_SHORTCODE" \
    --dataset_shortcode "$DATASET_SHORTCODE" \
    --kvq_adapter_path "$BASE_ADAPTER_PATH" \
    --swap_layers "${LAYERS[@]}" \
    --run_suffix "$RUN_SUFFIX" \
    --output_json_path "${RESULTS_DIR}/ood_detect_fcvr-${MODEL_SHORTCODE}-${DATASET_SHORTCODE}-${RUN_SUFFIX}_layers-${LAYER_TAG}.json" \
    --num_samples "$NUM_SAMPLES" \
    --batch_size "$BATCH_SIZE" \
    --seed "$SEED"

echo "===================================================="
echo "FCVR evaluation complete. Results in ${RESULTS_DIR}/"
echo "===================================================="

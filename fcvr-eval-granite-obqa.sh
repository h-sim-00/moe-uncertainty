#!/bin/bash
# ============================================================================
# FCVR (VGLR-FC) evaluation -- Granite-MoE / OBQA, BOTH variants.
#
# Evaluates the two FCVR variants trained by fcvr-tuning-granite-obqa.sh:
#   Variant A "pretrained-prior" -> paper-faithful (mean_base from pre-trained
#                                   Granite router; non-FCVR layers pre-trained).
#   Variant B "map-prior"        -> inherited (mean_base from fine-tuned MAP;
#                                   non-FCVR layers fine-tuned MAP).
#
# For each variant runs id_calibration (ACC/NLL/ECE/MCE) and ood_detection
# (answer_entropy + inf_log_var). Each variant's results go to their own JSON
# files -- nothing overwrites anything. Can be run standalone to re-evaluate
# without retraining (it is also invoked at the end of the training script).
#
# Prereq: Stage-1 adapter ./adapters/granite-obqa, and per variant the FCVR
# weights in ./router_weights/fcvr/fcvr-granite-obqa-<suffix>/ . Variant B also
# needs all 32 MAP router weights in ./router_weights/base/granite_obqa/ .
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
LAYERS=(5 6 7 8 19 20 28 29 30 31)

# Two variants: parallel arrays indexed together (must match training script).
PRIOR_SOURCES=("pretrained" "map")
RUN_SUFFIXES=("pretrained-prior" "map-prior")

BASE_ADAPTER_PATH="./adapters/${MODEL_SHORTCODE}-${DATASET_SHORTCODE}"
MAP_WEIGHTS_DIR="./router_weights/base/${MODEL_SHORTCODE}_${DATASET_SHORTCODE}"
RESULTS_DIR="./results/fcvr"

if [ ! -d "$BASE_ADAPTER_PATH" ]; then
    echo "ERROR: Stage-1 adapter not found at $BASE_ADAPTER_PATH" >&2
    exit 1
fi
mkdir -p "$RESULTS_DIR"

LAYER_TAG=$(printf '%s-' "${LAYERS[@]}"); LAYER_TAG=${LAYER_TAG%-}

for idx in 0 1; do
    PRIOR="${PRIOR_SOURCES[$idx]}"
    SUFFIX="${RUN_SUFFIXES[$idx]}"
    FCVR_WEIGHTS_DIR="./router_weights/fcvr/fcvr-${MODEL_SHORTCODE}-${DATASET_SHORTCODE}-${SUFFIX}"

    echo ""
    echo "===================================================="
    echo "EVAL variant: prior_source=${PRIOR}  suffix=${SUFFIX}"
    echo "Layers: ${LAYERS[*]} | MC samples: ${NUM_SAMPLES}"
    echo "===================================================="

    # Prereq: FCVR weights for each layer in this variant's dir.
    for i in "${LAYERS[@]}"; do
        if [ ! -f "${FCVR_WEIGHTS_DIR}/layer_${i}_weights.pt" ]; then
            echo "ERROR: missing FCVR weights ${FCVR_WEIGHTS_DIR}/layer_${i}_weights.pt" >&2
            echo "       Train variant '${SUFFIX}' first." >&2
            exit 1
        fi
    done
    # Variant B (map prior) also needs the 32 MAP routers at eval time.
    if [ "$PRIOR" = "map" ]; then
        for i in $(seq 0 31); do
            if [ ! -f "${MAP_WEIGHTS_DIR}/layer_${i}_weights.pt" ]; then
                echo "ERROR: missing MAP router weights ${MAP_WEIGHTS_DIR}/layer_${i}_weights.pt" >&2
                exit 1
            fi
        done
    fi

    # --- Task 1: ID calibration ---
    echo "---- ${SUFFIX}: Task 1 ID calibration ----"
    python evaluate_fcvr.py \
        --task "id_calibration" \
        --model_shortcode "$MODEL_SHORTCODE" \
        --dataset_shortcode "$DATASET_SHORTCODE" \
        --kvq_adapter_path "$BASE_ADAPTER_PATH" \
        --swap_layers "${LAYERS[@]}" \
        --run_suffix "$SUFFIX" \
        --prior_source "$PRIOR" \
        --output_json_path "${RESULTS_DIR}/id_calib_fcvr-${MODEL_SHORTCODE}-${DATASET_SHORTCODE}-${SUFFIX}_layers-${LAYER_TAG}.json" \
        --num_samples "$NUM_SAMPLES" \
        --batch_size "$BATCH_SIZE" \
        --seed "$SEED"

    # --- Task 2: OOD detection (answer_entropy + inf_log_var) ---
    echo "---- ${SUFFIX}: Task 2 OOD detection ----"
    python evaluate_fcvr.py \
        --task "ood_detection" \
        --model_shortcode "$MODEL_SHORTCODE" \
        --dataset_shortcode "$DATASET_SHORTCODE" \
        --kvq_adapter_path "$BASE_ADAPTER_PATH" \
        --swap_layers "${LAYERS[@]}" \
        --run_suffix "$SUFFIX" \
        --prior_source "$PRIOR" \
        --output_json_path "${RESULTS_DIR}/ood_detect_fcvr-${MODEL_SHORTCODE}-${DATASET_SHORTCODE}-${SUFFIX}_layers-${LAYER_TAG}.json" \
        --num_samples "$NUM_SAMPLES" \
        --batch_size "$BATCH_SIZE" \
        --seed "$SEED"
done

echo ""
echo "===================================================="
echo "FCVR evaluation of both variants complete."
echo "Results in ${RESULTS_DIR}/  (files tagged -pretrained-prior / -map-prior)"
echo "===================================================="

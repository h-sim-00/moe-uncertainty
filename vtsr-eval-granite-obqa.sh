#!/bin/bash
# ============================================================================
# EVAL ORCHESTRATOR -- VTSR (Variational Temperature Scaling Router) on
# Granite-MoE / OBQA. Evaluates the beta SWEEP {0.05, 0.1} from training.
# Runs BOTH tasks per beta: ID calibration + OoD detection.
#
# Reconstructs the trained model faithfully (all 32 layers -> fine-tuned MAP,
# then the SUSCEPTIBLE-10 layers -> VTSR with their trained temperature nets)
# and evaluates:
#   ID calibration : ACC / NLL / ECE / MCE on OBQA.
#   OoD detection  : AUROC/AUPRC on the paper's Table-2 targets
#                    {arc_e, arc_c (near), medmcqa_med, mmlu_law (far)}
#                    with OBQA as the fixed ID anchor, for THREE signals:
#                      answer_entropy, gate_ent (H(softmax(l_det/T))), inf_temp (T).
#
# The weights path is DERIVED from the training knobs (temperature_mode +
# run_suffix), so eval reads exactly what training wrote:
#   ./router_weights/vtsr_${TEMPERATURE_MODE}/vtsr-granite-obqa-susceptible10-beta<b>/
#
# Plain bash for a tmux session over ssh (NO SLURM). Runs from the repo root.
#
# Prereq: vtsr-tuning-granite-obqa.sh has completed for each beta, PLUS the
#         Stage-1 adapter and 32 MAP routers that the reconstruction loads.
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

# --- Parameters (must match the training runs) ---
MODEL_SHORTCODE="granite"
DATASET_SHORTCODE="obqa"
SEED=42
BATCH_SIZE=8
TEMPERATURE_MODE="shared"
BETAS=(0.05 0.1)

# Susceptible-10 layers -- the VTSR layers that were trained.
LAYERS=(5 6 7 8 19 20 28 29 30 31)

BASE_ADAPTER_PATH="./adapters/${MODEL_SHORTCODE}-${DATASET_SHORTCODE}"
MAP_WEIGHTS_DIR="./router_weights/base/${MODEL_SHORTCODE}_${DATASET_SHORTCODE}"

# --- Prereqs shared across betas ---
if [ ! -d "$BASE_ADAPTER_PATH" ]; then
    echo "ERROR: Stage-1 adapter not found at $BASE_ADAPTER_PATH" >&2
    exit 1
fi
if [ ! -f "${MAP_WEIGHTS_DIR}/layer_0_weights.pt" ]; then
    echo "ERROR: MAP routers not found under ${MAP_WEIGHTS_DIR}/ (needed for reconstruction)." >&2
    exit 1
fi

mkdir -p results/vtsr logs

for BETA in "${BETAS[@]}"; do
    RUN_SUFFIX="susceptible10-beta${BETA}"
    RUN_NAME="vtsr-${MODEL_SHORTCODE}-${DATASET_SHORTCODE}-${RUN_SUFFIX}"
    VTSR_WEIGHTS_DIR="./router_weights/vtsr_${TEMPERATURE_MODE}/${RUN_NAME}"

    # --- Prereq: this beta's trained weights ---
    for l in "${LAYERS[@]}"; do
        if [ ! -f "${VTSR_WEIGHTS_DIR}/layer_${l}_weights.pt" ]; then
            echo "ERROR: missing trained VTSR weight ${VTSR_WEIGHTS_DIR}/layer_${l}_weights.pt" >&2
            echo "       Run vtsr-tuning-granite-obqa.sh (beta=${BETA}) first." >&2
            exit 1
        fi
    done

    echo "===================================================="
    echo "VTSR evaluation (beta=${BETA}): layers=${LAYERS[*]} suffix=${RUN_SUFFIX}"
    echo "  weights: ${VTSR_WEIGHTS_DIR}/"
    echo "===================================================="

    # --- Task 1: ID calibration ---
    echo "---- ID calibration (beta=${BETA}) ----"
    python evaluate_vtsr.py \
        --task "id_calibration" \
        --model_shortcode "$MODEL_SHORTCODE" \
        --dataset_shortcode "$DATASET_SHORTCODE" \
        --kvq_adapter_path "$BASE_ADAPTER_PATH" \
        --swap_layers "${LAYERS[@]}" \
        --temperature_mode "$TEMPERATURE_MODE" \
        --run_suffix "$RUN_SUFFIX" \
        --output_json_path "./results/vtsr/id_calib_${RUN_NAME}.json" \
        --batch_size "$BATCH_SIZE" \
        --seed "$SEED" \
        2>&1 | tee "logs/vtsr-eval-id-calib-${RUN_SUFFIX}.log"

    # --- Task 2: OoD detection ---
    echo "---- OoD detection (beta=${BETA}) ----"
    python evaluate_vtsr.py \
        --task "ood_detection" \
        --model_shortcode "$MODEL_SHORTCODE" \
        --dataset_shortcode "$DATASET_SHORTCODE" \
        --kvq_adapter_path "$BASE_ADAPTER_PATH" \
        --swap_layers "${LAYERS[@]}" \
        --temperature_mode "$TEMPERATURE_MODE" \
        --run_suffix "$RUN_SUFFIX" \
        --output_json_path "./results/vtsr/ood_detect_${RUN_NAME}.json" \
        --batch_size "$BATCH_SIZE" \
        --seed "$SEED" \
        2>&1 | tee "logs/vtsr-eval-ood-detect-${RUN_SUFFIX}.log"

    echo "Done eval beta=${BETA}:"
    echo "  ./results/vtsr/id_calib_${RUN_NAME}.json"
    echo "  ./results/vtsr/ood_detect_${RUN_NAME}.json"
done

echo "===================================================="
echo "VTSR evaluation complete for betas: ${BETAS[*]}"
echo "===================================================="

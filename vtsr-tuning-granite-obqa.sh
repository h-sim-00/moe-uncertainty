#!/bin/bash
# ============================================================================
# ORCHESTRATOR -- VTSR (Variational Temperature Scaling Router) Stage 2 on
# Granite-MoE / OBQA.
#
# Trains the SUSCEPTIBLE-10 layers {5,6,7,8,19,20,28,29,30,31} JOINTLY
# (NON-progressive): a single run per beta swaps all 10 layers to VTSR and
# unfreezes all 10 temperature_nets at once.
#
# BETA SWEEP: runs the paper's VTSR grid beta in {0.05, 0.1} sequentially and
# picks the winner later by val NLL / target ECE. The temperature penalty is
# the per-token MEAN of -log(T) (summed over layers), so beta carries its
# intended paper-scale weight (a per-token SUM would swamp the task loss and
# make T explode -- see the earlier degenerate beta=1e-3 run).
#
# ISOLATION: each beta writes to its OWN dir via --run_suffix, so nothing
# overwrites anything -- not the other beta, not a last-5 run, not the old
# degenerate susceptible10 run:
#   ./router_weights/vtsr_shared/vtsr-granite-obqa-susceptible10-beta0.05/
#   ./router_weights/vtsr_shared/vtsr-granite-obqa-susceptible10-beta0.1/
# A per-beta preflight check ALSO aborts if that beta's dir already holds
# weights, so a re-run can't silently clobber a prior run.
#
# Plain bash for a tmux session over ssh (NO SLURM). Runs from the repo root.
#
# Prereq: Stage-1 (KVQ) adapter at ./adapters/granite-obqa AND all 32 MAP
#         router weights in ./router_weights/base/granite_obqa/ (VTSR loads
#         all 32 MAP routers as the frozen base before swapping the 10).
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

# --- Parameters (repo defaults, except beta = paper VTSR grid) ---
MODEL_SHORTCODE="granite"
DATASET_SHORTCODE="obqa"
SEED=42
EPOCHS=10
BATCH_SIZE=8
LEARNING_RATE=1e-5
TEMPERATURE_MODE="shared"    # directory label only; router always predicts a scalar T
BETAS=(0.05 0.1)             # paper VTSR grid for the -log(T) penalty weight

# Susceptible-10 layers, trained jointly (non-progressive) in ONE run per beta.
LAYERS=(5 6 7 8 19 20 28 29 30 31)

BASE_ADAPTER_PATH="./adapters/${MODEL_SHORTCODE}-${DATASET_SHORTCODE}"
MAP_WEIGHTS_DIR="./router_weights/base/${MODEL_SHORTCODE}_${DATASET_SHORTCODE}"

# --- Prereq: Stage-1 adapter ---
if [ ! -d "$BASE_ADAPTER_PATH" ]; then
    echo "ERROR: Stage-1 adapter not found at $BASE_ADAPTER_PATH" >&2
    exit 1
fi

# --- Prereq: all 32 MAP routers ---
for i in $(seq 0 31); do
    if [ ! -f "${MAP_WEIGHTS_DIR}/layer_${i}_weights.pt" ]; then
        echo "ERROR: missing MAP router weight ${MAP_WEIGHTS_DIR}/layer_${i}_weights.pt" >&2
        echo "       VTSR needs all 32 Stage-2a MAP routers before it can start." >&2
        exit 1
    fi
done

mkdir -p logs

for BETA in "${BETAS[@]}"; do
    RUN_SUFFIX="susceptible10-beta${BETA}"
    SAVE_DIR="./router_weights/vtsr_${TEMPERATURE_MODE}/vtsr-${MODEL_SHORTCODE}-${DATASET_SHORTCODE}-${RUN_SUFFIX}"

    # --- No-overwrite guard for THIS beta ---
    if compgen -G "${SAVE_DIR}/layer_*_weights.pt" > /dev/null; then
        echo "ERROR: weights already exist in ${SAVE_DIR}" >&2
        echo "       Refusing to overwrite. Move/rename that dir, then re-run." >&2
        exit 1
    fi

    echo "===================================================="
    echo "VTSR training (beta=${BETA}): ${MODEL_SHORTCODE} on ${DATASET_SHORTCODE}"
    echo "  susceptible-10 layers: ${LAYERS[*]}"
    echo "  epochs=${EPOCHS} batch=${BATCH_SIZE} lr=${LEARNING_RATE} beta=${BETA} mode=${TEMPERATURE_MODE} suffix=${RUN_SUFFIX} seed=${SEED}"
    echo "  -> weights: ${SAVE_DIR}/"
    echo "===================================================="

    # Non-progressive: swap ALL 10, load NONE, train ALL 10, single invocation.
    python vtsr-tuning.py \
        --model_shortcode "$MODEL_SHORTCODE" \
        --dataset_shortcode "$DATASET_SHORTCODE" \
        --base_adapter_path "$BASE_ADAPTER_PATH" \
        --swap_layers "${LAYERS[@]}" \
        --train_layers "${LAYERS[@]}" \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --lr "$LEARNING_RATE" \
        --temp_penalty_weight "$BETA" \
        --temperature_mode "$TEMPERATURE_MODE" \
        --run_suffix "$RUN_SUFFIX" \
        --seed "$SEED" \
        2>&1 | tee "logs/vtsr-granite-obqa-${RUN_SUFFIX}.log"

    echo "Done beta=${BETA} -> ${SAVE_DIR}/"
done

echo "===================================================="
echo "VTSR beta sweep complete: ${BETAS[*]}"
echo "Compare val NLL across logs/vtsr-granite-obqa-susceptible10-beta*.log to pick the winner,"
echo "then run ./vtsr-eval-granite-obqa.sh (evaluates both betas)."
echo "===================================================="

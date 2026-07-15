#!/bin/bash
# ============================================================================
# ORCHESTRATOR -- VTSR (Variational Temperature Scaling Router) Stage 2 on
# Granite-MoE / OBQA.
#
# Trains the SUSCEPTIBLE-10 layers {5,6,7,8,19,20,28,29,30,31} JOINTLY
# (NON-progressive): a single run swaps all 10 layers to VTSR and unfreezes
# all 10 temperature_nets at once. (No load step -- nothing is reloaded
# between layers, unlike the earlier progressive last-5 scheme.)
#
# ISOLATION: this run writes to its OWN directory via --run_suffix, so it can
# NEVER overwrite a last-5 run's weights -- not even on the overlap layers
# 28,29,30,31 that both selections share:
#   susceptible-10 -> ./router_weights/vtsr_shared/vtsr-granite-obqa-susceptible10/
#   last-5 (if run)-> ./router_weights/vtsr_shared/vtsr-granite-obqa/
# A preflight check below ALSO aborts if this run's own dir already holds
# weights, so a re-run can't silently clobber a prior susceptible-10 run.
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

# --- Parameters (repo defaults) ---
MODEL_SHORTCODE="granite"
DATASET_SHORTCODE="obqa"
SEED=42
EPOCHS=10
BATCH_SIZE=8
LEARNING_RATE=1e-5
TEMP_PENALTY_WEIGHT=1e-3     # beta on the -log(T) collapse penalty
TEMPERATURE_MODE="shared"    # directory label only; router always predicts a scalar T
RUN_SUFFIX="susceptible10"   # isolates this run's weights from the last-5 run

# Susceptible-10 layers, trained jointly (non-progressive) in ONE run.
LAYERS=(5 6 7 8 19 20 28 29 30 31)

BASE_ADAPTER_PATH="./adapters/${MODEL_SHORTCODE}-${DATASET_SHORTCODE}"
MAP_WEIGHTS_DIR="./router_weights/base/${MODEL_SHORTCODE}_${DATASET_SHORTCODE}"

# Exactly where granite_adapter.save_granite_bayesian_routers will write:
#   ./router_weights/vtsr_<mode>/vtsr-<model>-<dataset>-<suffix>/
SAVE_DIR="./router_weights/vtsr_${TEMPERATURE_MODE}/vtsr-${MODEL_SHORTCODE}-${DATASET_SHORTCODE}-${RUN_SUFFIX}"

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

# --- No-overwrite guard: refuse to touch an existing susceptible-10 run ---
if compgen -G "${SAVE_DIR}/layer_*_weights.pt" > /dev/null; then
    echo "ERROR: weights already exist in ${SAVE_DIR}" >&2
    echo "       Refusing to overwrite. Move/rename that dir, or change RUN_SUFFIX, then re-run." >&2
    exit 1
fi

mkdir -p logs

echo "===================================================="
echo "Starting NON-progressive VTSR Fine-tuning: ${MODEL_SHORTCODE} on ${DATASET_SHORTCODE}"
echo "  susceptible-10 layers: ${LAYERS[*]}"
echo "  epochs=${EPOCHS} batch=${BATCH_SIZE} lr=${LEARNING_RATE} beta=${TEMP_PENALTY_WEIGHT} mode=${TEMPERATURE_MODE} suffix=${RUN_SUFFIX} seed=${SEED}"
echo "  -> weights: ${SAVE_DIR}/"
echo "===================================================="

# Non-progressive: swap ALL 10, load NONE, train ALL 10, in a single invocation.
python vtsr-tuning.py \
    --model_shortcode "$MODEL_SHORTCODE" \
    --dataset_shortcode "$DATASET_SHORTCODE" \
    --base_adapter_path "$BASE_ADAPTER_PATH" \
    --swap_layers "${LAYERS[@]}" \
    --train_layers "${LAYERS[@]}" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LEARNING_RATE" \
    --temp_penalty_weight "$TEMP_PENALTY_WEIGHT" \
    --temperature_mode "$TEMPERATURE_MODE" \
    --run_suffix "$RUN_SUFFIX" \
    --seed "$SEED" \
    2>&1 | tee "logs/vtsr-granite-obqa-susceptible10.log"

echo "===================================================="
echo "Non-progressive VTSR training complete."
echo "Weights: ${SAVE_DIR}/"
echo "===================================================="

#!/bin/bash
# ============================================================================
# Stage 2 (paper: Variational Inference learning) -- FCVR / VGLR-FC training
# Granite-MoE + FCVR on OBQA. All selected layers trained JOINTLY in a single
# run (NO progressive per-layer training).
#
# Plain bash for running inside a tmux session over ssh (NO SLURM).
# Adapted from scripts/bash/fcvr-tuning.sh but:
#   - no SLURM header / al1624 cluster paths (env is handled by
#     utils.setup_environment via os.environ.setdefault)
#   - layer set = Susceptible layers {5-8, 19-20, 28-31} instead of last-5
#   - single joint fit: swap all selected layers (init from MAP), train them
#     all at once, load none -- rather than the repo's deepest-first loop
#   - runs from the repo root; adds repo root to PYTHONPATH so the python
#     source under scripts/python/ resolves `model` and `utils`.
#
# Prereq: Stage 1 (KVQ) adapter at ./adapters/granite-obqa AND all 32 MAP
# router weights at ./router_weights/base/granite_obqa/layer_{0..31}_weights.pt
# (FCVR loads all 32 MAP routers as the prior even though it only trains a
# subset). FCVR weights are written to ./router_weights/fcvr/.
# ============================================================================

set -eo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# --- Optional: activate conda env (uncomment if your tmux shell hasn't) ---
# source "$(conda info --base)/etc/profile.d/conda.sh"
# conda activate moe_env

echo "Repo root:      $REPO_ROOT"
echo "Python:         $(which python)"

# --- Parameters (repo defaults) ---
MODEL_SHORTCODE="granite"
DATASET_SHORTCODE="obqa"
SEED=42
EPOCHS=10
BATCH_SIZE=4
LEARNING_RATE=1e-5
BETA=0.01

# Layers to apply FCVR to (trained jointly in one run).
# Chosen set: {5,6,7,8, 19,20, 28,29,30,31}  (Susceptible-10 minus layers 0-1)
LAYERS=(5 6 7 8 19 20 28 29 30 31)

BASE_ADAPTER_PATH="./adapters/${MODEL_SHORTCODE}-${DATASET_SHORTCODE}"
MAP_WEIGHTS_DIR="./router_weights/base/${MODEL_SHORTCODE}_${DATASET_SHORTCODE}"

# --- Prerequisite checks (fail fast before loading the model) ---
if [ ! -d "$BASE_ADAPTER_PATH" ]; then
    echo "ERROR: Stage-1 adapter not found at $BASE_ADAPTER_PATH" >&2
    exit 1
fi
for i in $(seq 0 31); do
    if [ ! -f "${MAP_WEIGHTS_DIR}/layer_${i}_weights.pt" ]; then
        echo "ERROR: missing MAP router weights ${MAP_WEIGHTS_DIR}/layer_${i}_weights.pt" >&2
        echo "       FCVR needs all 32 MAP routers as the prior. Run Stage 2a first." >&2
        exit 1
    fi
done

mkdir -p logs

echo "===================================================="
echo "Starting Joint FCVR (Stage 2) Fine-tuning"
echo "Model: ${MODEL_SHORTCODE} | Dataset: ${DATASET_SHORTCODE}"
echo "Layers (trained jointly): ${LAYERS[*]}"
echo "epochs=${EPOCHS} batch=${BATCH_SIZE} lr=${LEARNING_RATE} beta=${BETA} seed=${SEED}"
echo "===================================================="

# Joint fit: swap = train = all selected layers; load nothing (all init from MAP).
python scripts/python/fcvr-tuning.py \
    --model_shortcode "$MODEL_SHORTCODE" \
    --dataset_shortcode "$DATASET_SHORTCODE" \
    --base_adapter_path "$BASE_ADAPTER_PATH" \
    --swap_layers "${LAYERS[@]}" \
    --load_layers \
    --train_layers "${LAYERS[@]}" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LEARNING_RATE" \
    --beta "$BETA" \
    --seed "$SEED"

echo "===================================================="
echo "FCVR training complete."
echo "Weights saved under ./router_weights/fcvr/fcvr-${MODEL_SHORTCODE}-${DATASET_SHORTCODE}/"
echo "===================================================="

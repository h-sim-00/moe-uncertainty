#!/bin/bash
set -euo pipefail

# Progressive FCVR (Full-Covariance Variational Router) fine-tuning for Granite on OBQA.
# Mirrors the repository's progressive scheme: train the last 5 MoE layers (31 -> 27),
# one layer at a time, loading the already-trained FCVR layers as frozen context.
#
# Prerequisites (Stage 1 + Stage 2a, already done for granite/obqa):
#   - Stage 1 KVQ adapter:      ./adapters/granite-obqa
#   - Stage 2a MAP router wts:  ./router_weights/base/granite_obqa
#
# Run from the repo root inside the moe_env conda env, e.g.:
#   bash scripts/bash/fcvr-tuning-granite-obqa.sh
# For overnight, detach it:
#   nohup bash scripts/bash/fcvr-tuning-granite-obqa.sh &

# --- Log everything to a timestamped file while still printing to console ---
mkdir -p logs
LOG_FILE="logs/fcvr-granite-obqa-$(date +%Y%m%d-%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Logging to ${LOG_FILE}"

# fcvr-tuning.py only lives under scripts/python/ and imports model/utils relative to
# the repo root, so copy it up before running (matches map-tuning-granite-obqa.sh).
cp ./scripts/python/fcvr-tuning.py .
# Ensure the temp copy is removed even if training fails.
trap 'rm -f fcvr-tuning.py' EXIT

# --- Parameters (repo defaults; beta overridden to 0.1 per request) ---
MODEL_SHORTCODE="granite"
DATASET_SHORTCODE="obqa"
SEED=42
EPOCHS=10
BATCH_SIZE=4
LEARNING_RATE=1e-5
BETA=0.1

BASE_ADAPTER_PATH="./adapters/${MODEL_SHORTCODE}-${DATASET_SHORTCODE}"

echo "===================================================="
echo "Starting Progressive FCVR Fine-tuning"
echo "Model: ${MODEL_SHORTCODE} | Dataset: ${DATASET_SHORTCODE}"
echo "epochs=${EPOCHS} batch=${BATCH_SIZE} lr=${LEARNING_RATE} beta=${BETA} seed=${SEED}"
echo "===================================================="

# Loop backwards from the last layer (31) to the 27th layer.
for i in $(seq 31 -1 27); do
    TRAIN_LAYER=$i

    # Swap layers [i .. 31] to the variational router.
    SWAP_LAYERS=()
    for j in $(seq "$i" 31); do SWAP_LAYERS+=("$j"); done

    # Load the already-trained FCVR layers [i+1 .. 31] (frozen context for this step).
    LOAD_LAYERS=()
    for j in $(seq $((i + 1)) 31); do LOAD_LAYERS+=("$j"); done

    echo "----------------------------------------------------"
    echo "Step: Training Layer ${TRAIN_LAYER}"
    echo "  - Swapping Layers: [${SWAP_LAYERS[*]}]"
    echo "  - Loading Pre-trained FCVR Layers: [${LOAD_LAYERS[*]:-none}]"
    echo "----------------------------------------------------"

    # Only pass --load_layers when non-empty (it is empty on the first step, i=31).
    # The ${arr[@]+...} form is safe under `set -u` even for an empty array.
    LOAD_ARG=()
    if [ ${#LOAD_LAYERS[@]} -gt 0 ]; then
        LOAD_ARG=(--load_layers "${LOAD_LAYERS[@]}")
    fi

    python fcvr-tuning.py \
        --model_shortcode "$MODEL_SHORTCODE" \
        --dataset_shortcode "$DATASET_SHORTCODE" \
        --base_adapter_path "$BASE_ADAPTER_PATH" \
        --swap_layers "${SWAP_LAYERS[@]}" \
        ${LOAD_ARG[@]+"${LOAD_ARG[@]}"} \
        --train_layers "$TRAIN_LAYER" \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --lr "$LEARNING_RATE" \
        --beta "$BETA" \
        --seed "$SEED"
done

echo "===================================================="
echo "All progressive FCVR training steps completed successfully."
echo "Log saved to ${LOG_FILE}"
echo "===================================================="

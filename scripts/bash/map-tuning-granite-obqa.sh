#!/bin/bash
set -euo pipefail

# Plain (non-SLURM) MAP fine-tuning pipeline for Granite on OBQA.
# Stage 1: KVQ (LoRA q/k/v) base-adapter tuning.
# Stage 2a: MAP (deterministic) router tuning.
#
# Run from the repo root, e.g.: bash scripts/bash/map-tuning-granite-obqa.sh

MODEL_SHORTCODE="granite"
DATASET_SHORTCODE="obqa"
SEED=42

# Stage 1 hyperparameters
KVQ_EPOCHS=3
KVQ_BATCH_SIZE=8
KVQ_LR=1e-4

# Stage 2a hyperparameters
MAP_EPOCHS=3
MAP_BATCH_SIZE=8
MAP_LR=1e-4

ADAPTER_PATH="./adapters/${MODEL_SHORTCODE}-${DATASET_SHORTCODE}"
# Note: router_weights/base uses an underscore-joined name internally
# (granite_adapter.py's save/load_granite_map_routers), unlike the
# hyphen-joined adapter path above — that's expected, not a typo.
ROUTER_WEIGHTS_PATH="./router_weights/base/${MODEL_SHORTCODE}_${DATASET_SHORTCODE}"

echo "===================================================="
echo "Stage 1: KVQ base-adapter tuning"
echo "Model: ${MODEL_SHORTCODE} | Dataset: ${DATASET_SHORTCODE}"
echo "===================================================="

python kvq-tuning.py \
    --model_shortcode "$MODEL_SHORTCODE" \
    --dataset_shortcode "$DATASET_SHORTCODE" \
    --epochs "$KVQ_EPOCHS" \
    --batch_size "$KVQ_BATCH_SIZE" \
    --lr "$KVQ_LR" \
    --seed "$SEED"

echo "Stage 1 complete. Adapter saved to ${ADAPTER_PATH}"

echo "===================================================="
echo "Stage 2a: MAP (deterministic) router tuning"
echo "Model: ${MODEL_SHORTCODE} | Dataset: ${DATASET_SHORTCODE}"
echo "===================================================="

# router-tuning.py only lives under scripts/python/ and imports
# model/utils relative to the repo root, so copy it up before running.
cp ./scripts/python/router-tuning.py .

python router-tuning.py \
    --model_shortcode "$MODEL_SHORTCODE" \
    --dataset_shortcode "$DATASET_SHORTCODE" \
    --base_adapter_path "$ADAPTER_PATH" \
    --epochs "$MAP_EPOCHS" \
    --batch_size "$MAP_BATCH_SIZE" \
    --lr "$MAP_LR" \
    --seed "$SEED"

rm router-tuning.py

echo "===================================================="
echo "All MAP fine-tuning steps complete."
echo "  Adapter:        ${ADAPTER_PATH}"
echo "  Router weights: ${ROUTER_WEIGHTS_PATH}"
echo "===================================================="

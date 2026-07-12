#!/bin/bash
# ============================================================================
# OVERNIGHT ORCHESTRATOR -- FCVR (VGLR-FC) Stage 2 on Granite-MoE / OBQA.
#
# Trains the Susceptible-10 layers {5-8, 19-20, 28-31} JOINTLY (non-progressive)
# in TWO variants, then evaluates BOTH. Nothing overwrites anything: each
# variant writes to its own weights dir and its own results files, and neither
# touches the progressive run's ./router_weights/fcvr/fcvr-granite-obqa/.
#
#   Variant A "pretrained-prior"  -> paper-faithful: FCVR mean_base seeded from
#                                    the PRE-TRAINED Granite router (no MAP load).
#                                    Non-FCVR layers stay pre-trained deterministic.
#   Variant B "map-prior"         -> inherited pipeline: FCVR mean_base seeded from
#                                    the fine-tuned MAP routers (Stage 2a).
#                                    Non-FCVR layers stay fine-tuned MAP.
#
# Plain bash for a tmux session over ssh (NO SLURM). Runs from the repo root.
#
# Prereq: Stage-1 (KVQ) adapter at ./adapters/granite-obqa.
#         Variant B ALSO needs all 32 MAP router weights in
#         ./router_weights/base/granite_obqa/ ; Variant A does not.
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

# --- Common parameters (repo defaults) ---
MODEL_SHORTCODE="granite"
DATASET_SHORTCODE="obqa"
SEED=42
EPOCHS=10
BATCH_SIZE=4
LEARNING_RATE=1e-5
BETA=0.01

# Susceptible-10 layers, trained jointly in one run.
LAYERS=(5 6 7 8 19 20 28 29 30 31)

# Two variants: parallel arrays indexed together.
PRIOR_SOURCES=("pretrained" "map")
RUN_SUFFIXES=("pretrained-prior" "map-prior")

BASE_ADAPTER_PATH="./adapters/${MODEL_SHORTCODE}-${DATASET_SHORTCODE}"
MAP_WEIGHTS_DIR="./router_weights/base/${MODEL_SHORTCODE}_${DATASET_SHORTCODE}"

# --- Prereq: Stage-1 adapter (needed by both variants) ---
if [ ! -d "$BASE_ADAPTER_PATH" ]; then
    echo "ERROR: Stage-1 adapter not found at $BASE_ADAPTER_PATH" >&2
    exit 1
fi

mkdir -p logs

echo "############################################################"
echo "# FCVR Stage 2 -- training BOTH variants, then evaluating   #"
echo "# Layers: ${LAYERS[*]}"
echo "# epochs=${EPOCHS} batch=${BATCH_SIZE} lr=${LEARNING_RATE} beta=${BETA} seed=${SEED}"
echo "############################################################"

for idx in 0 1; do
    PRIOR="${PRIOR_SOURCES[$idx]}"
    SUFFIX="${RUN_SUFFIXES[$idx]}"

    echo ""
    echo "===================================================="
    echo "TRAIN variant: prior_source=${PRIOR}  suffix=${SUFFIX}"
    echo "Weights -> ./router_weights/fcvr/fcvr-${MODEL_SHORTCODE}-${DATASET_SHORTCODE}-${SUFFIX}/"
    echo "===================================================="

    # Variant B (map prior) needs all 32 MAP router weights.
    if [ "$PRIOR" = "map" ]; then
        for i in $(seq 0 31); do
            if [ ! -f "${MAP_WEIGHTS_DIR}/layer_${i}_weights.pt" ]; then
                echo "ERROR: missing MAP router weights ${MAP_WEIGHTS_DIR}/layer_${i}_weights.pt" >&2
                echo "       Variant B (map-prior) needs all 32 MAP routers. Run Stage 2a first." >&2
                exit 1
            fi
        done
    fi

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
        --seed "$SEED" \
        --run_suffix "$SUFFIX" \
        --prior_source "$PRIOR"
done

echo ""
echo "############################################################"
echo "# Training of both variants complete. Starting evaluation. #"
echo "############################################################"

# Evaluate both variants (fcvr-eval-granite-obqa.sh loops the same two variants).
bash "$REPO_ROOT/fcvr-eval-granite-obqa.sh"

echo ""
echo "############################################################"
echo "# ALL DONE. Two sets of results in ./results/fcvr/         #"
echo "############################################################"

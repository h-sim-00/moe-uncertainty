#!/bin/bash
# ============================================================================
# Stage 2a (MAP router tuning) -- Granite-MoE / MedExQA  (branch codex-recom-iter1)
#
# Supervisor point 3 / prior-source ABLATION. Tunes all 32 deterministic routers
# on top of the existing Stage-1 adapter (everything else frozen) so an FCVR run
# can seed its prior mean from MAP-tuned routers (`--prior_source map
# --map_suffix iter1`) instead of the pre-trained Granite routers.
#
# NOTE: this stage is NOT in the ICML paper's protocol (App. D.2 freezes the
# pre-trained W_r); it is the inherited pipeline's extra step, run here only so
# both priors can be compared. See codex-recom-iter1-notes.md.
#
# Hyperparameters (user decision, 2026-08-17): 3 epochs / batch 4 / lr 1e-4,
# AdamW, best-val checkpoint + early stopping (patience 2), seed 42, prompt-
# masked explanation-only loss (same collator path as kvq-/fcvr-tuning).
#
# Writes ONLY:  ./router_weights/base/granite_medexqa-iter1/layer_{0..31}_weights.pt
# and refuses to run if that directory already exists.
#
# Plain bash for a tmux session over ssh (NO SLURM). Runs from the repo root.
#   source ~/.venvs/moe_env/bin/activate && bash router-tuning-granite-medexqa.sh
# ============================================================================

set -eo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

echo "Repo root: $REPO_ROOT"
echo "Python:    $(which python)"

# --- Parameters ---
MODEL_SHORTCODE="granite"
DATASET_SHORTCODE="medexqa"
BASE_ADAPTER_PATH="./adapters/granite-medexqa"      # existing Stage-1 adapter (read-only input)
MAP_SUFFIX="${MAP_SUFFIX:-iter1}"                    # -> ./router_weights/base/granite_medexqa-iter1
EPOCHS=3
BATCH_SIZE=4
LEARNING_RATE=1e-4
EARLY_STOP_PATIENCE=2
SEED=42

OUT_DIR="./router_weights/base/${MODEL_SHORTCODE}_${DATASET_SHORTCODE}-${MAP_SUFFIX}"

if [ ! -d "$BASE_ADAPTER_PATH" ]; then
    echo "ERROR: Stage-1 adapter $BASE_ADAPTER_PATH not found." >&2
    exit 1
fi
if [ -e "$OUT_DIR" ]; then
    echo "ERROR: $OUT_DIR already exists -- refusing to overwrite MAP router weights." >&2
    echo "       Move it aside or set MAP_SUFFIX=<other>." >&2
    exit 1
fi

mkdir -p logs

echo "############################################################"
echo "# Stage 2a MAP router tuning (prior-source ablation arm)    #"
echo "# adapter=${BASE_ADAPTER_PATH}"
echo "# epochs=${EPOCHS} batch=${BATCH_SIZE} lr=${LEARNING_RATE} patience=${EARLY_STOP_PATIENCE} seed=${SEED}"
echo "# -> ${OUT_DIR}"
echo "############################################################"

python scripts/python/router-tuning.py \
    --model_shortcode "$MODEL_SHORTCODE" \
    --dataset_shortcode "$DATASET_SHORTCODE" \
    --base_adapter_path "$BASE_ADAPTER_PATH" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LEARNING_RATE" \
    --early_stop_patience "$EARLY_STOP_PATIENCE" \
    --seed "$SEED" \
    --map_suffix "$MAP_SUFFIX"

echo ""
echo "MAP routers saved to ${OUT_DIR}"
echo "Next: FCVR with --prior_source map --map_suffix ${MAP_SUFFIX} (see run-iter1-granite-medexqa.sh, phase 'prior')"

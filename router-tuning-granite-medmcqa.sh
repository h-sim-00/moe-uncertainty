#!/bin/bash
# ============================================================================
# Stage 2a (MAP router tuning) -- Granite-MoE / MedMCQA explanations (medmcqa_gen)
# (branch MedMCQA; prior-source ABLATION arm, same role as router-tuning-granite-medexqa.sh)
#
# Tunes all 32 deterministic routers on top of the medmcqa_gen Stage-1 adapter
# (everything else frozen) so an FCVR run can seed its prior mean from MAP-tuned
# routers (`--prior_source map --map_suffix iter1`) instead of the pre-trained
# Granite routers. NOT in the ICML paper's protocol (App. D.2 freezes W_r).
#
# Hyperparameters: 3 epochs (ceiling) / batch 4 / lr 1e-4, AdamW, best-val
# checkpoint + early stopping (validation every EVAL_EVERY=1500 steps + epoch
# end, patience 2 evaluations), seed 42, prompt-masked explanation-only loss.
#
# Writes ONLY:  ./router_weights/base/granite_medmcqa_gen-iter1/layer_{0..31}_weights.pt
# and refuses to run if that directory already exists.
#
#   source ~/.venvs/moe_env/bin/activate && bash router-tuning-granite-medmcqa.sh
# ============================================================================

set -eo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

echo "Repo root: $REPO_ROOT"
echo "Python:    $(which python)"

# --- Parameters ---
MODEL_SHORTCODE="granite"
DATASET_SHORTCODE="medmcqa_gen"
BASE_ADAPTER_PATH="${BASE_ADAPTER_PATH:-./adapters/granite-medmcqa_gen}"   # Stage-1 adapter (read-only input)
MAP_SUFFIX="${MAP_SUFFIX:-iter1}"                    # -> ./router_weights/base/granite_medmcqa_gen-iter1
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-2}"
EVAL_EVERY="${EVAL_EVERY:-1500}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-768}"
SEED="${SEED:-42}"

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
echo "# epochs<=${EPOCHS} batch=${BATCH_SIZE} lr=${LEARNING_RATE} eval_every=${EVAL_EVERY} patience=${EARLY_STOP_PATIENCE} seed=${SEED}"
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
    --eval_every "$EVAL_EVERY" \
    --max_seq_len "$MAX_SEQ_LEN" \
    --seed "$SEED" \
    --map_suffix "$MAP_SUFFIX"

echo ""
echo "MAP routers saved to ${OUT_DIR}"
echo "Next: FCVR with --prior_source map --map_suffix ${MAP_SUFFIX} (see run-iter1-granite-medmcqa.sh, phase 'prior')"

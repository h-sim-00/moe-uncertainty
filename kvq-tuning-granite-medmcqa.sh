#!/bin/bash
# ============================================================================
# Stage-1 (KVQ) LoRA fine-tune -- Granite-MoE / MedMCQA explanations (medmcqa_gen),
# OPEN GENERATION.   (branch MedMCQA; replaces the MedExQA Stage-1 as the input to
# every later arm)
#
# Data (utils/data.py, load_exp_dataset("medmcqa_gen"), frozen by
# splits/medmcqa_gen-derived-seed42.csv):
#   train 30,000 rows of the OFFICIAL MedMCQA train split with a usable gold
#         explanation (`exp`, 10..160 words, single-answer, 4 options), stratified
#         by subject_name;
#   val    1,000 more such rows (stratified, disjoint) = early-stopping / selection set;
#   test   1,000 rows of the OFFICIAL validation split (stratified) = frozen test set.
# Target = " " + gold explanation (+EOS); loss on explanation tokens only
# (prompt-masked Seq2Seq collator), same recipe as the MedExQA generation run.
#
# Output adapter -> ./adapters/granite-medmcqa_gen  (consumed by Stage-2 FCVR).
# Refuses to run if that directory already exists (never overwrite weights).
#
# Hyperparameters (user brief 2026-08-17: 2-3 Stage-1 epochs). lr / batch are
# carried over from kvq-tuning-granite-medexqa.sh -- CONFIRM before launching:
#   EPOCHS=3, BATCH_SIZE=4, LR=5e-5, warmup 0.05 (cosine), Q/K/V LoRA only
#   (--finetune_mode qkv, as the MedExQA adapter; set FINETUNE_MODE=qkv_experts
#   for the paper's expert-LoRA variant, needs more memory), seed 42.
#   30k rows / bs 4 = 7,500 steps per epoch -> validation every EVAL_EVERY=1500
#   steps AND at every epoch end; early stopping after PATIENCE=3 evaluations
#   without val-loss improvement; the adapter on disk is always the best checkpoint.
#   MAX_SEQ_LEN=768: rows longer than that (prompt+explanation tokens) are DROPPED
#   (never truncated) -- run `python medmcqa-gen-inspect.py` first to see how many
#   (expected: ~0%).
#
# Plain bash for a tmux session over ssh (NO SLURM). Runs from the repo root.
#   source ~/.venvs/moe_env/bin/activate && bash kvq-tuning-granite-medmcqa.sh
# ============================================================================

set -eo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

MODEL_SHORTCODE="granite"
DATASET_SHORTCODE="medmcqa_gen"
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
SEED="${SEED:-42}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-3}"   # evaluations (mid-epoch + epoch-end)
EVAL_EVERY="${EVAL_EVERY:-1500}"                  # optimizer steps; 0 = epoch end only
MAX_SEQ_LEN="${MAX_SEQ_LEN:-768}"
FINETUNE_MODE="${FINETUNE_MODE:-qkv}"
ADAPTER_SUFFIX="${ADAPTER_SUFFIX:-}"              # optional; -> ./adapters/granite-medmcqa_gen-<suffix>

OUT_DIR="./adapters/${MODEL_SHORTCODE}-${DATASET_SHORTCODE}${ADAPTER_SUFFIX:+-$ADAPTER_SUFFIX}"

echo "Repo root: $REPO_ROOT"
echo "Python:    $(which python)"
mkdir -p logs

if [ -e "$OUT_DIR" ]; then
    echo "ERROR: $OUT_DIR already exists -- refusing to overwrite a Stage-1 adapter." >&2
    echo "       Move it aside or set ADAPTER_SUFFIX=<other>." >&2
    exit 1
fi

echo "############################################################"
echo "# Stage-1 KVQ (generation) -- granite / medmcqa_gen"
echo "# epochs<=${EPOCHS} batch=${BATCH_SIZE} lr=${LEARNING_RATE} warmup=${WARMUP_RATIO} seed=${SEED}"
echo "# eval every ${EVAL_EVERY} steps + epoch end, patience ${EARLY_STOP_PATIENCE} evals; max_seq_len=${MAX_SEQ_LEN}"
echo "# finetune_mode=${FINETUNE_MODE}  -> ${OUT_DIR} (best val-loss checkpoint)"
echo "############################################################"

EXTRA=()
[ -n "$ADAPTER_SUFFIX" ] && EXTRA+=(--adapter_suffix "$ADAPTER_SUFFIX")

python scripts/python/kvq-tuning.py \
    --model_shortcode "$MODEL_SHORTCODE" \
    --dataset_shortcode "$DATASET_SHORTCODE" \
    --finetune_mode "$FINETUNE_MODE" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LEARNING_RATE" \
    --warmup_ratio "$WARMUP_RATIO" \
    --seed "$SEED" \
    --early_stop_patience "$EARLY_STOP_PATIENCE" \
    --eval_every "$EVAL_EVERY" \
    --max_seq_len "$MAX_SEQ_LEN" \
    "${EXTRA[@]}"

echo ""
echo "Stage-1 adapter saved to ${OUT_DIR}"
echo "Next: bash fcvr-tuning-granite-medmcqa.sh   (or PHASES=stage2 bash run-iter1-granite-medmcqa.sh)"

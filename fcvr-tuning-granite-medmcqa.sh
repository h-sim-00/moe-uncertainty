#!/bin/bash
# ============================================================================
# FCVR (VGLR-FC) Stage 2 -- Granite-MoE / MedMCQA explanations (medmcqa_gen),
# OPEN TEXT GENERATION.   (branch MedMCQA)
#
# Trains the Susceptible-10 layers {5-8, 19-20, 28-31} JOINTLY with the
# pre-trained-router prior (paper D.2: Wr frozen pre-trained; FCVR mean_base
# seeds from it). Sweeps the KL weight beta over {0.01, 0.1}; pick by val NLL.
# Target = free-text MedMCQA gold explanation, prompt-masked (Seq2Seq) loss.
#
# Hyperparameters (user brief 2026-08-17: 3-5 FCVR epochs with early stopping):
#   EPOCHS=5 (ceiling), per-device batch 4 x grad-accum 4 = eff batch 16 (paper),
#   AdamW lr 1e-4 + cosine warmup 0.05 (paper D.2), S=1 train sample, seed 42.
#   30k rows / (4 x 4) = 1,875 optimizer steps per epoch -> validation every
#   EVAL_EVERY=500 optimizer steps AND at every epoch end; early stopping after
#   PATIENCE=3 evaluations without val-NLL improvement (best checkpoint kept).
#   KL mask = attention (real tokens only; the corrected default -- the MedExQA
#   weights used the legacy 'none'; 'none'/'answer' are ablation arms in the driver).
#   MAX_SEQ_LEN=768 drops (never truncates) over-long rows.
#
# Nothing overwrites anything: each beta writes to its own weights dir
#   ./router_weights/fcvr/fcvr-granite-medmcqa_gen-pretrained-prior-beta<b>/
# via --run_suffix (refuses to run if it exists). Distinct from every MedExQA / OBQA run.
#
# Prereq: Stage-1 adapter at ./adapters/granite-medmcqa_gen (kvq-tuning-granite-medmcqa.sh).
# Plain bash for a tmux session over ssh (NO SLURM). Runs from the repo root.
#   source ~/.venvs/moe_env/bin/activate && bash fcvr-tuning-granite-medmcqa.sh
# ============================================================================

set -eo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

echo "Repo root: $REPO_ROOT"
echo "Python:    $(which python)"

# --- Common parameters ---
MODEL_SHORTCODE="granite"
DATASET_SHORTCODE="medmcqa_gen"
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"             # 4 x 4 = effective batch 16
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-3}"   # evaluations
EVAL_EVERY="${EVAL_EVERY:-500}"                   # optimizer steps; 0 = epoch end only
MAX_SEQ_LEN="${MAX_SEQ_LEN:-768}"
PRIOR_SOURCE="pretrained"
KL_MASK="${KL_MASK:-attention}"

# Susceptible-10 layers, trained jointly in one run.
LAYERS=(5 6 7 8 19 20 28 29 30 31)

# VGLR beta grid.
BETAS=(${BETAS:-0.01 0.1})

BASE_ADAPTER_PATH="${BASE_ADAPTER_PATH:-./adapters/${MODEL_SHORTCODE}-${DATASET_SHORTCODE}}"

if [ ! -d "$BASE_ADAPTER_PATH" ]; then
    echo "ERROR: Stage-1 adapter not found at $BASE_ADAPTER_PATH" >&2
    echo "       Run: bash kvq-tuning-granite-medmcqa.sh" >&2
    exit 1
fi
for BETA in "${BETAS[@]}"; do
    W="./router_weights/fcvr/fcvr-${MODEL_SHORTCODE}-${DATASET_SHORTCODE}-pretrained-prior-beta${BETA}"
    if [ -e "$W" ]; then
        echo "ERROR: $W already exists -- refusing to overwrite FCVR weights (move it aside or change BETAS)." >&2
        exit 1
    fi
done

mkdir -p logs

echo "############################################################"
echo "# FCVR Stage 2 (pretrained-prior) GENERATION -- beta sweep {${BETAS[*]}}"
echo "# Layers: ${LAYERS[*]}"
echo "# AdamW lr=${LEARNING_RATE} cosine warmup=${WARMUP_RATIO} | eff batch $((BATCH_SIZE*GRAD_ACCUM))"
echo "# epochs<=${EPOCHS} eval every ${EVAL_EVERY} optim steps + epoch end, patience=${EARLY_STOP_PATIENCE} evals"
echo "# kl_mask=${KL_MASK} max_seq_len=${MAX_SEQ_LEN} seed=${SEED}"
echo "############################################################"

for BETA in "${BETAS[@]}"; do
    SUFFIX="pretrained-prior-beta${BETA}"

    echo ""
    echo "===================================================="
    echo "TRAIN beta=${BETA}  suffix=${SUFFIX}"
    echo "Weights -> ./router_weights/fcvr/fcvr-${MODEL_SHORTCODE}-${DATASET_SHORTCODE}-${SUFFIX}/"
    echo "===================================================="

    python scripts/python/fcvr-tuning.py \
        --model_shortcode "$MODEL_SHORTCODE" \
        --dataset_shortcode "$DATASET_SHORTCODE" \
        --base_adapter_path "$BASE_ADAPTER_PATH" \
        --swap_layers "${LAYERS[@]}" \
        --load_layers \
        --train_layers "${LAYERS[@]}" \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --grad_accum_steps "$GRAD_ACCUM" \
        --lr "$LEARNING_RATE" \
        --warmup_ratio "$WARMUP_RATIO" \
        --early_stop_patience "$EARLY_STOP_PATIENCE" \
        --eval_every "$EVAL_EVERY" \
        --max_seq_len "$MAX_SEQ_LEN" \
        --beta "$BETA" \
        --seed "$SEED" \
        --run_suffix "$SUFFIX" \
        --prior_source "$PRIOR_SOURCE" \
        --kl_mask "$KL_MASK"
done

echo ""
echo "############################################################"
echo "# Training of all beta runs complete."
echo "# Next: PHASES=val bash run-iter1-granite-medmcqa.sh   (readouts + labels + selection)"
echo "############################################################"

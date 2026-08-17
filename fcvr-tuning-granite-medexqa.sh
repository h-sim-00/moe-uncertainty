#!/bin/bash
# ============================================================================
# FCVR (VGLR-FC) Stage 2 -- Granite-MoE / MedExQA, OPEN TEXT GENERATION.
#
# Trains the Susceptible-10 layers {5-8, 19-20, 28-31} JOINTLY with the
# pre-trained-router prior (paper D.2: Wr frozen pre-trained; FCVR mean_base
# seeds from it). Sweeps the KL weight beta over {0.01, 0.1}; pick by val NLL.
#
# UNLIKE the OBQA run, the target is a FREE-TEXT explanation (MedExQA
# 'Explanation 1'), routed through generation_prompt_engineer with a prompt-
# masked (Seq2Seq) loss. Purpose: measure how the Inf-Logit-Var signal behaves
# per-token on generation after the router heads are trained on generation.
#
# Small-data hyperparameters (MedExQA ~740 train): 4 epochs, early-stop
# patience 2; otherwise paper D.2 (AdamW lr 1e-4 + cosine warmup 0.05, per-device
# batch 4 x grad-accum 4 = eff batch 16, S=1 train sample).
#
# Nothing overwrites anything: each beta writes to its own weights dir
#   ./router_weights/fcvr/fcvr-granite-medexqa-pretrained-prior-beta<b>/
# via --run_suffix. Distinct from every OBQA run.
#
# Prereq: Stage-1 (KVQ) generation adapter at ./adapters/granite-medexqa
#         (run kvq-tuning-granite-medexqa.sh first).
# Plain bash for a tmux session over ssh (NO SLURM). Runs from the repo root.
# ============================================================================

set -eo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate moe_env

echo "Repo root: $REPO_ROOT"
echo "Python:    $(which python)"

# --- Common parameters ---
MODEL_SHORTCODE="granite"
DATASET_SHORTCODE="medexqa"
SEED=42
EPOCHS=4                 # small dataset
BATCH_SIZE=4
GRAD_ACCUM=4             # 4 x 4 = effective batch 16
LEARNING_RATE=1e-4
WARMUP_RATIO=0.05
EARLY_STOP_PATIENCE=2    # small dataset
PRIOR_SOURCE="pretrained"

# Susceptible-10 layers, trained jointly in one run.
LAYERS=(5 6 7 8 19 20 28 29 30 31)

# VGLR beta grid.
BETAS=(0.01 0.1)

BASE_ADAPTER_PATH="./adapters/${MODEL_SHORTCODE}-${DATASET_SHORTCODE}"

if [ ! -d "$BASE_ADAPTER_PATH" ]; then
    echo "ERROR: Stage-1 adapter not found at $BASE_ADAPTER_PATH" >&2
    echo "       Run: bash kvq-tuning-granite-medexqa.sh" >&2
    exit 1
fi

mkdir -p logs

echo "############################################################"
echo "# FCVR Stage 2 (pretrained-prior) GENERATION -- beta sweep {${BETAS[*]}}"
echo "# Layers: ${LAYERS[*]}"
echo "# AdamW lr=${LEARNING_RATE} cosine warmup=${WARMUP_RATIO} | eff batch $((BATCH_SIZE*GRAD_ACCUM))"
echo "# epochs<=${EPOCHS} early-stop patience=${EARLY_STOP_PATIENCE} seed=${SEED}"
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
        --beta "$BETA" \
        --seed "$SEED" \
        --run_suffix "$SUFFIX" \
        --prior_source "$PRIOR_SOURCE"
done

echo ""
echo "############################################################"
echo "# Training of both beta runs complete. Starting signal eval.#"
echo "############################################################"

bash "$REPO_ROOT/fcvr-eval-granite-medexqa.sh"

echo ""
echo "############################################################"
echo "# ALL DONE. Signal results in ./results/token_analysis/     #"
echo "############################################################"

#!/bin/bash
# ============================================================================
# FCVR (VGLR-FC) Stage 2 -- Granite-MoE / OBQA, ANSWER-ONLY LOSS
# (branch exp4-train-ans), pretrained prior, beta sweep {0.01, 0.1}.
#
# Identical protocol to fcvr-tuning-granite-obqa.sh (Susceptible-10 layers,
# <=10 epochs with early stopping on val NLL, batch 4 x grad-accum 4 = eff 16,
# AdamW lr 1e-4 cosine warmup 0.05, seed 42, prior_source=pretrained). The
# ONLY changed variables vs the 14-Aug run are:
#   * training/val loss is answer-only (prompt masked to -100), so the ELBO
#     reconstruction term is log p(answer | prompt) as the paper states
#   * the Stage-1 base adapter is the answer-only one (granite-obqa-ansmask)
#
# Weights -> ./router_weights/fcvr/fcvr-granite-obqa-ansmask-pretrained-prior-beta<b>/
# Nothing overwrites previous runs. Chains into fcvr-eval-granite-obqa-ansmask.sh.
#
# Plain bash for a tmux session over ssh (NO SLURM). Runs from the repo root.
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

# --- Common parameters (paper D.2; identical to the 14-Aug run) ---
MODEL_SHORTCODE="granite"
DATASET_SHORTCODE="obqa"
SEED=42
EPOCHS=10
BATCH_SIZE=4
GRAD_ACCUM=4          # 4 x 4 = effective batch 16
LEARNING_RATE=1e-4
WARMUP_RATIO=0.05
EARLY_STOP_PATIENCE=3
PRIOR_SOURCE="pretrained"
ADAPTER_SUFFIX="ansmask"   # Stage-1 answer-only adapter (Q/K/V + experts)

# Susceptible-10 layers, trained jointly in one run.
LAYERS=(5 6 7 8 19 20 28 29 30 31)

# Paper's VGLR beta grid.
BETAS=(0.01 0.1)

BASE_ADAPTER_PATH="./adapters/${MODEL_SHORTCODE}-${DATASET_SHORTCODE}-${ADAPTER_SUFFIX}"

# --- Prereq: Stage-1 answer-only adapter (Q/K/V + experts) ---
if [ ! -d "$BASE_ADAPTER_PATH" ]; then
    echo "ERROR: Stage-1 adapter not found at $BASE_ADAPTER_PATH" >&2
    echo "       Run: bash kvq-tuning-granite-obqa-ansmask.sh" >&2
    exit 1
fi
if [ ! -f "$BASE_ADAPTER_PATH/expert_lora.pt" ]; then
    echo "ERROR: $BASE_ADAPTER_PATH has no expert_lora.pt -- that adapter was" >&2
    echo "       trained WITHOUT the expert networks. Re-run Stage 1 with" >&2
    echo "       --finetune_mode qkv_experts." >&2
    exit 1
fi

mkdir -p logs

echo "############################################################"
echo "# FCVR Stage 2, ANSWER-ONLY LOSS -- beta sweep {${BETAS[*]}}   #"
echo "# Layers: ${LAYERS[*]}"
echo "# AdamW lr=${LEARNING_RATE} cosine warmup=${WARMUP_RATIO} | eff batch $((BATCH_SIZE*GRAD_ACCUM))"
echo "# epochs<=${EPOCHS} early-stop patience=${EARLY_STOP_PATIENCE} seed=${SEED}"
echo "############################################################"

for BETA in "${BETAS[@]}"; do
    SUFFIX="${ADAPTER_SUFFIX}-pretrained-prior-beta${BETA}"

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
echo "# Training of both beta runs complete. Starting evaluation.#"
echo "############################################################"

bash "$REPO_ROOT/fcvr-eval-granite-obqa-ansmask.sh"

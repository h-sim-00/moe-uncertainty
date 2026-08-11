#!/bin/bash
# ============================================================================
# Stage 1 (MAP adaptation) -- Granite-MoE / OBQA, PAPER-FAITHFUL TARGET MODULES.
#
# Paper App. D.2, Stage 1: "We first fine-tune the base model using Low-Rank
# Adaptation (LoRA) [...] LoRA adapters are applied to the attention modules
# (Q/K/V projections) and the Expert networks. This stage runs for 3 epochs."
#
# The exp-3 Stage-1 run adapted Q/K/V ONLY. This run adds the expert networks
# (--finetune_mode qkv_experts) and changes nothing else: same 3 epochs, batch
# 8, lr 1e-4, seed 42 as the exp-3 granite/obqa Stage-1 run, so the expert
# adapters are the only variable.
#
# Nothing overwrites anything: this writes to
#   ./adapters/granite-obqa-experts/          (adapter_model.safetensors + expert_lora.pt)
# leaving exp-3's ./adapters/granite-obqa/ untouched.
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

# --- Parameters ---
MODEL_SHORTCODE="granite"
DATASET_SHORTCODE="obqa"
FINETUNE_MODE="qkv_experts"   # paper D.2: Q/K/V projections AND expert networks
ADAPTER_SUFFIX="experts"      # -> ./adapters/granite-obqa-experts
EXPERT_LORA_R=64              # rank for the per-expert adapters (see note below)
EPOCHS=3                      # paper D.2: Stage 1 runs for 3 epochs
BATCH_SIZE=8
LEARNING_RATE=1e-4
SEED=42

OUT_ADAPTER="./adapters/${MODEL_SHORTCODE}-${DATASET_SHORTCODE}-${ADAPTER_SUFFIX}"

if [ -d "$OUT_ADAPTER" ]; then
    echo "ERROR: $OUT_ADAPTER already exists -- refusing to overwrite a trained adapter." >&2
    echo "       Move it aside or change ADAPTER_SUFFIX." >&2
    exit 1
fi

mkdir -p logs

echo "############################################################"
echo "# Stage 1 MAP adaptation (Q/K/V + experts)                  #"
echo "# mode=${FINETUNE_MODE} expert_lora_r=${EXPERT_LORA_R}"
echo "# epochs=${EPOCHS} batch=${BATCH_SIZE} lr=${LEARNING_RATE} seed=${SEED}"
echo "# Adapter -> ${OUT_ADAPTER}"
echo "############################################################"
# NOTE ON MEMORY: expert LoRA at r=64 adds ~377M trainable parameters across the
# 32 MoE layers (40 experts x 2 matrices each). If this OOMs, drop
# EXPERT_LORA_R to 16 (~94M) -- the paper does not specify a rank.

python kvq-tuning.py \
    --model_shortcode "$MODEL_SHORTCODE" \
    --dataset_shortcode "$DATASET_SHORTCODE" \
    --finetune_mode "$FINETUNE_MODE" \
    --expert_lora_r "$EXPERT_LORA_R" \
    --adapter_suffix "$ADAPTER_SUFFIX" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LEARNING_RATE" \
    --seed "$SEED"

echo ""
echo "############################################################"
echo "# Stage 1 complete. Adapter + expert_lora.pt in:            #"
echo "#   ${OUT_ADAPTER}"
echo "# Next: bash fcvr-tuning-granite-obqa.sh                    #"
echo "############################################################"

#!/bin/bash
# ============================================================================
# Stage 1 (MAP adaptation) -- Granite-MoE / OBQA, Q/K/V + EXPERTS,
# ANSWER-ONLY LOSS (branch exp4-train-ans).
#
# Identical protocol to kvq-tuning-granite-obqa-experts-granite-14Aug.sh
# (AdamW, cosine LR, warmup 0.05, best-val checkpointing, 3 epochs, batch 8,
# lr 1e-4, seed 42, expert LoRA r=64). The ONLY changed variable is the
# training loss: labels now mask the prompt (loss on the answer letter only),
# via preprocess_answer_only_for_training + DataCollatorForSeq2Seq, instead of
# DataCollatorForLanguageModeling silently training on the full sequence.
#
# Run check-answer-only-labels.py first to verify the label masking.
#
# Nothing overwrites anything: this writes to
#   ./adapters/granite-obqa-ansmask/   (adapter_model.safetensors + expert_lora.pt)
# leaving all previous adapters untouched.
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

# --- Parameters (identical to the 14-Aug run except the loss) ---
MODEL_SHORTCODE="granite"
DATASET_SHORTCODE="obqa"
FINETUNE_MODE="qkv_experts"      # paper D.2: Q/K/V projections AND expert networks
ADAPTER_SUFFIX="ansmask"         # -> ./adapters/granite-obqa-ansmask
EXPERT_LORA_R=64
EPOCHS=3                         # paper D.2: Stage 1 runs for 3 epochs
BATCH_SIZE=8
LEARNING_RATE=1e-4
WARMUP_RATIO=0.05
SEED=42

OUT_ADAPTER="./adapters/${MODEL_SHORTCODE}-${DATASET_SHORTCODE}-${ADAPTER_SUFFIX}"

if [ -d "$OUT_ADAPTER" ]; then
    echo "ERROR: $OUT_ADAPTER already exists -- refusing to overwrite a trained adapter." >&2
    echo "       Move it aside or change ADAPTER_SUFFIX." >&2
    exit 1
fi

mkdir -p logs

echo "############################################################"
echo "# Stage 1 MAP adaptation, ANSWER-ONLY LOSS                  #"
echo "# mode=${FINETUNE_MODE} expert_lora_r=${EXPERT_LORA_R}"
echo "# epochs=${EPOCHS} batch=${BATCH_SIZE} lr=${LEARNING_RATE} warmup=${WARMUP_RATIO} seed=${SEED}"
echo "# Adapter -> ${OUT_ADAPTER}"
echo "############################################################"
echo "# NOTE: train/val loss is now per-answer-token NLL -- expect a"
echo "# different (higher) loss scale than the prompt-dominated runs."

python kvq-tuning.py \
    --model_shortcode "$MODEL_SHORTCODE" \
    --dataset_shortcode "$DATASET_SHORTCODE" \
    --finetune_mode "$FINETUNE_MODE" \
    --expert_lora_r "$EXPERT_LORA_R" \
    --adapter_suffix "$ADAPTER_SUFFIX" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LEARNING_RATE" \
    --warmup_ratio "$WARMUP_RATIO" \
    --seed "$SEED"

echo ""
echo "############################################################"
echo "# Stage 1 complete. Adapter + expert_lora.pt in:            #"
echo "#   ${OUT_ADAPTER}"
echo "# (best-val checkpoint on answer-only val NLL)              #"
echo "############################################################"

#!/bin/bash
# ============================================================================
# Stage-1 (KVQ) LoRA fine-tune -- Granite-MoE / MedExQA, OPEN GENERATION.
#
# MedExQA is a benchmark (965 rows, no native train split). The loader pools all
# specialties/splits, shuffles (seed 42), and carves ~740 train / 50 val / 175
# test. The generation target is the free-text 'Explanation 1'; the model is
# trained with a prompt-masked loss (Seq2Seq collator) so only explanation
# tokens contribute to the loss.
#
# Output adapter -> ./adapters/granite-medexqa  (consumed by Stage-2 FCVR).
#
# Small-data hyperparameters (see fcvr run for the FCVR-stage grid). Edit freely.
# Plain bash for a tmux session over ssh (NO SLURM). Runs from the repo root.
# ============================================================================

set -eo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate moe_env

MODEL_SHORTCODE="granite"
DATASET_SHORTCODE="medexqa"
EPOCHS=4                 # small dataset -> few epochs
BATCH_SIZE=4             # explanations are multi-token; keep the micro-batch modest
LEARNING_RATE=5e-5
SEED=42

echo "Repo root: $REPO_ROOT"
echo "Python:    $(which python)"
mkdir -p logs

echo "############################################################"
echo "# Stage-1 KVQ (generation) -- granite / medexqa"
echo "# epochs=${EPOCHS} batch=${BATCH_SIZE} lr=${LEARNING_RATE} seed=${SEED}"
echo "# -> ./adapters/${MODEL_SHORTCODE}-${DATASET_SHORTCODE}"
echo "############################################################"

python scripts/python/kvq-tuning.py \
    --model_shortcode "$MODEL_SHORTCODE" \
    --dataset_shortcode "$DATASET_SHORTCODE" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LEARNING_RATE" \
    --seed "$SEED"

echo ""
echo "Stage-1 adapter saved to ./adapters/${MODEL_SHORTCODE}-${DATASET_SHORTCODE}"
echo "Next: bash fcvr-tuning-granite-medexqa.sh"

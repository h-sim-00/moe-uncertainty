#!/bin/bash
# Quail driver for evaluate_exp4_wrong_answer_ood.py.
#
# Question: within each Albus OOD dataset, does the original exp4-train-ans
# final-position ILV rank wrong answer letters above correct answer letters?
#
# Usage on quail-1 (activate moe_env before starting):
#   tmux new -s exp4-wrong-ood
#   bash run-exp4-wrong-answer-ood.sh
#
# Optional smoke:
#   N_PER_DATASET=25 N_BOOT=100 TAG=smoke-exp4-wrong-ood \
#     bash run-exp4-wrong-answer-ood.sh
#
# Optional reproducible MC repeats (data stay fixed):
#   SAMPLING_SEED=43 bash run-exp4-wrong-answer-ood.sh
#
# Existing outputs are protected. Set OVERWRITE=1 only to replace them.
set -Eeo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

DATA_SEED="${DATA_SEED:-42}"
SAMPLING_SEED="${SAMPLING_SEED:-42}"
S="${S:-35}"
BATCH_SIZE="${BATCH_SIZE:-8}"
N_BOOT="${N_BOOT:-2000}"
N_PER_DATASET="${N_PER_DATASET:-0}"
TAG="${TAG:-exp4-wrong-answer-albus-ood}"
DATASETS="${DATASETS:-arc_e arc_c medmcqa_med mmlu_law}"
ADAPTER_PATH="${ADAPTER_PATH:-adapters/granite-obqa-ansmask}"
RUN_SUFFIX="${RUN_SUFFIX:-ansmask-pretrained-prior-beta0.01}"
LAYERS="${LAYERS:-5 6 7 8 19 20 28 29 30 31}"

read -r -a DATASET_ARGS <<< "$DATASETS"
read -r -a LAYER_ARGS <<< "$LAYERS"
EXTRA=()
[ "${INCLUDE_OBQA_ID:-0}" = "1" ] && EXTRA+=(--include_obqa_id)
[ "${OVERWRITE:-0}" = "1" ] && EXTRA+=(--overwrite)

mkdir -p logs results/exp4_wrong_answer_ood
RUN_TAG="$(date +%Y%m%d-%H%M%S)"
LOG="logs/exp4-wrong-answer-ood-${RUN_TAG}.log"

echo "============================================================"
echo "exp4 wrong-answer-on-OOD evaluation"
echo "data_seed=$DATA_SEED sampling_seed=$SAMPLING_SEED S=$S"
echo "datasets=${DATASET_ARGS[*]} n_per_dataset=$N_PER_DATASET n_boot=$N_BOOT"
echo "adapter=$ADAPTER_PATH"
echo "FCVR suffix=$RUN_SUFFIX layers=${LAYER_ARGS[*]}"
echo "log=$LOG"
echo "============================================================"

python evaluate_exp4_wrong_answer_ood.py \
  --datasets "${DATASET_ARGS[@]}" \
  --data_seed "$DATA_SEED" \
  --sampling_seed "$SAMPLING_SEED" \
  --num_samples "$S" \
  --batch_size "$BATCH_SIZE" \
  --n_boot "$N_BOOT" \
  --n_per_dataset "$N_PER_DATASET" \
  --tag "$TAG" \
  --adapter_path "$ADAPTER_PATH" \
  --run_suffix "$RUN_SUFFIX" \
  --swap_layers "${LAYER_ARGS[@]}" \
  "${EXTRA[@]}" 2>&1 | tee "$LOG"

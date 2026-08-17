#!/bin/bash
# ============================================================================
# FCVR signal readout -- Granite-MoE / MedExQA (open generation).
#
# For each trained beta run:
#   (1) teacher-forced over the GOLD explanation   (analyze_token_signals.py)
#   (2) autoregressive: the model generates its own explanation, signal read
#       over the generated tokens, PLUS the sequence-level abstention readout
#       (letter-probe correctness labels, ILV aggregates vs entropy/NLL
#       baselines, AUROC for predicting wrong answers)
#                                                   (analyze_token_signals.py --generate)
#   (3) input-level ID-vs-OoD bridge test: does the paper's Table 8 claim
#       (Inf-Logit-Var separates ID from OoD inputs) survive generation
#       training? If not, (1)/(2) are moot.        (fcvr_input_level_ood_check.py)
# (1)/(2) write ./results/token_analysis/step1_medexqa_<mode>_<tag>.{json,html,png,jsonl}
# (3) writes  ./results/input_level_ood/input_ood_medexqa_<suffix>_<tag>.json
#
# This does NOT run the MCQA OoD/calibration table (evaluate_fcvr.py) -- that
# readout is single-letter-specific and not meaningful for free generation.
#
# Prereq: trained FCVR weights from fcvr-tuning-granite-medexqa.sh and the
#         Stage-1 adapter ./adapters/granite-medexqa.
# ============================================================================

set -eo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate moe_env

MODEL_SHORTCODE="granite"
DATASET_SHORTCODE="medexqa"
KVQ_ADAPTER_PATH="./adapters/${MODEL_SHORTCODE}-${DATASET_SHORTCODE}"
PRIOR_SOURCE="pretrained"
LAYERS=(5 6 7 8 19 20 28 29 30 31)
BETAS=(0.01 0.1)
NUM_EXAMPLES=175          # all of the held-out MedExQA test rows
MAX_NEW_TOKENS=128

if [ ! -d "$KVQ_ADAPTER_PATH" ]; then
    echo "ERROR: Stage-1 adapter not found at $KVQ_ADAPTER_PATH" >&2
    exit 1
fi

mkdir -p results/token_analysis

for BETA in "${BETAS[@]}"; do
    SUFFIX="pretrained-prior-beta${BETA}"
    echo ""
    echo "===================================================="
    echo "SIGNAL READOUT  beta=${BETA}  suffix=${SUFFIX}"
    echo "===================================================="

    # (1) Teacher-forced over the gold explanation
    python analyze_token_signals.py \
        --model_shortcode "$MODEL_SHORTCODE" \
        --dataset_shortcode "$DATASET_SHORTCODE" \
        --kvq_adapter_path "$KVQ_ADAPTER_PATH" \
        --swap_layers "${LAYERS[@]}" \
        --run_suffix "$SUFFIX" \
        --prior_source "$PRIOR_SOURCE" \
        --source medexqa \
        --num_examples "$NUM_EXAMPLES" \
        --routing deterministic --num_samples 1 \
        --split test \
        --tag "beta${BETA}"

    # (2) Autoregressive: read the signal over the model's own generated tokens
    #     + sequence-level abstention readout (letter-probe correctness,
    #     ILV aggregates vs entropy/NLL baselines, AUROC predict-wrong)
    python analyze_token_signals.py \
        --model_shortcode "$MODEL_SHORTCODE" \
        --dataset_shortcode "$DATASET_SHORTCODE" \
        --kvq_adapter_path "$KVQ_ADAPTER_PATH" \
        --swap_layers "${LAYERS[@]}" \
        --run_suffix "$SUFFIX" \
        --prior_source "$PRIOR_SOURCE" \
        --source medexqa \
        --generate \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --num_examples "$NUM_EXAMPLES" \
        --routing deterministic --num_samples 1 \
        --split test \
        --tag "beta${BETA}"

    # (3) Bridge test: input-level ID-vs-OoD separation (paper Table 8 claim)
    #     on the generation-trained FCVR. If this separation is gone, the
    #     token-level readouts above have no support from the paper's mechanism.
    python fcvr_input_level_ood_check.py \
        --model_shortcode "$MODEL_SHORTCODE" \
        --dataset_shortcode "$DATASET_SHORTCODE" \
        --kvq_adapter_path "$KVQ_ADAPTER_PATH" \
        --swap_layers "${LAYERS[@]}" \
        --run_suffix "$SUFFIX" \
        --prior_source "$PRIOR_SOURCE" \
        --ood_datasets obqa mmlu_law \
        --num_examples "$NUM_EXAMPLES" \
        --routing deterministic --num_samples 1 \
        --split test \
        --tag "beta${BETA}"
done

echo ""
echo "Signal readouts saved under ./results/token_analysis/"
echo "  step1_medexqa_teacher_forced_beta<b>.{json,html,png}"
echo "  step1_medexqa_generate_beta<b>.{json,html,png} + _seqlevel.jsonl (abstention readout)"
echo "Input-level bridge test under ./results/input_level_ood/"
echo "  input_ood_medexqa_pretrained-prior-beta<b>_beta<b>.json"
echo "Read the VERDICT line in each JSON, and the 'abstention' block in the generate JSONs."

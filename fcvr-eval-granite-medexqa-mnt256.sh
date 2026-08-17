#!/bin/bash
# ============================================================================
# Rerun ONLY the autoregressive (--generate) signal readout with a larger
# generation cap: max_new_tokens 256 instead of 128.
#
# Why: with the 128 cap, ~40% of generated explanations were truncated
# mid-sentence, contaminating the last-k ILV aggregates and expl_f1 in the
# abstention readout. Teacher-forced, input-level OoD, and all training are
# unaffected by the cap and are NOT rerun here.
#
# Writes NEW files (tag beta<b>-mnt256) alongside the old ones -- nothing from
# the 128-cap run is overwritten:
#   results/token_analysis/step1_medexqa_generate_beta<b>-mnt256.{json,html,png}
#   + _pertoken.jsonl / _seqlevel.jsonl
#
# Prereq: trained FCVR weights (fcvr-tuning-granite-medexqa.sh) and the
#         Stage-1 adapter ./adapters/granite-medexqa.
# Plain bash for a tmux session over ssh (NO SLURM). Runs from the repo root.
# ============================================================================

set -eo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

MODEL_SHORTCODE="granite"
DATASET_SHORTCODE="medexqa"
KVQ_ADAPTER_PATH="./adapters/${MODEL_SHORTCODE}-${DATASET_SHORTCODE}"
PRIOR_SOURCE="pretrained"
LAYERS=(5 6 7 8 19 20 28 29 30 31)
BETAS=(0.01 0.1)
NUM_EXAMPLES=175
MAX_NEW_TOKENS=256

if [ ! -d "$KVQ_ADAPTER_PATH" ]; then
    echo "ERROR: Stage-1 adapter not found at $KVQ_ADAPTER_PATH" >&2
    exit 1
fi

for BETA in "${BETAS[@]}"; do
    SUFFIX="pretrained-prior-beta${BETA}"
    echo ""
    echo "===================================================="
    echo "GENERATE READOUT (max_new_tokens=${MAX_NEW_TOKENS})  beta=${BETA}"
    echo "===================================================="

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
        --split test \
        --tag "beta${BETA}-mnt256"
done

echo ""
echo "Done. New files: results/token_analysis/step1_medexqa_generate_beta<b>-mnt256.*"
echo "Old 128-cap files untouched. Compare the 'verdict' and 'abstention' blocks."

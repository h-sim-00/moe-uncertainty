#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Step 1: does the Granite+FCVR Inf-Logit-Var router signal transfer to
# per-token generation? Runs analyze_token_signals.py on:
#   (1) free-form prose  -> the TRUE "MCQA-signal -> generation" transfer test
#   (2) OBQA MCQA prompts -> in-domain reference (where the signal was trained)
#
# quail-1 / tmux (no SLURM). Logs to logs/. Weights are read from the SHARED
# router_weights/ tree (gitignored, produced on the remote). Edit the CONFIG
# block so RUN_SUFFIX / PRIOR_SOURCE / SWAP_LAYERS match the FCVR run that
# produced your OoD numbers.
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")"

# ===================== CONFIG (edit to match your trained run) ==============
MODEL="granite"
DATASET="obqa"
KVQ_ADAPTER="./adapters/granite-obqa"                 # Stage-1 LoRA adapter
SWAP_LAYERS="5 6 7 8 19 20 28 29 30 31"               # Susceptible-10 (paper)
PRIOR_SOURCE="pretrained"                             # must match training: pretrained | map
RUN_SUFFIX="pretrained-prior-beta0.1"                 # must match training --run_suffix
SPIKE_PCT="0.15"
NUM_EXAMPLES="50"                                     # obqa questions to analyse
# ===========================================================================

STAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p logs results/token_analysis
LOG="logs/step1_token_analysis_${STAMP}.log"

FCVR_DIR="./router_weights/fcvr/fcvr-${MODEL}-${DATASET}-${RUN_SUFFIX}"

echo "=== Step 1 token-signal analysis ===" | tee "$LOG"
echo "FCVR weights dir: ${FCVR_DIR}" | tee -a "$LOG"

# --- preflight -------------------------------------------------------------
if [[ ! -d "$KVQ_ADAPTER" ]]; then
  echo "ERROR: KVQ adapter not found at $KVQ_ADAPTER" | tee -a "$LOG"; exit 1
fi
if [[ ! -d "$FCVR_DIR" ]]; then
  echo "ERROR: FCVR weights dir not found: $FCVR_DIR" | tee -a "$LOG"
  echo "Available FCVR weight dirs (pick the right RUN_SUFFIX):" | tee -a "$LOG"
  ls -1 ./router_weights/fcvr/ 2>/dev/null | tee -a "$LOG" || echo "  (none)" | tee -a "$LOG"
  exit 1
fi
missing=0
for l in $SWAP_LAYERS; do
  if [[ ! -f "${FCVR_DIR}/layer_${l}_weights.pt" ]]; then
    echo "ERROR: missing FCVR weights: ${FCVR_DIR}/layer_${l}_weights.pt" | tee -a "$LOG"; missing=1
  fi
done
[[ $missing -eq 1 ]] && exit 1
if [[ "$PRIOR_SOURCE" == "map" && ! -d "./router_weights/base/${MODEL}_${DATASET}" ]]; then
  echo "ERROR: prior_source=map but MAP dir ./router_weights/base/${MODEL}_${DATASET} missing" | tee -a "$LOG"; exit 1
fi
echo "Preflight OK." | tee -a "$LOG"

run () {  # $1=source  $2=extra flags
  echo "" | tee -a "$LOG"
  echo ">>> source=$1 $2" | tee -a "$LOG"
  python analyze_token_signals.py \
    --model_shortcode "$MODEL" \
    --dataset_shortcode "$DATASET" \
    --kvq_adapter_path "$KVQ_ADAPTER" \
    --swap_layers $SWAP_LAYERS \
    --prior_source "$PRIOR_SOURCE" \
    --run_suffix "$RUN_SUFFIX" \
    --source "$1" \
    --num_examples "$NUM_EXAMPLES" \
    --spike_pct "$SPIKE_PCT" \
    --tag "${RUN_SUFFIX}" \
    $2 2>&1 | tee -a "$LOG"
}

# (1) free-form prose, RAW (no chat template) -> the real generation transfer test
run builtin ""
# (2) OBQA MCQA prompts -> in-domain reference (chat template applied automatically)
run obqa ""

echo "" | tee -a "$LOG"
echo "Done. Results in results/token_analysis/ ; log: $LOG" | tee -a "$LOG"

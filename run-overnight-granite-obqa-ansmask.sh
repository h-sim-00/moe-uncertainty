#!/bin/bash
# ============================================================================
# OVERNIGHT DRIVER -- Granite-MoE / OBQA, ANSWER-ONLY LOSS (branch exp4-train-ans):
# label sanity check + Stage 1 + Stage 2 + eval, in one go.
#
#   Check    check-answer-only-labels.py            loss positions = answer letter only
#   Stage 1  kvq-tuning-granite-obqa-ansmask.sh     LoRA on Q/K/V AND experts
#   Stage 2  fcvr-tuning-granite-obqa-ansmask.sh    FCVR routers, beta in {0.01, 0.1}
#   Eval     (auto-chained by the Stage-2 script)   id_calibration + ood_detection
#
# COLLISION SAFETY: before touching the GPU this refuses to start if ANY output
# path it would write already exists -- adapter dir, per-beta router weight dirs,
# or result JSONs. Nothing from the 14-Aug run (or an earlier attempt at this
# run) can be silently overwritten. Delete or rename the offending path
# deliberately if you actually want to redo a stage.
#
# Usage (in tmux, so an ssh drop does not kill it):
#     tmux new -s ansmask
#     bash run-overnight-granite-obqa-ansmask.sh
#     # detach with Ctrl-b d ; reattach later with: tmux attach -t ansmask
#
# Env knobs:
#     SKIP_PREFLIGHT=1   skip the ~3 min GPU memory/time smoke test
#     WANDB_API_KEY=...  needed for W&B logging + alerts
# ============================================================================

set -Eeo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# --- Optional: activate conda env (uncomment if your tmux shell hasn't) ---
# source "$(conda info --base)/etc/profile.d/conda.sh"
# conda activate moe_env

# --- Config (must match the two stage scripts) ---
MODEL_SHORTCODE="granite"
DATASET_SHORTCODE="obqa"
ADAPTER_SUFFIX="ansmask"
BETAS=(0.01 0.1)
LAYERS=(5 6 7 8 19 20 28 29 30 31)
WANDB_PROJECT="${WANDB_PROJECT:-moe-uncertainty}"

RUN_TAG="$(date +%Y%m%d-%H%M%S)"
mkdir -p logs
LOG="logs/overnight-${MODEL_SHORTCODE}-${DATASET_SHORTCODE}-${ADAPTER_SUFFIX}-${RUN_TAG}.log"

# Everything from here on goes to the terminal AND the log file.
exec > >(tee -a "$LOG") 2>&1

echo "############################################################"
echo "# Overnight run  ${RUN_TAG}  (ANSWER-ONLY LOSS)"
echo "# Repo:   $REPO_ROOT"
echo "# Branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?')"
echo "# Commit: $(git rev-parse --short HEAD 2>/dev/null || echo '?')"
echo "# Python: $(which python)"
echo "# Log:    $LOG"
echo "############################################################"

# ---------------------------------------------------------------------------
# Failure handling: log it, then fire a W&B alert so you hear about it in bed.
# ---------------------------------------------------------------------------
notify() {
    local level="$1" title="$2" text="$3"
    python - "$level" "$title" "$text" <<'PY' || echo "(W&B alert failed -- check WANDB_API_KEY)"
import os, sys
level, title, text = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    import wandb
    run = wandb.init(project=os.environ.get("WANDB_PROJECT", "moe-uncertainty"),
                     name=f"overnight-watchdog-{os.environ.get('RUN_TAG', '')}",
                     job_type="alert", reinit=True)
    wandb.alert(title=title, text=text,
                level=wandb.AlertLevel.ERROR if level == "error" else wandb.AlertLevel.INFO)
    run.finish()
    print(f"W&B alert sent: [{level}] {title}")
except Exception as e:
    print(f"W&B alert failed: {type(e).__name__}: {e}")
PY
}
export RUN_TAG WANDB_PROJECT

on_error() {
    local exit_code=$? line=$1
    echo ""
    echo "############################################################"
    echo "# FAILED at line ${line} (exit ${exit_code}) -- ${STEP:-unknown step}"
    echo "# Last 20 log lines are above; full log: ${LOG}"
    echo "############################################################"
    notify error "Overnight ansmask run FAILED: ${STEP:-unknown step}" \
        "Host $(hostname), run ${RUN_TAG}, exit ${exit_code} at line ${line}. Log: ${LOG}"
    exit "$exit_code"
}
trap 'on_error $LINENO' ERR

# ---------------------------------------------------------------------------
# 0. Collision check -- fail before spending any GPU time.
# ---------------------------------------------------------------------------
STEP="collision check"
echo ""
echo "==== Step 0/4: checking that nothing would be overwritten ===="

LAYER_TAG=$(printf '%s-' "${LAYERS[@]}"); LAYER_TAG=${LAYER_TAG%-}
COLLISIONS=()

ADAPTER_DIR="./adapters/${MODEL_SHORTCODE}-${DATASET_SHORTCODE}-${ADAPTER_SUFFIX}"
[ -e "$ADAPTER_DIR" ] && COLLISIONS+=("$ADAPTER_DIR")

for BETA in "${BETAS[@]}"; do
    SUFFIX="${ADAPTER_SUFFIX}-pretrained-prior-beta${BETA}"
    W="./router_weights/fcvr/fcvr-${MODEL_SHORTCODE}-${DATASET_SHORTCODE}-${SUFFIX}"
    [ -e "$W" ] && COLLISIONS+=("$W")
    for KIND in id_calib ood_detect; do
        R="./results/fcvr/${KIND}_fcvr-${MODEL_SHORTCODE}-${DATASET_SHORTCODE}-${SUFFIX}_layers-${LAYER_TAG}.json"
        [ -e "$R" ] && COLLISIONS+=("$R")
    done
done

if [ ${#COLLISIONS[@]} -gt 0 ]; then
    echo "ERROR: these output paths already exist -- refusing to overwrite:" >&2
    printf '  %s\n' "${COLLISIONS[@]}" >&2
    echo "" >&2
    echo "Move/rename them (or change ADAPTER_SUFFIX here AND in the stage" >&2
    echo "scripts) if you really want to redo this run." >&2
    exit 1
fi
echo "OK: no collisions. Writing to:"
echo "  adapter : $ADAPTER_DIR"
echo "  weights : ./router_weights/fcvr/fcvr-${MODEL_SHORTCODE}-${DATASET_SHORTCODE}-${ADAPTER_SUFFIX}-pretrained-prior-beta<b>/"
echo "  results : ./results/fcvr/*_fcvr-${MODEL_SHORTCODE}-${DATASET_SHORTCODE}-${ADAPTER_SUFFIX}-pretrained-prior-beta<b>_layers-${LAYER_TAG}.json"

# ---------------------------------------------------------------------------
# 1. Label sanity check -- the whole point of this run is the loss masking;
#    fail fast (CPU-only, ~1 min) if the loss positions are not exactly the
#    answer letter.
# ---------------------------------------------------------------------------
STEP="answer-only label check"
echo ""
echo "==== Step 1/4: answer-only label sanity check ===="
python check-answer-only-labels.py

# ---------------------------------------------------------------------------
# 2. Pre-flight: does Stage 1 actually fit, and how slow is it?
#    (Smoke test still uses the old full-sequence loader, which pads to the
#    global-longest sequence -- a conservative upper bound on the new
#    per-batch-padded memory footprint.)
# ---------------------------------------------------------------------------
if [ "${SKIP_PREFLIGHT:-0}" != "1" ]; then
    STEP="pre-flight memory check"
    echo ""
    echo "==== Step 2/4: pre-flight GPU check (set SKIP_PREFLIGHT=1 to skip) ===="
    python expert-lora-memory-check.py \
        --model_shortcode "$MODEL_SHORTCODE" \
        --dataset_shortcode "$DATASET_SHORTCODE" \
        --finetune_mode qkv_experts \
        --expert_lora_r 64 \
        --batch_size 8 \
        --epochs 3
else
    echo ""
    echo "==== Step 2/4: pre-flight SKIPPED (SKIP_PREFLIGHT=1) ===="
fi

# ---------------------------------------------------------------------------
# 3. Stage 1 -- MAP adaptation, answer-only loss.
# ---------------------------------------------------------------------------
STEP="Stage 1 (kvq + experts, answer-only)"
echo ""
echo "==== Step 3/4: Stage 1 -- $(date) ===="
bash "$REPO_ROOT/kvq-tuning-granite-obqa-ansmask.sh"

if [ ! -f "$ADAPTER_DIR/expert_lora.pt" ]; then
    echo "ERROR: Stage 1 finished but $ADAPTER_DIR/expert_lora.pt is missing." >&2
    exit 1
fi
echo "Stage 1 done -- $(date)"

# ---------------------------------------------------------------------------
# 4. Stage 2 (both betas) -- the script auto-chains the evaluation at the end.
# ---------------------------------------------------------------------------
STEP="Stage 2 (FCVR, answer-only) + evaluation"
echo ""
echo "==== Step 4/4: Stage 2 + eval -- $(date) ===="
bash "$REPO_ROOT/fcvr-tuning-granite-obqa-ansmask.sh"

# ---------------------------------------------------------------------------
# Done.
# ---------------------------------------------------------------------------
echo ""
echo "############################################################"
echo "# ALL DONE -- $(date)"
echo "# Results:"
ls -1 ./results/fcvr/*"${ADAPTER_SUFFIX}"-pretrained-prior-beta*.json 2>/dev/null | sed 's/^/#   /' || true
echo "# Full log: ${LOG}"
echo "############################################################"

notify info "Overnight ansmask run finished: ${MODEL_SHORTCODE}/${DATASET_SHORTCODE}" \
    "Host $(hostname), run ${RUN_TAG}. Label check + Stage 1 + Stage 2 (beta ${BETAS[*]}) + eval all completed. Log: ${LOG}"

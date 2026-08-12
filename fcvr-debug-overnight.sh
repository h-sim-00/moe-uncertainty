#!/bin/bash
# ============================================================================
# token-debug overnight: Inf-Logit-Var inversion ablation study.
#
# Phase 0 (no training, existing exp-3 checkpoints read-only):
#   - full diagnostic OoD eval (per-layer AUROC, diag/offdiag/log_det
#     decomposition, familiarity gradient train>test>OoD) on the exp-3
#     beta=0.01 and beta=0.1 checkpoints
#   - untrained control: same eval with never-trained (seed-42 random init)
#     FCVR heads -- if it is just as inverted, the signal was never learned
#
# Training arms (exp-3 baseline; ONE change each; train -> id_calib -> ood):
#   1. debug-beta0              KL off
#   2. debug-beta0.001          KL 10x weaker
#   3. debug-2trunk-beta0.01    separate Cholesky trunk (thesis' original arch)
#   4. debug-priorstd2-beta0.01 prior N(l_det, 4I): trace attractor 40 -> 160
#   5. debug-ln-beta0.01        LayerNorm on the variational trunk input
#   6. debug-long-beta0.01      30 epochs, patience 10 (longest arm -> last)
#
# Isolation: every arm suffix starts "debug-" -> weights in
#   ./router_weights/fcvr/fcvr-granite-obqa-debug-*/   (disjoint from exp-3)
# and ALL results go under ./results-debug/ (never ./results/).
#
# Deliberately NOT `set -e`: an unattended overnight run logs a failed step
# and moves on instead of dying. Failures are summarised at the end.
#
# Plain bash for a tmux session over ssh (NO SLURM). Runs from the repo root.
# Prereq: Stage-1 KVQ adapter at ./adapters/granite-obqa and (for Phase 0)
#         the exp-3 FCVR checkpoints fcvr-granite-obqa-pretrained-prior-beta*.
# ============================================================================

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# --- Optional: activate conda env (uncomment if your tmux shell hasn't) ---
# source "$(conda info --base)/etc/profile.d/conda.sh"
# conda activate moe_env

echo "Repo root: $REPO_ROOT"
echo "Python:    $(which python)"

# --- Common parameters (exp-3 baseline, paper D.2) ---
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
NUM_SAMPLES=35
EVAL_BATCH_SIZE=8

# Susceptible-10 layers, trained jointly.
LAYERS=(5 6 7 8 19 20 28 29 30 31)
LAYER_TAG=$(printf '%s-' "${LAYERS[@]}"); LAYER_TAG=${LAYER_TAG%-}

BASE_ADAPTER_PATH="./adapters/${MODEL_SHORTCODE}-${DATASET_SHORTCODE}"
WANDB_PROJECT="moe-uncertainty-debug"
RD="./results-debug"

# --- Preflight ---
if [ ! -d "$BASE_ADAPTER_PATH" ]; then
    echo "ERROR: Stage-1 adapter not found at $BASE_ADAPTER_PATH" >&2
    exit 1
fi
if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "WARNING: WANDB_API_KEY is not set -- wandb logging may fail/prompt." >&2
fi

mkdir -p "$RD/phase0" "$RD/arms" "$RD/logs"
FAIL_LOG="$RD/failures.txt"
: > "$FAIL_LOG"

# run_step <tag> <cmd...> : tee output to a per-step log, record failures.
run_step() {
    local tag="$1"; shift
    echo ""
    echo "=== [$(date '+%F %T')] $tag ==="
    "$@" 2>&1 | tee "$RD/logs/${tag}.log"
    local rc=${PIPESTATUS[0]}
    if [ "$rc" -ne 0 ]; then
        echo "$(date '+%F %T')  $tag  rc=$rc" >> "$FAIL_LOG"
        echo "!!! $tag FAILED (rc=$rc) -- continuing" >&2
    fi
    return "$rc"
}

# eval_ood <suffix> <outdir> <wandb_tag> [extra eval args...]
eval_ood() {
    local SFX="$1" OUT="$2" TAG="$3"; shift 3
    run_step "eval-ood-${SFX}" python evaluate_fcvr.py \
        --task ood_detection \
        --model_shortcode "$MODEL_SHORTCODE" \
        --dataset_shortcode "$DATASET_SHORTCODE" \
        --kvq_adapter_path "$BASE_ADAPTER_PATH" \
        --swap_layers "${LAYERS[@]}" \
        --run_suffix "$SFX" \
        --prior_source "$PRIOR_SOURCE" \
        --output_json_path "${OUT}/ood_detect_fcvr-${MODEL_SHORTCODE}-${DATASET_SHORTCODE}-${SFX}_layers-${LAYER_TAG}.json" \
        --per_example_jsonl_path "${OUT}/per_example_fcvr-${MODEL_SHORTCODE}-${DATASET_SHORTCODE}-${SFX}.jsonl" \
        --num_samples "$NUM_SAMPLES" \
        --batch_size "$EVAL_BATCH_SIZE" \
        --seed "$SEED" \
        --wandb_project "$WANDB_PROJECT" \
        --wandb_tags "$TAG" eval \
        "$@"
}

# eval_id <suffix> <outdir> <wandb_tag> [extra eval args...]
eval_id() {
    local SFX="$1" OUT="$2" TAG="$3"; shift 3
    run_step "eval-id-${SFX}" python evaluate_fcvr.py \
        --task id_calibration \
        --model_shortcode "$MODEL_SHORTCODE" \
        --dataset_shortcode "$DATASET_SHORTCODE" \
        --kvq_adapter_path "$BASE_ADAPTER_PATH" \
        --swap_layers "${LAYERS[@]}" \
        --run_suffix "$SFX" \
        --prior_source "$PRIOR_SOURCE" \
        --output_json_path "${OUT}/id_calib_fcvr-${MODEL_SHORTCODE}-${DATASET_SHORTCODE}-${SFX}_layers-${LAYER_TAG}.json" \
        --num_samples "$NUM_SAMPLES" \
        --batch_size "$EVAL_BATCH_SIZE" \
        --seed "$SEED" \
        --wandb_project "$WANDB_PROJECT" \
        --wandb_tags "$TAG" eval \
        "$@"
}

echo "############################################################"
echo "# PHASE 0: diagnostics on existing exp-3 checkpoints        #"
echo "############################################################"

eval_ood "pretrained-prior-beta0.01" "$RD/phase0" phase0 || true
eval_ood "pretrained-prior-beta0.1"  "$RD/phase0" phase0 || true
# Untrained control: heads at seed-42 random init, nothing loaded. The suffix
# only names the output files/wandb run -- no weights are read or written.
eval_ood "untrained-init-seed42" "$RD/phase0" phase0 --untrained_control || true

echo ""
echo "############################################################"
echo "# TRAINING ARMS (exp-3 baseline, one change each)           #"
echo "############################################################"

COMMON_TRAIN=(
    --model_shortcode "$MODEL_SHORTCODE"
    --dataset_shortcode "$DATASET_SHORTCODE"
    --base_adapter_path "$BASE_ADAPTER_PATH"
    --swap_layers "${LAYERS[@]}"
    --load_layers
    --train_layers "${LAYERS[@]}"
    --epochs "$EPOCHS"
    --batch_size "$BATCH_SIZE"
    --grad_accum_steps "$GRAD_ACCUM"
    --lr "$LEARNING_RATE"
    --warmup_ratio "$WARMUP_RATIO"
    --early_stop_patience "$EARLY_STOP_PATIENCE"
    --seed "$SEED"
    --prior_source "$PRIOR_SOURCE"
    --wandb_project "$WANDB_PROJECT"
)

# Arm table. TRAIN_DELTA is appended AFTER the common args, so argparse
# last-wins lets arm 6 override --epochs/--early_stop_patience cleanly.
# EVAL_DELTA carries the flags eval must mirror to rebuild the same router.
ARM_SUFFIX=( "debug-beta0"
             "debug-beta0.001"
             "debug-2trunk-beta0.01"
             "debug-priorstd2-beta0.01"
             "debug-ln-beta0.01"
             "debug-long-beta0.01" )
ARM_TRAIN_DELTA=( "--beta 0.0"
                  "--beta 0.001"
                  "--beta 0.01 --separate_trunks"
                  "--beta 0.01 --prior_std 2.0"
                  "--beta 0.01 --input_layernorm"
                  "--beta 0.01 --epochs 30 --early_stop_patience 10" )
ARM_EVAL_DELTA=( ""
                 ""
                 "--separate_trunks"
                 "--prior_std 2.0"
                 "--input_layernorm"
                 "" )

for k in "${!ARM_SUFFIX[@]}"; do
    SFX="${ARM_SUFFIX[$k]}"
    ARM_TAG="arm$((k+1))-${SFX}"
    read -r -a TRAIN_DELTA <<< "${ARM_TRAIN_DELTA[$k]}"
    EVAL_DELTA=()
    if [ -n "${ARM_EVAL_DELTA[$k]}" ]; then
        read -r -a EVAL_DELTA <<< "${ARM_EVAL_DELTA[$k]}"
    fi

    echo ""
    echo "===================================================="
    echo "ARM $((k+1)): ${SFX}   (delta: ${ARM_TRAIN_DELTA[$k]})"
    echo "Weights -> ./router_weights/fcvr/fcvr-${MODEL_SHORTCODE}-${DATASET_SHORTCODE}-${SFX}/"
    echo "===================================================="

    if run_step "train-${SFX}" python scripts/python/fcvr-tuning.py \
            "${COMMON_TRAIN[@]}" "${TRAIN_DELTA[@]}" \
            --run_suffix "$SFX" \
            --wandb_tags "$ARM_TAG" train; then
        # ${arr[@]+...} guards the empty-array expansion under `set -u`.
        eval_id  "$SFX" "$RD/arms" "$ARM_TAG" ${EVAL_DELTA[@]+"${EVAL_DELTA[@]}"} || true
        eval_ood "$SFX" "$RD/arms" "$ARM_TAG" ${EVAL_DELTA[@]+"${EVAL_DELTA[@]}"} || true
    else
        echo "SKIPPING evals for ${SFX} (training failed)"
    fi
done

echo ""
echo "############################################################"
echo "# OVERNIGHT RUN COMPLETE                                    #"
echo "############################################################"
if [ -s "$FAIL_LOG" ]; then
    echo "FAILURES:"
    cat "$FAIL_LOG"
else
    echo "No failures."
fi
echo "Results:  $RD/phase0 and $RD/arms"
echo "Logs:     $RD/logs"
echo "wandb:    project ${WANDB_PROJECT}"

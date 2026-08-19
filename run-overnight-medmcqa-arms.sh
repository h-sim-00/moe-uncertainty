#!/bin/bash
# ============================================================================
# OVERNIGHT DRIVER -- MedMCQA answer-only vs answer+explanation comparison
# (branch MedMCQA-comparison). ONE command, end to end:
#
#   preflight  split manifest, data inspection, label-mask check for BOTH arms
#              (incl. "first loss token == letter token"), GPU memory check for
#              arm B, collision sweep, evaluator smoke test
#   stage1     Stage-1 LoRA (Q/K/V + experts, exp4-train-ans values) per arm
#   map        Stage-2a MAP routers per arm            (det row)
#   fcvr       Stage-2 FCVR beta=0.01, pretrained prior, Susceptible-10 per arm
#   eval       letter read-out (evaluate_letter.py) on val + test for:
#                zero-shot (under each arm's prompt), OBQA-ansmask reference
#                (if its weights are on this host), and arm A / arm B x
#                {kvq_ft, det, fcvr}
#   report     letter_eval_report.py -> results/reports/letter_arms_{val,test}.md
#
#   arm A  target_mode=letter              prompt -> "<letter>"            (answer-only loss)
#   arm B  target_mode=answer_explanation  prompt -> "<letter>\nExplanation: <exp><eos>"
#   Same 30k/1k/1k MedMCQA rows, same HPs; only system prompt + target differ.
#
# COLLISION SAFETY: refuses to start if ANY output it would write exists
# (adapters, router weights, eval JSONs) unless RESUME=1 (skip finished steps)
# or ALLOW_EXISTING=1 (overwrite -- you really have to mean it).
#
# Usage on quail-1 (tmux so an ssh drop does not kill it; conda env moe_env):
#     tmux new -s arms
#     bash run-overnight-medmcqa-arms.sh
#     # detach Ctrl-b d ; reattach: tmux attach -t arms
#   after a crash (fix it, then):   RESUME=1 bash run-overnight-medmcqa-arms.sh
#
# Env knobs (all optional):
#   PHASES="preflight,stage1,map,fcvr,eval,report"   subset/order of phases
#   ARMS="letter answer_explanation"                 which arms
#   BETAS="0.01"   SEED=42   S=35   LAYERS="5 6 7 8 19 20 28 29 30 31"
#   STAGE1_BATCH=8 (both arms; drop to 4 if the memory check says OOM)
#   STAGE1_EPOCHS=3 STAGE1_LR=1e-4 EXPERT_LORA_R=64 STAGE1_EVAL_EVERY=1500 STAGE1_PATIENCE=3 MAX_SEQ_LEN=768
#   MAP_EPOCHS=3 MAP_BATCH=4 MAP_LR=1e-4 MAP_EVAL_EVERY=1500 MAP_PATIENCE=2
#   FCVR_EPOCHS=5 FCVR_BATCH=4 GRAD_ACCUM=4 FCVR_LR=1e-4 FCVR_EVAL_EVERY=500 FCVR_PATIENCE=3 KL_MASK=attention
#   N_VAL=0 N_TEST=0 (0 = full 1000)   EVAL_BATCH=8
#   OBQA_REF_ADAPTER=adapters/granite-obqa-ansmask  OBQA_REF_FCVR_SUFFIX=ansmask-pretrained-prior-beta0.01
#   SKIP_PREFLIGHT=1 (skip the GPU memory check only)  SKIP_INSPECT=1 (skip medmcqa-gen-inspect.py)
#   RESUME=1  ALLOW_EXISTING=1  WANDB_API_KEY=...  WANDB_PROJECT=moe-uncertainty
# ============================================================================

set -Eeo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# --- Optional: activate conda env (uncomment if your tmux shell hasn't) ---
# source "$(conda info --base)/etc/profile.d/conda.sh"
# conda activate moe_env

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
MODEL_SHORTCODE="granite"
DATASET_SHORTCODE="medmcqa_gen"
PHASES="${PHASES:-preflight,stage1,map,fcvr,eval,report}"
ARMS="${ARMS:-letter answer_explanation}"
SEED="${SEED:-42}"
S="${S:-35}"
read -r -a LAYERS <<< "${LAYERS:-5 6 7 8 19 20 28 29 30 31}"
read -r -a BETAS <<< "${BETAS:-0.01}"

# Stage 1 (exp4-train-ans values; identical for both arms)
FINETUNE_MODE="qkv_experts"
EXPERT_LORA_R="${EXPERT_LORA_R:-64}"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-3}"
STAGE1_BATCH="${STAGE1_BATCH:-8}"
STAGE1_LR="${STAGE1_LR:-1e-4}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
STAGE1_EVAL_EVERY="${STAGE1_EVAL_EVERY:-1500}"
STAGE1_PATIENCE="${STAGE1_PATIENCE:-3}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-768}"

# Stage 2a MAP (MedMCQA launcher values)
MAP_EPOCHS="${MAP_EPOCHS:-3}"
MAP_BATCH="${MAP_BATCH:-4}"
MAP_LR="${MAP_LR:-1e-4}"
MAP_EVAL_EVERY="${MAP_EVAL_EVERY:-1500}"
MAP_PATIENCE="${MAP_PATIENCE:-2}"

# Stage 2 FCVR (MedMCQA launcher values)
FCVR_EPOCHS="${FCVR_EPOCHS:-5}"
FCVR_BATCH="${FCVR_BATCH:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
FCVR_LR="${FCVR_LR:-1e-4}"
FCVR_EVAL_EVERY="${FCVR_EVAL_EVERY:-500}"
FCVR_PATIENCE="${FCVR_PATIENCE:-3}"
PRIOR_SOURCE="pretrained"
KL_MASK="${KL_MASK:-attention}"

# Eval
N_VAL="${N_VAL:-0}"
N_TEST="${N_TEST:-0}"
EVAL_BATCH="${EVAL_BATCH:-8}"
OUT_DIR="results/letter_eval"
OBQA_REF_ADAPTER="${OBQA_REF_ADAPTER:-adapters/granite-obqa-ansmask}"
OBQA_REF_FCVR_SUFFIX="${OBQA_REF_FCVR_SUFFIX:-ansmask-pretrained-prior-beta0.01}"

WANDB_PROJECT="${WANDB_PROJECT:-moe-uncertainty}"
RUN_TAG="$(date +%Y%m%d-%H%M%S)"
mkdir -p logs "$OUT_DIR" results/reports results/data
LOG="logs/overnight-medmcqa-arms-${RUN_TAG}.log"
exec > >(tee -a "$LOG") 2>&1
export RUN_TAG WANDB_PROJECT

echo "############################################################"
echo "# MedMCQA comparison overnight run ${RUN_TAG}"
echo "# Repo:   $REPO_ROOT"
echo "# Branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?')  Commit: $(git rev-parse --short HEAD 2>/dev/null || echo '?')"
echo "# Python: $(which python)"
echo "# Phases: $PHASES   Arms: $ARMS   Betas: ${BETAS[*]}   Seed: $SEED   S: $S"
echo "# Stage1: ${FINETUNE_MODE} r=${EXPERT_LORA_R} epochs=${STAGE1_EPOCHS} bs=${STAGE1_BATCH} lr=${STAGE1_LR} eval_every=${STAGE1_EVAL_EVERY} patience=${STAGE1_PATIENCE} max_seq_len=${MAX_SEQ_LEN}"
echo "# MAP:    epochs=${MAP_EPOCHS} bs=${MAP_BATCH} lr=${MAP_LR} eval_every=${MAP_EVAL_EVERY} patience=${MAP_PATIENCE}"
echo "# FCVR:   epochs<=${FCVR_EPOCHS} bs=${FCVR_BATCH}x${GRAD_ACCUM} lr=${FCVR_LR} eval_every=${FCVR_EVAL_EVERY} patience=${FCVR_PATIENCE} prior=${PRIOR_SOURCE} kl_mask=${KL_MASK} layers=${LAYERS[*]}"
echo "# RESUME=${RESUME:-0} ALLOW_EXISTING=${ALLOW_EXISTING:-0}   Log: $LOG"
echo "############################################################"

# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
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

on_error() {
    local exit_code=$? line=$1
    echo ""
    echo "############################################################"
    echo "# FAILED at line ${line} (exit ${exit_code}) -- ${STEP:-unknown step}"
    echo "# Full log: ${LOG}   (fix, then: RESUME=1 bash run-overnight-medmcqa-arms.sh)"
    echo "############################################################"
    notify error "MedMCQA-arms run FAILED: ${STEP:-unknown step}" \
        "Host $(hostname), run ${RUN_TAG}, exit ${exit_code} at line ${line}. Log: ${LOG}"
    exit "$exit_code"
}
trap 'on_error $LINENO' ERR

has_phase() { [[ ",$PHASES," == *",$1,"* ]]; }

arm_sfx() {   # target_mode -> artefact suffix
    case "$1" in
        letter) echo "armA-letter" ;;
        answer_explanation) echo "armB-ansexp" ;;
        *) echo "ERROR: unknown arm '$1' (letter|answer_explanation)" >&2; return 1 ;;
    esac
}
adapter_dir() { echo "./adapters/${MODEL_SHORTCODE}-${DATASET_SHORTCODE}-$(arm_sfx "$1")"; }
map_dir()     { echo "./router_weights/base/${MODEL_SHORTCODE}_${DATASET_SHORTCODE}-$(arm_sfx "$1")"; }
fcvr_sfx()    { echo "$(arm_sfx "$1")-${PRIOR_SOURCE}-prior-beta$2"; }
fcvr_dir()    { echo "./router_weights/fcvr/fcvr-${MODEL_SHORTCODE}-${DATASET_SHORTCODE}-$(fcvr_sfx "$1" "$2")"; }
LAST_LAYER="${LAYERS[${#LAYERS[@]}-1]}"
eval_json()   { echo "${OUT_DIR}/$1_$2.json"; }      # tag split

# skip_done <path>: true (skip) if RESUME=1 and the artefact exists
skip_done() {
    if [ "${RESUME:-0}" = "1" ] && [ -e "$1" ]; then
        echo "  [RESUME] exists, skipping: $1"; return 0
    fi
    return 1
}

n_arg() { [ "$1" -gt 0 ] 2>/dev/null && echo "--n $1" || true; }

# every output path this run would write (for the collision sweep)
all_outputs() {
    for ARM in $ARMS; do
        has_phase stage1 && echo "$(adapter_dir "$ARM")"
        has_phase map && echo "$(map_dir "$ARM")"
        for B in "${BETAS[@]}"; do has_phase fcvr && echo "$(fcvr_dir "$ARM" "$B")"; done
        if has_phase eval; then
            SFX=$(arm_sfx "$ARM")
            for SPLIT in val test; do
                echo "$(eval_json "zero-shot-prompt-${SFX}" "$SPLIT")"
                echo "$(eval_json "${SFX}_kvq_s${SEED}" "$SPLIT")"
                echo "$(eval_json "${SFX}_det_s${SEED}" "$SPLIT")"
                for B in "${BETAS[@]}"; do echo "$(eval_json "${SFX}_fcvr-beta${B}_S${S}-s${SEED}" "$SPLIT")"; done
            done
        fi
    done
}

# ============================================================================
# PHASE: preflight
# ============================================================================
if has_phase preflight; then
    STEP="preflight: collision sweep"
    echo ""; echo "==== preflight 1/6: collision sweep ===="
    COLLISIONS=()
    while IFS= read -r P; do [ -n "$P" ] && [ -e "$P" ] && COLLISIONS+=("$P"); done < <(all_outputs)
    if [ ${#COLLISIONS[@]} -gt 0 ]; then
        if [ "${RESUME:-0}" = "1" ]; then
            echo "RESUME=1: the following outputs exist and will be SKIPPED (not overwritten):"
            printf '  %s\n' "${COLLISIONS[@]}"
        elif [ "${ALLOW_EXISTING:-0}" = "1" ]; then
            echo "ALLOW_EXISTING=1: the following outputs exist and WILL BE OVERWRITTEN:"
            printf '  %s\n' "${COLLISIONS[@]}"
        else
            echo "ERROR: these output paths already exist -- refusing to overwrite:" >&2
            printf '  %s\n' "${COLLISIONS[@]}" >&2
            echo "Use RESUME=1 to skip finished steps, or ALLOW_EXISTING=1 to overwrite, or move them." >&2
            exit 1
        fi
    else
        echo "OK: no collisions. Will write:"; all_outputs | sed 's/^/  /'
    fi

    STEP="preflight: split manifest"
    echo ""; echo "==== preflight 2/6: frozen split manifest (write if absent, else verify) ===="
    python write-medmcqa-split-manifest.py --seed "$SEED"

    if [ "${SKIP_INSPECT:-0}" != "1" ]; then
        STEP="preflight: data inspection"
        echo ""; echo "==== preflight 3/6: medmcqa-gen-inspect.py (token lengths, 'Ans.' prefix rate) ===="
        if [ -f results/data/medmcqa_gen_inspect.json ]; then
            echo "  results/data/medmcqa_gen_inspect.json exists -- skipping"
        else
            python medmcqa-gen-inspect.py --seed "$SEED" --max_seq_len "$MAX_SEQ_LEN"
        fi
    fi

    STEP="preflight: label-mask checks"
    echo ""; echo "==== preflight 4/6: label-mask check, ALL val rows, both arms ===="
    for ARM in $ARMS; do
        echo "--- target_mode=$ARM ---"
        python check-answer-only-labels.py --dataset_shortcode "$DATASET_SHORTCODE" --target_mode "$ARM" \
            --num_batches 0 --seed "$SEED"
    done

    if [ "${SKIP_PREFLIGHT:-0}" != "1" ]; then
        STEP="preflight: GPU memory check (arm B, worst-case batch)"
        echo ""; echo "==== preflight 5/6: Stage-1 memory check, answer_explanation @ bs ${STAGE1_BATCH} (SKIP_PREFLIGHT=1 to skip) ===="
        python expert-lora-memory-check.py --model_shortcode "$MODEL_SHORTCODE" --dataset_shortcode "$DATASET_SHORTCODE" \
            --finetune_mode "$FINETUNE_MODE" --expert_lora_r "$EXPERT_LORA_R" --batch_size "$STAGE1_BATCH" \
            --lr "$STAGE1_LR" --epochs "$STAGE1_EPOCHS" --target_mode answer_explanation --max_seq_len "$MAX_SEQ_LEN" \
            --seed "$SEED" 2>&1 | tee "logs/arms-memcheck-${RUN_TAG}.log"
        if grep -q "VERDICT: OUT OF MEMORY" "logs/arms-memcheck-${RUN_TAG}.log"; then
            echo "ERROR: Stage-1 does not fit at bs ${STAGE1_BATCH}. Re-run with STAGE1_BATCH=4 (applies to BOTH arms)." >&2
            exit 1
        fi
    else
        echo ""; echo "==== preflight 5/6: memory check SKIPPED (SKIP_PREFLIGHT=1) ===="
    fi

    STEP="preflight: evaluator smoke test"
    echo ""; echo "==== preflight 6/6: evaluate_letter.py smoke test (zero-shot, 50 val rows) ===="
    python evaluate_letter.py --dataset_shortcode "$DATASET_SHORTCODE" --split val --target_mode letter \
        --method zero_shot --n 50 --batch_size "$EVAL_BATCH" --seed "$SEED" --n_boot 200 \
        --tag "smoke-zero-shot-${RUN_TAG}" --out_dir "${OUT_DIR}/smoke"
    echo "preflight done -- $(date)"
fi

# ============================================================================
# PHASE: stage1  (both arms, sequentially)
# ============================================================================
if has_phase stage1; then
    for ARM in $ARMS; do
        SFX=$(arm_sfx "$ARM"); ADIR=$(adapter_dir "$ARM")
        STEP="Stage 1 ($ARM -> $ADIR)"
        echo ""; echo "==== Stage 1 [$ARM] -> $ADIR -- $(date) ===="
        if skip_done "$ADIR/expert_lora.pt"; then continue; fi
        SLOG="logs/arms-${SFX}-stage1-${RUN_TAG}.log"
        python scripts/python/kvq-tuning.py \
            --model_shortcode "$MODEL_SHORTCODE" --dataset_shortcode "$DATASET_SHORTCODE" \
            --finetune_mode "$FINETUNE_MODE" --expert_lora_r "$EXPERT_LORA_R" \
            --adapter_suffix "$SFX" --target_mode "$ARM" \
            --epochs "$STAGE1_EPOCHS" --batch_size "$STAGE1_BATCH" --lr "$STAGE1_LR" --warmup_ratio "$WARMUP_RATIO" \
            --seed "$SEED" --early_stop_patience "$STAGE1_PATIENCE" --eval_every "$STAGE1_EVAL_EVERY" \
            --max_seq_len "$MAX_SEQ_LEN" 2>&1 | tee "$SLOG"
        [ -f "$ADIR/expert_lora.pt" ] || { echo "ERROR: Stage 1 finished but $ADIR/expert_lora.pt is missing." >&2; exit 1; }
        { echo "== stage1 $ARM ($(date)) =="; grep -E "Optim: AdamW|token lengths|Train samples|validation loss|Early stopping|Fine-tuning complete" "$SLOG"; } \
            >> "${OUT_DIR}/train_info_${SFX}.txt" || true
        echo "Stage 1 [$ARM] done -- $(date)"
    done
fi

# ============================================================================
# PHASE: map  (Stage-2a MAP routers per arm -> det row)
# ============================================================================
if has_phase map; then
    for ARM in $ARMS; do
        SFX=$(arm_sfx "$ARM"); ADIR=$(adapter_dir "$ARM"); MDIR=$(map_dir "$ARM")
        STEP="Stage 2a MAP ($ARM -> $MDIR)"
        echo ""; echo "==== Stage 2a MAP [$ARM] -> $MDIR -- $(date) ===="
        if skip_done "$MDIR/layer_${LAST_LAYER}_weights.pt"; then continue; fi
        [ -d "$ADIR" ] || { echo "ERROR: Stage-1 adapter $ADIR missing (run stage1 first)." >&2; exit 1; }
        SLOG="logs/arms-${SFX}-map-${RUN_TAG}.log"
        python scripts/python/router-tuning.py \
            --model_shortcode "$MODEL_SHORTCODE" --dataset_shortcode "$DATASET_SHORTCODE" \
            --base_adapter_path "$ADIR" --map_suffix "$SFX" --target_mode "$ARM" \
            --epochs "$MAP_EPOCHS" --batch_size "$MAP_BATCH" --lr "$MAP_LR" \
            --early_stop_patience "$MAP_PATIENCE" --eval_every "$MAP_EVAL_EVERY" --max_seq_len "$MAX_SEQ_LEN" \
            --seed "$SEED" 2>&1 | tee "$SLOG"
        { echo "== map $ARM ($(date)) =="; grep -E "validation loss|Early stopping|complete" "$SLOG"; } \
            >> "${OUT_DIR}/train_info_${SFX}.txt" || true
        echo "MAP [$ARM] done -- $(date)"
    done
fi

# ============================================================================
# PHASE: fcvr  (Stage-2 FCVR per arm x beta, pretrained prior)
# ============================================================================
if has_phase fcvr; then
    for ARM in $ARMS; do
        SFX=$(arm_sfx "$ARM"); ADIR=$(adapter_dir "$ARM")
        for B in "${BETAS[@]}"; do
            FDIR=$(fcvr_dir "$ARM" "$B"); RSFX=$(fcvr_sfx "$ARM" "$B")
            STEP="Stage 2 FCVR ($ARM beta=$B -> $FDIR)"
            echo ""; echo "==== Stage 2 FCVR [$ARM] beta=$B -> $FDIR -- $(date) ===="
            if skip_done "$FDIR/layer_${LAST_LAYER}_weights.pt"; then continue; fi
            [ -d "$ADIR" ] || { echo "ERROR: Stage-1 adapter $ADIR missing (run stage1 first)." >&2; exit 1; }
            SLOG="logs/arms-${SFX}-fcvr-beta${B}-${RUN_TAG}.log"
            python scripts/python/fcvr-tuning.py \
                --model_shortcode "$MODEL_SHORTCODE" --dataset_shortcode "$DATASET_SHORTCODE" \
                --base_adapter_path "$ADIR" --target_mode "$ARM" \
                --swap_layers "${LAYERS[@]}" --load_layers --train_layers "${LAYERS[@]}" \
                --epochs "$FCVR_EPOCHS" --batch_size "$FCVR_BATCH" --grad_accum_steps "$GRAD_ACCUM" \
                --lr "$FCVR_LR" --warmup_ratio "$WARMUP_RATIO" --early_stop_patience "$FCVR_PATIENCE" \
                --eval_every "$FCVR_EVAL_EVERY" --max_seq_len "$MAX_SEQ_LEN" --beta "$B" --seed "$SEED" \
                --run_suffix "$RSFX" --prior_source "$PRIOR_SOURCE" --kl_mask "$KL_MASK" 2>&1 | tee "$SLOG"
            { echo "== fcvr $ARM beta=$B ($(date)) =="; grep -E "val NLL|validation|Early stopping|complete" "$SLOG"; } \
                >> "${OUT_DIR}/train_info_${SFX}.txt" || true
            echo "FCVR [$ARM] beta=$B done -- $(date)"
        done
    done
fi

# ============================================================================
# PHASE: eval  (letter read-out, val + test)
# ============================================================================
run_eval() {   # <tag> <split> <prompt-arm> <method> [extra args...]
    local TAG="$1" SPLIT="$2" PARM="$3" METHOD="$4"; shift 4
    local N; if [ "$SPLIT" = "val" ]; then N="$N_VAL"; else N="$N_TEST"; fi
    STEP="eval $TAG [$SPLIT]"
    echo ""; echo "---- eval $TAG [$SPLIT] method=$METHOD prompt=$PARM -- $(date) ----"
    if skip_done "$(eval_json "$TAG" "$SPLIT")"; then return 0; fi
    local OW=(); [ "${ALLOW_EXISTING:-0}" = "1" ] && OW=(--overwrite)
    # shellcheck disable=SC2046
    python evaluate_letter.py --model_shortcode "$MODEL_SHORTCODE" --dataset_shortcode "$DATASET_SHORTCODE" \
        --split "$SPLIT" --target_mode "$PARM" --method "$METHOD" --batch_size "$EVAL_BATCH" \
        --num_samples "$S" --seed "$SEED" --tag "$TAG" --out_dir "$OUT_DIR" $(n_arg "$N") "${OW[@]}" "$@"
}

if has_phase eval; then
    for SPLIT in val test; do
        # 0. untuned base model under each arm's prompt
        for ARM in $ARMS; do
            run_eval "zero-shot-prompt-$(arm_sfx "$ARM")" "$SPLIT" "$ARM" zero_shot
        done
        # 1. OBQA-trained corrected-recipe reference (exp4-train-ans), if present on this host
        if [ -d "$OBQA_REF_ADAPTER" ]; then
            run_eval "ref-obqa-ansmask_kvq" "$SPLIT" letter kvq_ft --kvq_adapter_path "$OBQA_REF_ADAPTER"
            REF_FCVR="./router_weights/fcvr/fcvr-${MODEL_SHORTCODE}-obqa-${OBQA_REF_FCVR_SUFFIX}"
            if [ -d "$REF_FCVR" ]; then
                run_eval "ref-obqa-ansmask_fcvr_S${S}-s${SEED}" "$SPLIT" letter fcvr \
                    --kvq_adapter_path "$OBQA_REF_ADAPTER" --weights_dataset_shortcode obqa \
                    --swap_layers "${LAYERS[@]}" --run_suffix "$OBQA_REF_FCVR_SUFFIX" --prior_source pretrained
            else
                echo "  (no $REF_FCVR -> skipping OBQA FCVR reference)"
            fi
        else
            echo "  (no $OBQA_REF_ADAPTER on this host -> skipping OBQA reference rows)"
        fi
        # 2. the two arms x {kvq_ft, det, fcvr}
        for ARM in $ARMS; do
            SFX=$(arm_sfx "$ARM"); ADIR=$(adapter_dir "$ARM")
            [ -d "$ADIR" ] || { echo "ERROR: $ADIR missing -- run stage1 first." >&2; exit 1; }
            run_eval "${SFX}_kvq_s${SEED}" "$SPLIT" "$ARM" kvq_ft --kvq_adapter_path "$ADIR"
            if [ -d "$(map_dir "$ARM")" ]; then
                run_eval "${SFX}_det_s${SEED}" "$SPLIT" "$ARM" det --kvq_adapter_path "$ADIR" --map_suffix "$SFX"
            else
                echo "  (no $(map_dir "$ARM") -> skipping det row for $ARM)"
            fi
            for B in "${BETAS[@]}"; do
                if [ -d "$(fcvr_dir "$ARM" "$B")" ]; then
                    run_eval "${SFX}_fcvr-beta${B}_S${S}-s${SEED}" "$SPLIT" "$ARM" fcvr --kvq_adapter_path "$ADIR" \
                        --swap_layers "${LAYERS[@]}" --run_suffix "$(fcvr_sfx "$ARM" "$B")" --prior_source "$PRIOR_SOURCE"
                else
                    echo "  (no $(fcvr_dir "$ARM" "$B") -> skipping fcvr beta=$B row for $ARM)"
                fi
            done
        done
    done
    echo "eval done -- $(date)"
fi

# ============================================================================
# PHASE: report
# ============================================================================
if has_phase report; then
    for SPLIT in val test; do
        STEP="report [$SPLIT]"
        echo ""; echo "==== report [$SPLIT] ===="
        python letter_eval_report.py --split "$SPLIT" --in_dir "$OUT_DIR" --out_dir results/reports || \
            echo "  (no $SPLIT results yet)"
    done
    echo ""; echo "Training info (optimizer steps, val curves, early stopping):"
    for ARM in $ARMS; do
        F="${OUT_DIR}/train_info_$(arm_sfx "$ARM").txt"
        [ -f "$F" ] && { echo "--- $F ---"; cat "$F"; }
    done
fi

echo ""
echo "############################################################"
echo "# ALL DONE -- $(date)"
echo "# Table:   results/reports/letter_arms_test.md  (and _val.md)"
echo "# Evals:   ${OUT_DIR}/*.json  (+ *_perexample.jsonl)"
echo "# Full log: ${LOG}"
echo "############################################################"
notify info "MedMCQA-arms run finished (${PHASES})" \
    "Host $(hostname), run ${RUN_TAG}. Arms: ${ARMS}. Betas: ${BETAS[*]}. Table: results/reports/letter_arms_test.md. Log: ${LOG}"

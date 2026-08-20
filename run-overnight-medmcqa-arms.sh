#!/bin/bash
# ============================================================================
# OVERNIGHT DRIVER -- MedMCQA answer-only vs answer+explanation comparison
# (branch MedMCQA-comparison). ONE command, end to end; helpers in medmcqa-arms-lib.sh.
#
#   arm A  target_mode=letter              prompt -> "<letter>"                       (answer-only loss)
#   arm B  target_mode=answer_explanation  prompt -> "<letter>\nExplanation: <exp><eos>" (loss on both)
#   IDENTICAL prompt, identical rows (shared eligible-ID list), identical HPs; only the target differs.
#
#   preflight  collision sweep, split manifest, data inspection, label-mask check for
#              BOTH arms on all val rows (incl. "first loss token == letter token"),
#              arm-B GPU memory check (worst-case batch), evaluator smoke test
#   stage1     Stage-1 LoRA (Q/K/V + experts, exp4-train-ans values) per arm
#   map        Stage-2a MAP routers per arm                                  (det row)
#   fcvr       Stage-2 FCVR beta (default 0.01), pretrained prior, Susceptible-10 per arm
#   eval       evaluate_letter.py on val + test: zero-shot, OBQA-ansmask reference (if on
#              this host), arm A / arm B x {kvq, det} (deterministic, 1 seed) and x fcvr
#              (stochastic read-out, EVAL_SEEDS seeds)
#   report     letter_eval_report.py -> results/reports/letter_arms_{val,test}.md
#              (rows, PAIRED arm B - arm A deltas with bootstrap CIs, multi-seed summary)
#
# COLLISION SAFETY: refuses to start if ANY output it would write exists unless
# RESUME=1 (skip finished steps) or ALLOW_EXISTING=1 (overwrite).
#
# Usage on quail-1 (tmux so an ssh drop does not kill it; conda env moe_env):
#     tmux new -s arms
#     bash run-overnight-medmcqa-arms.sh
#   after a crash (fix it, then):   RESUME=1 bash run-overnight-medmcqa-arms.sh
#
# Env knobs (all optional):
#   PHASES="preflight,stage1,map,fcvr,eval,report"   ARMS="letter answer_explanation"
#   BETAS="0.01"  SEED=42  EVAL_SEEDS="42 43 44"  S=35  LAYERS="5 6 7 8 19 20 28 29 30 31"
#   STAGE1_BATCH=8 (both arms; 4 if the memory check says OOM)  STAGE1_EPOCHS=3 STAGE1_LR=1e-4
#   EXPERT_LORA_R=64 STAGE1_EVAL_EVERY=1500 STAGE1_PATIENCE=3 MAX_SEQ_LEN=768
#   MAP_EPOCHS=3 MAP_BATCH=4 MAP_LR=1e-4 MAP_EVAL_EVERY=1500 MAP_PATIENCE=2
#   FCVR_EPOCHS=5 FCVR_BATCH=4 GRAD_ACCUM=4 FCVR_LR=1e-4 FCVR_EVAL_EVERY=500 FCVR_PATIENCE=3 KL_MASK=attention
#   N_VAL=0 N_TEST=0 (0 = all)  EVAL_BATCH=8
#   OBQA_REF_ADAPTER=adapters/granite-obqa-ansmask  OBQA_REF_FCVR_SUFFIX=ansmask-pretrained-prior-beta0.01
#   SKIP_PREFLIGHT=1 (GPU memory check only)  SKIP_INSPECT=1  RESUME=1  ALLOW_EXISTING=1
#   WANDB_API_KEY=...  WANDB_PROJECT=moe-uncertainty
# ============================================================================
set -Eeo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
# source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate moe_env   # if your tmux shell hasn't

# ---- config ------------------------------------------------------------------
MODEL_SHORTCODE="granite"; DATASET_SHORTCODE="medmcqa_gen"
PHASES="${PHASES:-preflight,stage1,map,fcvr,eval,report}"
ARMS="${ARMS:-letter answer_explanation}"
SEED="${SEED:-42}"; SEED0="$SEED"; S="${S:-35}"
read -r -a EVAL_SEEDS <<< "${EVAL_SEEDS:-42 43 44}"
read -r -a LAYERS <<< "${LAYERS:-5 6 7 8 19 20 28 29 30 31}"
read -r -a BETAS <<< "${BETAS:-0.01}"
# Stage 1 (exp4-train-ans values; identical for both arms)
FINETUNE_MODE="qkv_experts"; EXPERT_LORA_R="${EXPERT_LORA_R:-64}"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-3}"; STAGE1_BATCH="${STAGE1_BATCH:-8}"; STAGE1_LR="${STAGE1_LR:-1e-4}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"; STAGE1_EVAL_EVERY="${STAGE1_EVAL_EVERY:-1500}"
STAGE1_PATIENCE="${STAGE1_PATIENCE:-3}"; MAX_SEQ_LEN="${MAX_SEQ_LEN:-768}"
# Stage 2a MAP / Stage 2 FCVR (MedMCQA launcher values)
MAP_EPOCHS="${MAP_EPOCHS:-3}"; MAP_BATCH="${MAP_BATCH:-4}"; MAP_LR="${MAP_LR:-1e-4}"
MAP_EVAL_EVERY="${MAP_EVAL_EVERY:-1500}"; MAP_PATIENCE="${MAP_PATIENCE:-2}"
FCVR_EPOCHS="${FCVR_EPOCHS:-5}"; FCVR_BATCH="${FCVR_BATCH:-4}"; GRAD_ACCUM="${GRAD_ACCUM:-4}"
FCVR_LR="${FCVR_LR:-1e-4}"; FCVR_EVAL_EVERY="${FCVR_EVAL_EVERY:-500}"; FCVR_PATIENCE="${FCVR_PATIENCE:-3}"
PRIOR_SOURCE="pretrained"; KL_MASK="${KL_MASK:-attention}"
# Eval
N_VAL="${N_VAL:-0}"; N_TEST="${N_TEST:-0}"; EVAL_BATCH="${EVAL_BATCH:-8}"; OUT_DIR="results/letter_eval"
OBQA_REF_ADAPTER="${OBQA_REF_ADAPTER:-adapters/granite-obqa-ansmask}"
OBQA_REF_FCVR_SUFFIX="${OBQA_REF_FCVR_SUFFIX:-ansmask-pretrained-prior-beta0.01}"

WANDB_PROJECT="${WANDB_PROJECT:-moe-uncertainty}"; RUN_TAG="$(date +%Y%m%d-%H%M%S)"
export RUN_TAG WANDB_PROJECT
mkdir -p logs "$OUT_DIR" results/reports results/data
LOG="logs/overnight-medmcqa-arms-${RUN_TAG}.log"; exec > >(tee -a "$LOG") 2>&1
# shellcheck source=medmcqa-arms-lib.sh
source "$REPO_ROOT/medmcqa-arms-lib.sh"
trap 'on_error $LINENO' ERR
LAST_LAYER="${LAYERS[${#LAYERS[@]}-1]}"

echo "############################################################"
echo "# MedMCQA comparison overnight run ${RUN_TAG}   branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?') commit=$(git rev-parse --short HEAD 2>/dev/null || echo '?')"
echo "# Phases: $PHASES | Arms: $ARMS | Betas: ${BETAS[*]} | train seed $SEED | eval seeds ${EVAL_SEEDS[*]} | S=$S"
echo "# Stage1: ${FINETUNE_MODE} r=${EXPERT_LORA_R} epochs=${STAGE1_EPOCHS} bs=${STAGE1_BATCH} lr=${STAGE1_LR} eval_every=${STAGE1_EVAL_EVERY} patience=${STAGE1_PATIENCE} max_seq_len=${MAX_SEQ_LEN}"
echo "# MAP:    epochs=${MAP_EPOCHS} bs=${MAP_BATCH} lr=${MAP_LR} eval_every=${MAP_EVAL_EVERY} patience=${MAP_PATIENCE}"
echo "# FCVR:   epochs<=${FCVR_EPOCHS} bs=${FCVR_BATCH}x${GRAD_ACCUM} lr=${FCVR_LR} eval_every=${FCVR_EVAL_EVERY} patience=${FCVR_PATIENCE} prior=${PRIOR_SOURCE} kl_mask=${KL_MASK} layers=${LAYERS[*]}"
echo "# RESUME=${RESUME:-0} ALLOW_EXISTING=${ALLOW_EXISTING:-0} | python=$(which python) | log: $LOG"
echo "############################################################"

# ============================================================================
if has_phase preflight; then
    STEP="preflight: collision sweep";      echo ""; echo "==== preflight 1/6: collision sweep ====";   collision_sweep
    STEP="preflight: split manifest";       echo ""; echo "==== preflight 2/6: frozen split manifest (write if absent, else verify) ===="
    python write-medmcqa-split-manifest.py --seed "$SEED"
    if [ "${SKIP_INSPECT:-0}" != "1" ]; then
        STEP="preflight: data inspection";  echo ""; echo "==== preflight 3/6: medmcqa-gen-inspect.py (token lengths, 'Ans.' prefix rate) ===="
        if [ -f results/data/medmcqa_gen_inspect.json ]; then echo "  results/data/medmcqa_gen_inspect.json exists -- skipping"
        else python medmcqa-gen-inspect.py --seed "$SEED" --max_seq_len "$MAX_SEQ_LEN"; fi
    fi
    STEP="preflight: label-mask checks";    echo ""; echo "==== preflight 4/6: label-mask check, ALL val rows, both arms (shared eligible-ID list) ===="
    for ARM in $ARMS; do
        echo "--- target_mode=$ARM ---"
        python check-answer-only-labels.py --dataset_shortcode "$DATASET_SHORTCODE" --target_mode "$ARM" \
            --max_seq_len "$MAX_SEQ_LEN" --num_batches 0 --seed "$SEED"
    done
    if [ "${SKIP_PREFLIGHT:-0}" != "1" ]; then
        STEP="preflight: GPU memory check (arm B, worst-case batch)"
        echo ""; echo "==== preflight 5/6: Stage-1 memory check, answer_explanation @ bs ${STAGE1_BATCH} (SKIP_PREFLIGHT=1 to skip) ===="
        python expert-lora-memory-check.py --model_shortcode "$MODEL_SHORTCODE" --dataset_shortcode "$DATASET_SHORTCODE" \
            --finetune_mode "$FINETUNE_MODE" --expert_lora_r "$EXPERT_LORA_R" --batch_size "$STAGE1_BATCH" --lr "$STAGE1_LR" \
            --epochs "$STAGE1_EPOCHS" --target_mode answer_explanation --max_seq_len "$MAX_SEQ_LEN" --seed "$SEED" \
            2>&1 | tee "logs/arms-memcheck-${RUN_TAG}.log"
        if grep -q "VERDICT: OUT OF MEMORY" "logs/arms-memcheck-${RUN_TAG}.log"; then
            echo "ERROR: Stage-1 does not fit at bs ${STAGE1_BATCH}. Re-run with STAGE1_BATCH=4 (applies to BOTH arms)." >&2; exit 1
        fi
    else echo ""; echo "==== preflight 5/6: memory check SKIPPED (SKIP_PREFLIGHT=1) ===="; fi
    STEP="preflight: evaluator smoke test"; echo ""; echo "==== preflight 6/6: evaluate_letter.py smoke test (zero-shot, 50 val rows) ===="
    python evaluate_letter.py --dataset_shortcode "$DATASET_SHORTCODE" --split val --method zero_shot --n 50 \
        --batch_size "$EVAL_BATCH" --seed "$SEED" --n_boot 200 --tag "smoke-zero-shot-${RUN_TAG}" --out_dir "${OUT_DIR}/smoke"
    echo "preflight done -- $(date)"
fi

# ============================================================================
if has_phase stage1; then for ARM in $ARMS; do
    SFX=$(arm_sfx "$ARM"); ADIR=$(adapter_dir "$ARM")
    run_stage "Stage 1 [$ARM] -> $ADIR" "$ADIR/expert_lora.pt" "logs/arms-${SFX}-stage1-${RUN_TAG}.log" \
        "${OUT_DIR}/train_info_${SFX}.txt" "Optim: AdamW|eligibility|token lengths|Train samples|validation loss|Early stopping|Fine-tuning complete" -- \
        python scripts/python/kvq-tuning.py --model_shortcode "$MODEL_SHORTCODE" --dataset_shortcode "$DATASET_SHORTCODE" \
            --finetune_mode "$FINETUNE_MODE" --expert_lora_r "$EXPERT_LORA_R" --adapter_suffix "$SFX" --target_mode "$ARM" \
            --epochs "$STAGE1_EPOCHS" --batch_size "$STAGE1_BATCH" --lr "$STAGE1_LR" --warmup_ratio "$WARMUP_RATIO" \
            --seed "$SEED" --early_stop_patience "$STAGE1_PATIENCE" --eval_every "$STAGE1_EVAL_EVERY" --max_seq_len "$MAX_SEQ_LEN"
done; fi

# ============================================================================
if has_phase map; then for ARM in $ARMS; do
    SFX=$(arm_sfx "$ARM"); ADIR=$(adapter_dir "$ARM"); MDIR=$(map_dir "$ARM")
    [ -d "$ADIR" ] || { echo "ERROR: Stage-1 adapter $ADIR missing (run stage1 first)." >&2; exit 1; }
    run_stage "Stage 2a MAP [$ARM] -> $MDIR" "$MDIR/layer_${LAST_LAYER}_weights.pt" "logs/arms-${SFX}-map-${RUN_TAG}.log" \
        "${OUT_DIR}/train_info_${SFX}.txt" "validation loss|Early stopping|complete" -- \
        python scripts/python/router-tuning.py --model_shortcode "$MODEL_SHORTCODE" --dataset_shortcode "$DATASET_SHORTCODE" \
            --base_adapter_path "$ADIR" --map_suffix "$SFX" --target_mode "$ARM" \
            --epochs "$MAP_EPOCHS" --batch_size "$MAP_BATCH" --lr "$MAP_LR" --early_stop_patience "$MAP_PATIENCE" \
            --eval_every "$MAP_EVAL_EVERY" --max_seq_len "$MAX_SEQ_LEN" --seed "$SEED"
done; fi

# ============================================================================
if has_phase fcvr; then for ARM in $ARMS; do for B in "${BETAS[@]}"; do
    SFX=$(arm_sfx "$ARM"); ADIR=$(adapter_dir "$ARM"); FDIR=$(fcvr_dir "$ARM" "$B")
    [ -d "$ADIR" ] || { echo "ERROR: Stage-1 adapter $ADIR missing (run stage1 first)." >&2; exit 1; }
    run_stage "Stage 2 FCVR [$ARM] beta=$B -> $FDIR" "$FDIR/layer_${LAST_LAYER}_weights.pt" "logs/arms-${SFX}-fcvr-beta${B}-${RUN_TAG}.log" \
        "${OUT_DIR}/train_info_${SFX}.txt" "validation loss|Early stopping|complete" -- \
        python scripts/python/fcvr-tuning.py --model_shortcode "$MODEL_SHORTCODE" --dataset_shortcode "$DATASET_SHORTCODE" \
            --base_adapter_path "$ADIR" --target_mode "$ARM" --swap_layers "${LAYERS[@]}" --load_layers --train_layers "${LAYERS[@]}" \
            --epochs "$FCVR_EPOCHS" --batch_size "$FCVR_BATCH" --grad_accum_steps "$GRAD_ACCUM" --lr "$FCVR_LR" \
            --warmup_ratio "$WARMUP_RATIO" --early_stop_patience "$FCVR_PATIENCE" --eval_every "$FCVR_EVAL_EVERY" \
            --max_seq_len "$MAX_SEQ_LEN" --beta "$B" --seed "$SEED" --run_suffix "$(fcvr_sfx "$ARM" "$B")" \
            --prior_source "$PRIOR_SOURCE" --kl_mask "$KL_MASK"
done; done; fi

# ============================================================================
if has_phase eval; then for SPLIT in val test; do
    run_eval zero-shot "$SPLIT" "$SEED0" zero_shot                                  # untuned model, comparison prompt
    if [ -d "$OBQA_REF_ADAPTER" ]; then                                            # exp4-train-ans OBQA model on MedMCQA
        run_eval "ref-obqa-ansmask_kvq-s${SEED0}" "$SPLIT" "$SEED0" kvq_ft --kvq_adapter_path "$OBQA_REF_ADAPTER" --system_prompt mcq
        REF_FCVR="./router_weights/fcvr/fcvr-${MODEL_SHORTCODE}-obqa-${OBQA_REF_FCVR_SUFFIX}"
        if [ -d "$REF_FCVR" ]; then for SEED in "${EVAL_SEEDS[@]}"; do
            run_eval "ref-obqa-ansmask_fcvr_S${S}-s${SEED}" "$SPLIT" "$SEED" fcvr --kvq_adapter_path "$OBQA_REF_ADAPTER" \
                --system_prompt mcq --weights_dataset_shortcode obqa --swap_layers "${LAYERS[@]}" \
                --run_suffix "$OBQA_REF_FCVR_SUFFIX" --prior_source pretrained
        done; else echo "  (no $REF_FCVR -> skipping OBQA FCVR reference)"; fi
    else echo "  (no $OBQA_REF_ADAPTER on this host -> skipping OBQA reference rows)"; fi
    for ARM in $ARMS; do
        SFX=$(arm_sfx "$ARM"); ADIR=$(adapter_dir "$ARM")
        [ -d "$ADIR" ] || { echo "ERROR: $ADIR missing -- run stage1 first." >&2; exit 1; }
        run_eval "$(tag_kvq "$ARM" "$SEED0")" "$SPLIT" "$SEED0" kvq_ft --kvq_adapter_path "$ADIR"
        if [ -d "$(map_dir "$ARM")" ]; then
            run_eval "$(tag_det "$ARM" "$SEED0")" "$SPLIT" "$SEED0" det --kvq_adapter_path "$ADIR" --map_suffix "$SFX"
        else echo "  (no $(map_dir "$ARM") -> skipping det row for $ARM)"; fi
        for B in "${BETAS[@]}"; do
            if [ -d "$(fcvr_dir "$ARM" "$B")" ]; then for SEED in "${EVAL_SEEDS[@]}"; do
                run_eval "$(tag_fcvr "$ARM" "$B" "$SEED")" "$SPLIT" "$SEED" fcvr --kvq_adapter_path "$ADIR" \
                    --swap_layers "${LAYERS[@]}" --run_suffix "$(fcvr_sfx "$ARM" "$B")" --prior_source "$PRIOR_SOURCE"
            done; else echo "  (no $(fcvr_dir "$ARM" "$B") -> skipping fcvr beta=$B rows for $ARM)"; fi
        done
    done
done; echo "eval done -- $(date)"; fi

# ============================================================================
if has_phase report; then
    for SPLIT in val test; do
        STEP="report [$SPLIT]"; echo ""; echo "==== report [$SPLIT] ===="
        python letter_eval_report.py --split "$SPLIT" --in_dir "$OUT_DIR" --out_dir results/reports || echo "  (no $SPLIT results yet)"
    done
    echo ""; echo "Training info (optimizer steps, eligibility, val curves, early stopping):"
    for ARM in $ARMS; do F="${OUT_DIR}/train_info_$(arm_sfx "$ARM").txt"; [ -f "$F" ] && { echo "--- $F ---"; cat "$F"; }; done
fi

echo ""; echo "############################################################"
echo "# ALL DONE -- $(date)"
echo "# Table: results/reports/letter_arms_test.md (and _val.md) | evals: ${OUT_DIR}/ | log: ${LOG}"
echo "############################################################"
notify info "MedMCQA-arms run finished (${PHASES})" \
    "Host $(hostname), run ${RUN_TAG}. Arms: ${ARMS}. Betas: ${BETAS[*]}. Table: results/reports/letter_arms_test.md. Log: ${LOG}"

#!/bin/bash
# ============================================================================
# OVERNIGHT DRIVER -- OBQA answer-only vs answer+explanation (fact1) comparison
# (branch OBQA-comparison). ONE command, end to end; helpers in medmcqa-arms-lib.sh
# (fully generic: every path derives from DATASET_SHORTCODE).
#
#   arm A  Stage-1 adapter FROZEN from exp4-train-ans (adapters/granite-obqa-ansmask);
#          FCVR routers RETRAINED here (phase fcvr_arma) because exp4's FCVR run
#          averaged padding positions into the KL regularizer (kl.mean() with no
#          mask over the right-padded batch -- effective beta drifted with batch
#          length). The retrain is exp4-identical (legacy `obqa` rows, MCQ prompt,
#          answer-only path, 10 epochs) EXCEPT --kl_mask attention; new weights go
#          to fcvr-granite-obqa-ansmask-klattn-pretrained-prior-beta* and the old
#          buggy exp4 dirs (…-ansmask-pretrained-prior-beta*) are never written.
#          Arm A was still trained with the plain MCQ system prompt on unfiltered
#          rows (and 10 FCVR epochs vs arm B's 5), so the arms differ in training
#          prompt/rows/epochs as well as target -- the accepted confound of this
#          branch (state it in any write-up). Evaluated on the SAME obqa_gen rows
#          via --system_prompt mcq --weights_dataset_shortcode obqa.
#   arm B  target_mode=answer_explanation  prompt -> "<letter>\nExplanation: <fact1><eos>"
#          (loss on letter AND OpenBookQA's gold science fact), trained fresh on
#          obqa_gen (HF "additional" config; split replicates exp4's byte-for-byte).
#
#   preflight  arm-A adapter presence (this run only makes sense on quail-1),
#              collision sweep, split manifest + legacy-obqa parity cross-check,
#              label-mask check for both target modes on all val rows, arm-B GPU
#              memory check, evaluator smoke test
#   fcvr_arma  Stage-2 FCVR retrain for arm A: exp4's exact recipe on the frozen
#              adapter, only change --kl_mask attention -> NEW klattn dirs
#   stage1     Stage-1 LoRA (Q/K/V + experts) -- arm B only
#   map        Stage-2a MAP routers -- arm B only            (det row, unpaired)
#   fcvr       Stage-2 FCVR beta (default 0.01), pretrained prior, Susceptible-10 -- arm B
#   eval       evaluate_letter.py on val + test: zero-shot, arm A (frozen Stage-1 +
#              klattn FCVR, mcq prompt) x {kvq, fcvr}, arm B x {kvq, det, fcvr};
#              EVAL_OLD_ARMA_FCVR=1 adds unpaired rows on exp4's OLD buggy FCVR
#              weights (quantifies the padding-KL bug alone)
#   report     letter_eval_report.py -> results/reports/obqa_gen/letter_arms_{val,test}.md
#   ood        evaluate_ilv_ood_arms.py --id_dataset obqa_gen (paired ILV OoD arms)
#
# COLLISION SAFETY: refuses to start if ANY output it would write exists unless
# RESUME=1 (skip finished steps) or ALLOW_EXISTING=1 (overwrite). OUT_DIR is
# separate from the MedMCQA run's (identical tags would collide otherwise).
#
# Usage on quail-1 (tmux so an ssh drop does not kill it; venv moe_env):
#     tmux new -s obqa-arms
#     bash run-overnight-obqa-arms.sh
#   after a crash (fix it, then):   RESUME=1 bash run-overnight-obqa-arms.sh
#
# Env knobs (all optional):
#   PHASES="preflight,fcvr_arma,stage1,map,fcvr,eval,report,ood"   ARMS="answer_explanation"
#   ARMA_FCVR_EPOCHS=10 (exp4's value)  EVAL_OLD_ARMA_FCVR=1 (extra rows on the old buggy weights)
#   BETAS="0.01"  SEED=42  EVAL_SEEDS="42 43 44"  S=35  LAYERS="5 6 7 8 19 20 28 29 30 31"
#   STAGE1_BATCH=8 STAGE1_EPOCHS=3 STAGE1_LR=1e-4 EXPERT_LORA_R=64
#   STAGE1_EVAL_EVERY=1500 STAGE1_PATIENCE=3 MAX_SEQ_LEN=768
#   MAP_EPOCHS=3 MAP_BATCH=4 MAP_LR=1e-4 MAP_EVAL_EVERY=1500 MAP_PATIENCE=2
#   FCVR_EPOCHS=5 FCVR_BATCH=4 GRAD_ACCUM=4 FCVR_LR=1e-4 FCVR_EVAL_EVERY=500 FCVR_PATIENCE=3 KL_MASK=attention
#   N_VAL=0 N_TEST=0 (0 = all)  EVAL_BATCH=8  OOD_N_PER_DOMAIN=500
#   ARMA_ADAPTER=adapters/granite-obqa-ansmask  ARMA_FCVR_PREFIX=ansmask-pretrained-prior
#   SKIP_PREFLIGHT=1 (GPU memory check only)  RESUME=1  ALLOW_EXISTING=1
#   WANDB_API_KEY=...  WANDB_PROJECT=moe-uncertainty
# ============================================================================
set -Eeo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# ---- config ------------------------------------------------------------------
MODEL_SHORTCODE="granite"; DATASET_SHORTCODE="obqa_gen"
PHASES="${PHASES:-preflight,fcvr_arma,stage1,map,fcvr,eval,report,ood}"
ARMS="${ARMS:-answer_explanation}"                 # arm A is frozen; only arm B trains
SEED="${SEED:-42}"; SEED0="$SEED"; S="${S:-35}"
read -r -a EVAL_SEEDS <<< "${EVAL_SEEDS:-42 43 44}"
read -r -a LAYERS <<< "${LAYERS:-5 6 7 8 19 20 28 29 30 31}"
read -r -a BETAS <<< "${BETAS:-0.01}"
# Stage 1 (MedMCQA-arms values, identical HPs for the new arm B)
FINETUNE_MODE="qkv_experts"; EXPERT_LORA_R="${EXPERT_LORA_R:-64}"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-3}"; STAGE1_BATCH="${STAGE1_BATCH:-8}"; STAGE1_LR="${STAGE1_LR:-1e-4}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"; STAGE1_EVAL_EVERY="${STAGE1_EVAL_EVERY:-1500}"
STAGE1_PATIENCE="${STAGE1_PATIENCE:-3}"; MAX_SEQ_LEN="${MAX_SEQ_LEN:-768}"
# Stage 2a MAP / Stage 2 FCVR (MedMCQA-arms values)
MAP_EPOCHS="${MAP_EPOCHS:-3}"; MAP_BATCH="${MAP_BATCH:-4}"; MAP_LR="${MAP_LR:-1e-4}"
MAP_EVAL_EVERY="${MAP_EVAL_EVERY:-1500}"; MAP_PATIENCE="${MAP_PATIENCE:-2}"
FCVR_EPOCHS="${FCVR_EPOCHS:-5}"; FCVR_BATCH="${FCVR_BATCH:-4}"; GRAD_ACCUM="${GRAD_ACCUM:-4}"
FCVR_LR="${FCVR_LR:-1e-4}"; FCVR_EVAL_EVERY="${FCVR_EVAL_EVERY:-500}"; FCVR_PATIENCE="${FCVR_PATIENCE:-3}"
PRIOR_SOURCE="pretrained"; KL_MASK="${KL_MASK:-attention}"
# Eval. OUT_DIR/report dir are DISTINCT from the MedMCQA run's: the arm tags are
# identical across the two studies and results/letter_eval would collide.
N_VAL="${N_VAL:-0}"; N_TEST="${N_TEST:-0}"; EVAL_BATCH="${EVAL_BATCH:-8}"; OUT_DIR="results/letter_eval_obqa_gen"
REPORT_DIR="results/reports/obqa_gen"
# Arm A: Stage-1 adapter frozen from exp4 (quail-1 only); FCVR routers retrained
# by the fcvr_arma phase into NEW klattn dirs. FCVR dir per beta:
#   router_weights/fcvr/fcvr-granite-obqa-${ARMA_FCVR_PREFIX}-beta<B>
# exp4's OLD (padding-in-KL) dirs keep the ansmask-pretrained-prior prefix and are
# read-only here (optional EVAL_OLD_ARMA_FCVR=1 reference rows).
ARMA_ADAPTER="${ARMA_ADAPTER:-adapters/granite-obqa-ansmask}"
ARMA_FCVR_PREFIX="${ARMA_FCVR_PREFIX:-ansmask-klattn-pretrained-prior}"
ARMA_OLD_FCVR_PREFIX="ansmask-pretrained-prior"      # exp4's buggy-KL weights (never written)
ARMA_FCVR_EPOCHS="${ARMA_FCVR_EPOCHS:-10}"           # exp4's value (arm B uses FCVR_EPOCHS=5)
ARMA_WEIGHTS_DS="obqa"
# OoD arms comparison
OOD_N_PER_DOMAIN="${OOD_N_PER_DOMAIN:-500}"; OOD_TAG="obqa-arms"; OOD_DIR="results/ilv_ood_arms"

WANDB_PROJECT="${WANDB_PROJECT:-moe-uncertainty}"; RUN_TAG="$(date +%Y%m%d-%H%M%S)"
export RUN_TAG WANDB_PROJECT
mkdir -p logs "$OUT_DIR" "$REPORT_DIR" results/data
LOG="logs/overnight-obqa-arms-${RUN_TAG}.log"; exec > >(tee -a "$LOG") 2>&1
# shellcheck source=medmcqa-arms-lib.sh
source "$REPO_ROOT/medmcqa-arms-lib.sh"
trap 'on_error $LINENO' ERR
LAST_LAYER="${LAYERS[${#LAYERS[@]}-1]}"

arma_fcvr_sfx() { echo "${ARMA_FCVR_PREFIX}-beta$1"; }                                  # <beta>
arma_fcvr_dir() { echo "./router_weights/fcvr/fcvr-${MODEL_SHORTCODE}-${ARMA_WEIGHTS_DS}-$(arma_fcvr_sfx "$1")"; }
arma_old_fcvr_sfx() { echo "${ARMA_OLD_FCVR_PREFIX}-beta$1"; }                          # <beta>
arma_old_fcvr_dir() { echo "./router_weights/fcvr/fcvr-${MODEL_SHORTCODE}-${ARMA_WEIGHTS_DS}-$(arma_old_fcvr_sfx "$1")"; }
tag_old_fcvr() { echo "armA-letter-exp4kl_fcvr-beta$1_S${S}-s$2"; }                     # <beta> <seed>

# Extend the lib's collision sweep with the arm-A retrained FCVR dirs, the arm-A
# eval rows, and the ood outputs (the lib's all_outputs only enumerates $ARMS = arm B).
eval "$(declare -f all_outputs | sed 's/^all_outputs/lib_all_outputs/')"
all_outputs() {
    lib_all_outputs
    local B SPLIT SEED
    if has_phase fcvr_arma; then
        for B in "${BETAS[@]}"; do echo "$(arma_fcvr_dir "$B")"; done
    fi
    if has_phase eval; then
        for SPLIT in val test; do
            echo "$(eval_json "$(tag_kvq letter "$SEED0")" "$SPLIT")"
            for B in "${BETAS[@]}"; do for SEED in "${EVAL_SEEDS[@]}"; do
                echo "$(eval_json "$(tag_fcvr letter "$B" "$SEED")" "$SPLIT")"
                if [ "${EVAL_OLD_ARMA_FCVR:-0}" = "1" ]; then echo "$(eval_json "$(tag_old_fcvr "$B" "$SEED")" "$SPLIT")"; fi
            done; done
        done
    fi
    if has_phase ood; then
        local STEM="${OOD_DIR}/${OOD_TAG}_test_data-s${SEED0}_mc-s${SEED0}"
        echo "${STEM}.json"; echo "${STEM}.md"; echo "${STEM}_perexample.jsonl"
    fi
}

echo "############################################################"
echo "# OBQA comparison overnight run ${RUN_TAG}   branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?') commit=$(git rev-parse --short HEAD 2>/dev/null || echo '?')"
echo "# Phases: $PHASES | Arms trained: $ARMS (arm A = frozen exp4 ${ARMA_ADAPTER}) | Betas: ${BETAS[*]} | train seed $SEED | eval seeds ${EVAL_SEEDS[*]} | S=$S"
echo "# Stage1: ${FINETUNE_MODE} r=${EXPERT_LORA_R} epochs=${STAGE1_EPOCHS} bs=${STAGE1_BATCH} lr=${STAGE1_LR} eval_every=${STAGE1_EVAL_EVERY} patience=${STAGE1_PATIENCE} max_seq_len=${MAX_SEQ_LEN}"
echo "# MAP:    epochs=${MAP_EPOCHS} bs=${MAP_BATCH} lr=${MAP_LR} eval_every=${MAP_EVAL_EVERY} patience=${MAP_PATIENCE}"
echo "# FCVR:   epochs<=${FCVR_EPOCHS} bs=${FCVR_BATCH}x${GRAD_ACCUM} lr=${FCVR_LR} eval_every=${FCVR_EVAL_EVERY} patience=${FCVR_PATIENCE} prior=${PRIOR_SOURCE} kl_mask=${KL_MASK} layers=${LAYERS[*]}"
echo "# RESUME=${RESUME:-0} ALLOW_EXISTING=${ALLOW_EXISTING:-0} | python=$(which python) | log: $LOG"
echo "############################################################"

# ============================================================================
if has_phase preflight; then
    STEP="preflight: arm-A weights";        echo ""; echo "==== preflight 1/7: frozen exp4 arm-A Stage-1 adapter present? ===="
    [ -d "$ARMA_ADAPTER" ] || { echo "ERROR: $ARMA_ADAPTER missing. The exp4-train-ans arm-A adapter lives on quail-1 only; this driver is pointless without it." >&2; exit 1; }
    if has_phase fcvr_arma; then
        echo "  OK: $ARMA_ADAPTER (klattn FCVR dirs are OUTPUTS of the fcvr_arma phase; collision sweep covers them)"
    else
        for B in "${BETAS[@]}"; do
            [ -d "$(arma_fcvr_dir "$B")" ] || { echo "ERROR: $(arma_fcvr_dir "$B") missing (arm-A klattn FCVR weights for beta=$B). Run the fcvr_arma phase first." >&2; exit 1; }
        done
        echo "  OK: $ARMA_ADAPTER + klattn FCVR dirs for betas ${BETAS[*]}"
    fi
    if [ "${EVAL_OLD_ARMA_FCVR:-0}" = "1" ]; then for B in "${BETAS[@]}"; do
        [ -d "$(arma_old_fcvr_dir "$B")" ] || { echo "ERROR: EVAL_OLD_ARMA_FCVR=1 but $(arma_old_fcvr_dir "$B") (exp4's old buggy-KL weights) is missing." >&2; exit 1; }
    done; fi
    STEP="preflight: collision sweep";      echo ""; echo "==== preflight 2/7: collision sweep ====";   collision_sweep
    STEP="preflight: split manifest";       echo ""; echo "==== preflight 3/7: frozen split manifest + legacy-obqa parity check (write if absent, else verify) ===="
    python write-obqa-split-manifest.py --seed "$SEED"
    STEP="preflight: label-mask checks";    echo ""; echo "==== preflight 4/7: label-mask check, ALL val rows, both target modes (shared eligible-ID list) ===="
    for TM in letter answer_explanation; do
        echo "--- target_mode=$TM ---"
        python check-answer-only-labels.py --dataset_shortcode "$DATASET_SHORTCODE" --target_mode "$TM" \
            --max_seq_len "$MAX_SEQ_LEN" --num_batches 0 --seed "$SEED"
    done
    if [ "${SKIP_PREFLIGHT:-0}" != "1" ]; then
        STEP="preflight: GPU memory check (arm B, worst-case batch)"
        echo ""; echo "==== preflight 5/7: Stage-1 memory check, answer_explanation @ bs ${STAGE1_BATCH} (SKIP_PREFLIGHT=1 to skip) ===="
        python expert-lora-memory-check.py --model_shortcode "$MODEL_SHORTCODE" --dataset_shortcode "$DATASET_SHORTCODE" \
            --finetune_mode "$FINETUNE_MODE" --expert_lora_r "$EXPERT_LORA_R" --batch_size "$STAGE1_BATCH" --lr "$STAGE1_LR" \
            --epochs "$STAGE1_EPOCHS" --target_mode answer_explanation --max_seq_len "$MAX_SEQ_LEN" --seed "$SEED" \
            2>&1 | tee "logs/obqa-arms-memcheck-${RUN_TAG}.log"
        if grep -q "VERDICT: OUT OF MEMORY" "logs/obqa-arms-memcheck-${RUN_TAG}.log"; then
            echo "ERROR: Stage-1 does not fit at bs ${STAGE1_BATCH}. Re-run with STAGE1_BATCH=4." >&2; exit 1
        fi
    else echo ""; echo "==== preflight 5/7: memory check SKIPPED (SKIP_PREFLIGHT=1) ===="; fi
    STEP="preflight: evaluator smoke test"; echo ""; echo "==== preflight 6/7: evaluate_letter.py smoke test (zero-shot, 50 val rows) ===="
    python evaluate_letter.py --dataset_shortcode "$DATASET_SHORTCODE" --split val --method zero_shot --n 50 \
        --batch_size "$EVAL_BATCH" --seed "$SEED" --n_boot 200 --tag "smoke-zero-shot-${RUN_TAG}" --out_dir "${OUT_DIR}/smoke"
    STEP="preflight: arm-A smoke test";     echo ""; echo "==== preflight 7/7: arm-A (exp4, mcq prompt) smoke test (kvq, 50 val rows) ===="
    python evaluate_letter.py --dataset_shortcode "$DATASET_SHORTCODE" --split val --method kvq_ft --n 50 \
        --kvq_adapter_path "$ARMA_ADAPTER" --system_prompt mcq \
        --batch_size "$EVAL_BATCH" --seed "$SEED" --n_boot 200 --tag "smoke-armA-kvq-${RUN_TAG}" --out_dir "${OUT_DIR}/smoke"
    echo "preflight done -- $(date)"
fi

# ============================================================================
# Arm-A FCVR retrain: exp4's exact recipe (legacy `obqa` shortcode -> unfiltered
# rows + MCQ prompt + answer-only path; 10 epochs, bs 4x4, epoch-end validation)
# on the FROZEN Stage-1 adapter, fresh FCVR heads (no --load_layers), pretrained
# prior (no MAP weights involved). The ONLY change vs exp4 is --kl_mask attention
# (exp4 averaged padding positions into the KL). --eval_every 0 --max_seq_len 0
# --target_mode explanation pin exp4's behaviour explicitly. Output goes to the
# NEW klattn dirs; exp4's old dirs are never touched.
if has_phase fcvr_arma; then for B in "${BETAS[@]}"; do
    AFDIR=$(arma_fcvr_dir "$B")
    [ -d "$ARMA_ADAPTER" ] || { echo "ERROR: $ARMA_ADAPTER missing." >&2; exit 1; }
    run_stage "Stage 2 FCVR [arm A klattn] beta=$B -> $AFDIR" "$AFDIR/layer_${LAST_LAYER}_weights.pt" \
        "logs/obqa-arms-armA-klattn-fcvr-beta${B}-${RUN_TAG}.log" \
        "${OUT_DIR}/train_info_armA-klattn.txt" "validation loss|Early stopping|complete" -- \
        python scripts/python/fcvr-tuning.py --model_shortcode "$MODEL_SHORTCODE" --dataset_shortcode "$ARMA_WEIGHTS_DS" \
            --base_adapter_path "$ARMA_ADAPTER" --swap_layers "${LAYERS[@]}" --train_layers "${LAYERS[@]}" \
            --epochs "$ARMA_FCVR_EPOCHS" --batch_size 4 --grad_accum_steps 4 --lr 1e-4 --warmup_ratio 0.05 \
            --early_stop_patience 3 --eval_every 0 --max_seq_len 0 --target_mode explanation \
            --beta "$B" --seed "$SEED" --run_suffix "$(arma_fcvr_sfx "$B")" --prior_source pretrained \
            --kl_mask attention
done; fi

# ============================================================================
if has_phase stage1; then for ARM in $ARMS; do
    SFX=$(arm_sfx "$ARM"); ADIR=$(adapter_dir "$ARM")
    run_stage "Stage 1 [$ARM] -> $ADIR" "$ADIR/expert_lora.pt" "logs/obqa-arms-${SFX}-stage1-${RUN_TAG}.log" \
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
    run_stage "Stage 2a MAP [$ARM] -> $MDIR" "$MDIR/layer_${LAST_LAYER}_weights.pt" "logs/obqa-arms-${SFX}-map-${RUN_TAG}.log" \
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
    run_stage "Stage 2 FCVR [$ARM] beta=$B -> $FDIR" "$FDIR/layer_${LAST_LAYER}_weights.pt" "logs/obqa-arms-${SFX}-fcvr-beta${B}-${RUN_TAG}.log" \
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
    # ---- arm A: frozen exp4 Stage-1 adapter + klattn-retrained FCVR, its native mcq
    # prompt, obqa-suffixed weights, evaluated on the SAME obqa_gen rows/ids so the
    # report pairs it with arm B.
    run_eval "$(tag_kvq letter "$SEED0")" "$SPLIT" "$SEED0" kvq_ft \
        --kvq_adapter_path "$ARMA_ADAPTER" --system_prompt mcq
    for B in "${BETAS[@]}"; do
        if [ -d "$(arma_fcvr_dir "$B")" ]; then for SEED in "${EVAL_SEEDS[@]}"; do
            run_eval "$(tag_fcvr letter "$B" "$SEED")" "$SPLIT" "$SEED" fcvr \
                --kvq_adapter_path "$ARMA_ADAPTER" --system_prompt mcq \
                --weights_dataset_shortcode "$ARMA_WEIGHTS_DS" --swap_layers "${LAYERS[@]}" \
                --run_suffix "$(arma_fcvr_sfx "$B")" --prior_source pretrained
        done; else echo "  (no $(arma_fcvr_dir "$B") -> skipping arm-A fcvr beta=$B rows)"; fi
        # optional unpaired reference: exp4's OLD buggy-KL FCVR weights (bug-effect probe)
        if [ "${EVAL_OLD_ARMA_FCVR:-0}" = "1" ]; then for SEED in "${EVAL_SEEDS[@]}"; do
            run_eval "$(tag_old_fcvr "$B" "$SEED")" "$SPLIT" "$SEED" fcvr \
                --kvq_adapter_path "$ARMA_ADAPTER" --system_prompt mcq \
                --weights_dataset_shortcode "$ARMA_WEIGHTS_DS" --swap_layers "${LAYERS[@]}" \
                --run_suffix "$(arma_old_fcvr_sfx "$B")" --prior_source pretrained
        done; fi
    done
    # (no arm-A det row: exp4 had no MAP stage -- the arm-B det row is unpaired)
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
        python letter_eval_report.py --split "$SPLIT" --in_dir "$OUT_DIR" --out_dir "$REPORT_DIR" || echo "  (no $SPLIT results yet)"
    done
    echo ""; echo "Training info (optimizer steps, eligibility, val curves, early stopping):"
    for ARM in $ARMS; do F="${OUT_DIR}/train_info_$(arm_sfx "$ARM").txt"; [ -f "$F" ] && { echo "--- $F ---"; cat "$F"; }; done
fi

# ============================================================================
if has_phase ood; then
    STEP="ood: paired ILV OoD arms comparison"
    echo ""; echo "==== ood: evaluate_ilv_ood_arms.py --id_dataset obqa_gen -- $(date) ===="
    OOD_STEM="${OOD_DIR}/${OOD_TAG}_test_data-s${SEED0}_mc-s${SEED0}"
    if skip_done "${OOD_STEM}.json"; then :; else
        OW=(); [ "${ALLOW_EXISTING:-0}" = "1" ] && OW=(--overwrite)
        python evaluate_ilv_ood_arms.py --id_dataset "$DATASET_SHORTCODE" --split test \
            --n_per_domain "$OOD_N_PER_DOMAIN" --data_seed "$SEED0" --sampling_seed "$SEED0" \
            --num_samples "$S" --batch_size "$EVAL_BATCH" --swap_layers "${LAYERS[@]}" \
            --arm_a_adapter "$ARMA_ADAPTER" --arm_b_adapter "$(adapter_dir answer_explanation)" "${OW[@]}"
    fi
fi

echo ""; echo "############################################################"
echo "# ALL DONE -- $(date)"
echo "# Table: ${REPORT_DIR}/letter_arms_test.md (and _val.md) | evals: ${OUT_DIR}/ | OoD: ${OOD_DIR}/ | log: ${LOG}"
echo "############################################################"
notify info "OBQA-arms run finished (${PHASES})" \
    "Host $(hostname), run ${RUN_TAG}. Arms trained: ${ARMS} (arm A frozen exp4). Betas: ${BETAS[*]}. Table: ${REPORT_DIR}/letter_arms_test.md. Log: ${LOG}"

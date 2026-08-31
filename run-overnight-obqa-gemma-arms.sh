#!/bin/bash
# ============================================================================
# DRIVER -- OBQA answer-only vs answer+explanation (fact1) comparison with
# Gemma 4 26B-A4B-it (branch OBQA-gemma). Same protocol as
# run-overnight-obqa-qwen-arms.sh (Qwen3.6) / run-overnight-obqa-arms.sh (Granite):
# BOTH arms trained fresh, data-parallel DDP training. Helpers in
# medmcqa-arms-lib.sh (every path derives from MODEL_SHORTCODE / DATASET_SHORTCODE,
# so nothing here can touch a Granite or Qwen artefact).
#
#   arm A  target_mode=letter, system prompt 'mcq' (letter-only instruction,
#          as the Granite OBQA arm A was trained) -> loss on the bare letter
#   arm B  target_mode=answer_explanation, system prompt 'comparison'
#          -> "<letter>\nExplanation: <fact1><eos>", loss on letter AND fact
#          (eos = Gemma's turn terminator <turn|>, see model.load_tokenizer)
#   Both arms: same obqa_gen rows (shared eligible-ID list, frozen under
#   splits/*-gemma4.txt), same epochs / HPs; only prompt+target differ.
#
#   preflight  collision sweep, split manifest (+ legacy-obqa parity), label-mask
#              check for both arms (each with its own system prompt), Stage-1 and
#              FCVR GPU memory checks at the PER-RANK batch, evaluator smoke test
#   stage1     Stage-1 LoRA (attention-only, finetune_mode=qkv) -- both arms
#   map        Stage-2a MAP routers (all 30 layers) -- both arms   (det rows)
#   fcvr       Stage-2 FCVR beta (default 0.01), pretrained prior -- both arms x
#              EVERY layer set in LAYER_SETS (default "depth", see below)
#   eval       evaluate_letter.py on val + test: zero-shot, arm A x {kvq, det,
#              fcvr x layer sets x EVAL_SEEDS} with --system_prompt mcq, arm B likewise
#   report     letter_eval_report.py -> results/reports/obqa_gen_gemma4/letter_arms_{val,test}.md
#   ood        evaluate_ilv_ood_arms.py --id_dataset obqa_gen --model_shortcode gemma4, per layer set
#
# LAYER SETS (medmcqa-arms-lib.sh: layer_set_layers / layer_set_sfx, keyed by MODEL_SHORTCODE):
#   depth    5 6 7 8 18 19 26 27 28 29  -- Granite's Susceptible-10 blocks {5-8, 19-20,
#            last four of 32} at the same RELATIVE depth in Gemma's 30 layers
#            (max |i/30 - j/32| = 0.03; the last layer stays last). Every artefact
#            carries "-layers-depth": fcvr-gemma4-obqa_gen-armA-letter-pretrained-prior-beta0.01-layers-depth,
#            tag armA-letter_fcvr-layers-depth-beta0.01_S35-s42, OoD stem obqa-arms-gemma4-layers-depth_*,
#            W&B run fcvr-gemma4-obqa_gen-...-layers-depth. This is the ONLY set here:
#            Granite's literal indices 30/31 do not exist in a 30-layer model, and
#            layer_set_layers refuses 'literal' for gemma4.
#   Stage-1 adapters and MAP routers are layer-independent (shared by any set).
#
# MULTI-GPU: training phases run under `torchrun --nproc_per_node=$NGPUS`
# (NGPUS defaults to the GPUs SLURM gave the job, else nvidia-smi's count).
# STAGE1_BATCH / MAP_BATCH / FCVR_BATCH are the EFFECTIVE (global) per-step
# batches of the Granite protocol (8 / 4 / 4x4 accum); the per-rank batch is
# derived as <global>/NGPUS and must divide exactly. Eval phases run on one GPU.
#
# COLLISION SAFETY: refuses to start if ANY output it would write exists unless
# RESUME=1 (skip finished steps) or ALLOW_EXISTING=1 (overwrite).
#
# Usage (Isambard-AI, inside an sbatch job -- see sbatch-obqa-gemma.sh /
# submit-obqa-gemma-chain.sh; conda env qwen_env, which already has gemma4):
#     PHASES=preflight            bash run-overnight-obqa-gemma-arms.sh
#     RESUME=1 PHASES=stage1,map,fcvr bash run-overnight-obqa-gemma-arms.sh
#     RESUME=1 PHASES=eval,report,ood bash run-overnight-obqa-gemma-arms.sh
#
# Env knobs (all optional):
#   PHASES="preflight,stage1,map,fcvr,eval,report,ood"   ARMS="letter answer_explanation"   NGPUS=4
#   BETAS="0.01"  SEED=42  EVAL_SEEDS="42 43 44"  S=35
#   LAYER_SETS="depth"  LAYERS_depth="5 6 7 8 18 19 26 27 28 29"
#   STAGE1_BATCH=8 STAGE1_ACCUM=1 STAGE1_EPOCHS=3 STAGE1_LR=1e-4 STAGE1_EVAL_EVERY=1500 STAGE1_PATIENCE=3 MAX_SEQ_LEN=768
#   MAP_BATCH=4 MAP_ACCUM=1 MAP_EPOCHS=3 MAP_LR=1e-4 MAP_EVAL_EVERY=1500 MAP_PATIENCE=2
#   FCVR_BATCH=4 GRAD_ACCUM=4 FCVR_EPOCHS=5 FCVR_LR=1e-4 FCVR_EVAL_EVERY=500 FCVR_PATIENCE=3 KL_MASK=attention
#   N_VAL=0 N_TEST=0 (0 = all)  EVAL_BATCH=8  OOD_N_PER_DOMAIN=500
#   GRADIENT_CHECKPOINTING=1 (default here)  SKIP_PREFLIGHT=1 (GPU memory checks only)  RESUME=1  ALLOW_EXISTING=1
#   WANDB_API_KEY=...  WANDB_PROJECT=moe-uncertainty-gemma4
# ============================================================================
set -Eeo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# ---- config ------------------------------------------------------------------
MODEL_SHORTCODE="gemma4"; DATASET_SHORTCODE="obqa_gen"
PHASES="${PHASES:-preflight,stage1,map,fcvr,eval,report,ood}"
ARMS="${ARMS:-letter answer_explanation}"
SEED="${SEED:-42}"; SEED0="$SEED"; S="${S:-35}"
read -r -a EVAL_SEEDS <<< "${EVAL_SEEDS:-42 43 44}"
read -r -a LAYER_SETS <<< "${LAYER_SETS:-depth}"
read -r -a BETAS <<< "${BETAS:-0.01}"
# Stage 1 (Granite-arms values; attention-only LoRA as on OBQA-qwen)
FINETUNE_MODE="qkv"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-3}"; STAGE1_BATCH="${STAGE1_BATCH:-8}"; STAGE1_ACCUM="${STAGE1_ACCUM:-1}"; STAGE1_LR="${STAGE1_LR:-1e-4}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"; STAGE1_EVAL_EVERY="${STAGE1_EVAL_EVERY:-1500}"
STAGE1_PATIENCE="${STAGE1_PATIENCE:-3}"; MAX_SEQ_LEN="${MAX_SEQ_LEN:-768}"
# Stage 2a MAP / Stage 2 FCVR (Granite-arms values)
MAP_EPOCHS="${MAP_EPOCHS:-3}"; MAP_BATCH="${MAP_BATCH:-4}"; MAP_ACCUM="${MAP_ACCUM:-1}"; MAP_LR="${MAP_LR:-1e-4}"
MAP_EVAL_EVERY="${MAP_EVAL_EVERY:-1500}"; MAP_PATIENCE="${MAP_PATIENCE:-2}"
FCVR_EPOCHS="${FCVR_EPOCHS:-5}"; FCVR_BATCH="${FCVR_BATCH:-4}"; GRAD_ACCUM="${GRAD_ACCUM:-4}"
FCVR_LR="${FCVR_LR:-1e-4}"; FCVR_EVAL_EVERY="${FCVR_EVAL_EVERY:-500}"; FCVR_PATIENCE="${FCVR_PATIENCE:-3}"
PRIOR_SOURCE="pretrained"; KL_MASK="${KL_MASK:-attention}"
# Eval. Separate dirs from the Granite / Qwen OBQA runs (identical arm tags would collide).
N_VAL="${N_VAL:-0}"; N_TEST="${N_TEST:-0}"; EVAL_BATCH="${EVAL_BATCH:-8}"
OUT_DIR="results/letter_eval_obqa_gen_${MODEL_SHORTCODE}"
REPORT_DIR="results/reports/obqa_gen_${MODEL_SHORTCODE}"
OOD_N_PER_DOMAIN="${OOD_N_PER_DOMAIN:-500}"; OOD_TAG="obqa-arms-${MODEL_SHORTCODE}"; OOD_DIR="results/ilv_ood_arms"
export GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}"

# ---- GPUs / launcher -----------------------------------------------------------
if [ -z "${NGPUS:-}" ]; then
    if [ -n "${SLURM_GPUS_ON_NODE:-}" ]; then NGPUS="$SLURM_GPUS_ON_NODE"
    elif command -v nvidia-smi >/dev/null 2>&1; then NGPUS="$(nvidia-smi -L 2>/dev/null | grep -c GPU || echo 1)"
    else NGPUS=1; fi
fi
[ "$NGPUS" -ge 1 ] 2>/dev/null || NGPUS=1
if [ "$NGPUS" -gt 1 ]; then LAUNCH=(torchrun --standalone --nproc_per_node="$NGPUS"); else LAUNCH=(python); fi
per_rank() {   # <global batch> <name> -> global/NGPUS, must divide exactly
    local G="$1" NAME="$2"
    if [ $(( G % NGPUS )) -ne 0 ]; then
        echo "ERROR: $NAME=$G is not divisible by NGPUS=$NGPUS (per-rank batch must be an integer; set NGPUS or the batch)." >&2; exit 1
    fi
    echo $(( G / NGPUS ))
}
STAGE1_BATCH_RANK=$(per_rank "$STAGE1_BATCH" STAGE1_BATCH)
MAP_BATCH_RANK=$(per_rank "$MAP_BATCH" MAP_BATCH)
FCVR_BATCH_RANK=$(per_rank "$FCVR_BATCH" FCVR_BATCH)

WANDB_PROJECT="${WANDB_PROJECT:-moe-uncertainty-gemma4}"; RUN_TAG="$(date +%Y%m%d-%H%M%S)"
export RUN_TAG WANDB_PROJECT
mkdir -p logs "$OUT_DIR" "$REPORT_DIR" results/data
LOG="logs/overnight-obqa-gemma-arms-${RUN_TAG}.log"; exec > >(tee -a "$LOG") 2>&1
# shellcheck source=medmcqa-arms-lib.sh
source "$REPO_ROOT/medmcqa-arms-lib.sh"
trap 'on_error $LINENO' ERR

# ---- layer sets ------------------------------------------------------------------
# set_layers <name>: fill the LAYERS array (+ LAST_LAYER) for that set; validated up front
# (layer_set_layers is keyed by MODEL_SHORTCODE=gemma4: 'depth' -> 5 6 7 8 18 19 26 27 28 29, 'literal' refused).
set_layers() {
    local L; L="$(layer_set_layers "$1")" || exit 1
    [ -n "$L" ] || { echo "ERROR: layer set '$1' is empty" >&2; exit 1; }
    read -r -a LAYERS <<< "$L"; LAST_LAYER="${LAYERS[${#LAYERS[@]}-1]}"
}
for LS in "${LAYER_SETS[@]}"; do set_layers "$LS"; echo "# layer set '$LS': ${LAYERS[*]}"; done
# MAP saves every decoder layer; its done-marker is the highest layer any set will read.
MAP_MARKER_LAYER=0; for LS in "${LAYER_SETS[@]}"; do set_layers "$LS"; [ "$LAST_LAYER" -gt "$MAP_MARKER_LAYER" ] && MAP_MARKER_LAYER="$LAST_LAYER"; done
# Layer-set-aware artefact names (override the lib's 2/3-arg versions; an empty
# or "literal" set name reproduces the lib's plain names exactly).
fcvr_sfx() { echo "$(arm_sfx "$1")-${PRIOR_SOURCE}-prior-beta$2$(layer_set_sfx "${3:-}")"; }            # <arm> <beta> [set]
fcvr_dir() { echo "./router_weights/fcvr/fcvr-${MODEL_SHORTCODE}-${DATASET_SHORTCODE}-$(fcvr_sfx "$1" "$2" "${3:-}")"; }
tag_fcvr() { echo "$(arm_sfx "$1")_fcvr$(layer_set_sfx "${4:-}")-beta$2_S${S}-s$3"; }                    # <arm> <beta> <seed> [set]
ood_tag()  { echo "${OOD_TAG}$(layer_set_sfx "${1:-}")"; }                                               # [set]
ood_stem() { echo "${OOD_DIR}/$(ood_tag "${1:-}")_test_data-s${SEED0}_mc-s${SEED0}"; }                   # [set]

# arm -> training/eval system prompt (utils.prompt.SYSTEM_INSTRUCTIONS key)
arm_prompt() {
    case "$1" in
        letter) echo "mcq" ;;
        answer_explanation) echo "comparison" ;;
        *) echo "ERROR: unknown arm '$1'" >&2; return 1 ;;
    esac
}

# Every output path this run would write (collision sweep) -- replaces the lib's
# version so the per-layer-set FCVR / eval / ood outputs are covered.
all_outputs() {
    local ARM B SPLIT SEED LS
    has_phase eval && for SPLIT in val test; do echo "$(eval_json zero-shot "$SPLIT")"; done
    for ARM in $ARMS; do
        has_phase stage1 && echo "$(adapter_dir "$ARM")"
        has_phase map && echo "$(map_dir "$ARM")"
        for LS in "${LAYER_SETS[@]}"; do for B in "${BETAS[@]}"; do has_phase fcvr && echo "$(fcvr_dir "$ARM" "$B" "$LS")"; done; done
        if has_phase eval; then
            for SPLIT in val test; do
                echo "$(eval_json "$(tag_kvq "$ARM" "$SEED0")" "$SPLIT")"
                echo "$(eval_json "$(tag_det "$ARM" "$SEED0")" "$SPLIT")"
                for LS in "${LAYER_SETS[@]}"; do for B in "${BETAS[@]}"; do for SEED in "${EVAL_SEEDS[@]}"; do
                    echo "$(eval_json "$(tag_fcvr "$ARM" "$B" "$SEED" "$LS")" "$SPLIT")"
                done; done; done
            done
        fi
    done
    if has_phase ood; then for LS in "${LAYER_SETS[@]}"; do
        echo "$(ood_stem "$LS").json"; echo "$(ood_stem "$LS").md"; echo "$(ood_stem "$LS")_perexample.jsonl"
    done; fi
}

echo "############################################################"
echo "# OBQA comparison (Gemma 4 26B-A4B) run ${RUN_TAG}   branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?') commit=$(git rev-parse --short HEAD 2>/dev/null || echo '?')"
echo "# Phases: $PHASES | Arms trained: $ARMS | Betas: ${BETAS[*]} | train seed $SEED | eval seeds ${EVAL_SEEDS[*]} | S=$S"
echo "# GPUs: NGPUS=$NGPUS launcher='${LAUNCH[*]}' | per-rank batches: stage1 ${STAGE1_BATCH_RANK}x${STAGE1_ACCUM} map ${MAP_BATCH_RANK}x${MAP_ACCUM} fcvr ${FCVR_BATCH_RANK}x${GRAD_ACCUM} | grad-ckpt=${GRADIENT_CHECKPOINTING}"
echo "# Stage1: ${FINETUNE_MODE} epochs=${STAGE1_EPOCHS} eff-bs=${STAGE1_BATCH}x${STAGE1_ACCUM} lr=${STAGE1_LR} eval_every=${STAGE1_EVAL_EVERY} patience=${STAGE1_PATIENCE} max_seq_len=${MAX_SEQ_LEN}"
echo "# MAP:    epochs=${MAP_EPOCHS} eff-bs=${MAP_BATCH}x${MAP_ACCUM} lr=${MAP_LR} eval_every=${MAP_EVAL_EVERY} patience=${MAP_PATIENCE}"
echo "# FCVR:   epochs<=${FCVR_EPOCHS} eff-bs=${FCVR_BATCH}x${GRAD_ACCUM} lr=${FCVR_LR} eval_every=${FCVR_EVAL_EVERY} patience=${FCVR_PATIENCE} prior=${PRIOR_SOURCE} kl_mask=${KL_MASK}"
for LS in "${LAYER_SETS[@]}"; do set_layers "$LS"; echo "#          layer set '$LS' (suffix '$(layer_set_sfx "$LS")'): ${LAYERS[*]}"; done
echo "# W&B project: ${WANDB_PROJECT} | RESUME=${RESUME:-0} ALLOW_EXISTING=${ALLOW_EXISTING:-0} | python=$(which python) | log: $LOG"
echo "############################################################"

# ============================================================================
if has_phase preflight; then
    STEP="preflight: collision sweep";      echo ""; echo "==== preflight 1/6: collision sweep ====";   collision_sweep
    STEP="preflight: split manifest";       echo ""; echo "==== preflight 2/6: frozen split manifest + legacy-obqa parity check (write if absent, else verify) ===="
    python write-obqa-split-manifest.py --seed "$SEED"
    STEP="preflight: label-mask checks";    echo ""; echo "==== preflight 3/6: label-mask check, ALL val rows, both arms (own system prompt; shared eligible-ID list) ===="
    for ARM in $ARMS; do
        echo "--- target_mode=$ARM system_prompt=$(arm_prompt "$ARM") ---"
        python check-answer-only-labels.py --model_shortcode "$MODEL_SHORTCODE" --dataset_shortcode "$DATASET_SHORTCODE" \
            --target_mode "$ARM" --system_prompt "$(arm_prompt "$ARM")" \
            --max_seq_len "$MAX_SEQ_LEN" --num_batches 0 --seed "$SEED"
    done
    if [ "${SKIP_PREFLIGHT:-0}" != "1" ]; then
        STEP="preflight: GPU memory check (Stage-1, worst-case batch)"
        echo ""; echo "==== preflight 4/6: Stage-1 memory check, answer_explanation @ per-rank bs ${STAGE1_BATCH_RANK} (SKIP_PREFLIGHT=1 to skip) ===="
        python expert-lora-memory-check.py --model_shortcode "$MODEL_SHORTCODE" --dataset_shortcode "$DATASET_SHORTCODE" \
            --stage stage1 --finetune_mode "$FINETUNE_MODE" --batch_size "$STAGE1_BATCH_RANK" --lr "$STAGE1_LR" \
            --epochs "$STAGE1_EPOCHS" --target_mode answer_explanation --system_prompt comparison \
            --max_seq_len "$MAX_SEQ_LEN" --seed "$SEED" 2>&1 | tee "logs/obqa-gemma-memcheck-stage1-${RUN_TAG}.log"
        if grep -q "VERDICT: OUT OF MEMORY" "logs/obqa-gemma-memcheck-stage1-${RUN_TAG}.log"; then
            echo "ERROR: Stage-1 does not fit at per-rank bs ${STAGE1_BATCH_RANK}. Re-run with STAGE1_BATCH=4 STAGE1_ACCUM=2 (same effective batch)." >&2; exit 1
        fi
        for LS in "${LAYER_SETS[@]}"; do set_layers "$LS"
            STEP="preflight: GPU memory check (FCVR, layer set $LS, worst-case batch)"
            echo ""; echo "==== preflight 5/6: FCVR memory check, layer set '$LS' (${LAYERS[*]}), answer_explanation @ per-rank bs ${FCVR_BATCH_RANK} ===="
            python expert-lora-memory-check.py --model_shortcode "$MODEL_SHORTCODE" --dataset_shortcode "$DATASET_SHORTCODE" \
                --stage fcvr --swap_layers "${LAYERS[@]}" --beta "${BETAS[0]}" --batch_size "$FCVR_BATCH_RANK" --lr "$FCVR_LR" \
                --epochs "$FCVR_EPOCHS" --target_mode answer_explanation --system_prompt comparison \
                --max_seq_len "$MAX_SEQ_LEN" --seed "$SEED" 2>&1 | tee "logs/obqa-gemma-memcheck-fcvr-${LS}-${RUN_TAG}.log"
            if grep -q "VERDICT: OUT OF MEMORY" "logs/obqa-gemma-memcheck-fcvr-${LS}-${RUN_TAG}.log"; then
                echo "ERROR: FCVR (layer set $LS) does not fit at per-rank bs ${FCVR_BATCH_RANK}. Re-run with FCVR_BATCH=$NGPUS GRAD_ACCUM=$(( 16 / NGPUS )) (same effective batch 16)." >&2; exit 1
            fi
        done
    else echo ""; echo "==== preflight 4-5/6: memory checks SKIPPED (SKIP_PREFLIGHT=1) ===="; fi
    STEP="preflight: evaluator smoke test"; echo ""; echo "==== preflight 6/6: evaluate_letter.py smoke test (zero-shot, 50 val rows) ===="
    python evaluate_letter.py --model_shortcode "$MODEL_SHORTCODE" --dataset_shortcode "$DATASET_SHORTCODE" --split val --method zero_shot --n 50 \
        --batch_size "$EVAL_BATCH" --seed "$SEED" --n_boot 200 --tag "smoke-zero-shot-${RUN_TAG}" --out_dir "${OUT_DIR}/smoke"
    echo "preflight done -- $(date)"
fi

# ============================================================================
if has_phase stage1; then for ARM in $ARMS; do
    SFX=$(arm_sfx "$ARM"); ADIR=$(adapter_dir "$ARM")
    run_stage "Stage 1 [$ARM] -> $ADIR" "$ADIR/adapter_config.json" "logs/obqa-gemma-${SFX}-stage1-${RUN_TAG}.log" \
        "${OUT_DIR}/train_info_${SFX}.txt" "Optim: AdamW|eligibility|training prompt|token lengths|Train samples|validation loss|Early stopping|Fine-tuning complete" -- \
        "${LAUNCH[@]}" scripts/python/kvq-tuning.py --model_shortcode "$MODEL_SHORTCODE" --dataset_shortcode "$DATASET_SHORTCODE" \
            --finetune_mode "$FINETUNE_MODE" --adapter_suffix "$SFX" --target_mode "$ARM" --system_prompt "$(arm_prompt "$ARM")" \
            --epochs "$STAGE1_EPOCHS" --batch_size "$STAGE1_BATCH_RANK" --grad_accum_steps "$STAGE1_ACCUM" \
            --lr "$STAGE1_LR" --warmup_ratio "$WARMUP_RATIO" \
            --seed "$SEED" --early_stop_patience "$STAGE1_PATIENCE" --eval_every "$STAGE1_EVAL_EVERY" --max_seq_len "$MAX_SEQ_LEN"
done; fi

# ============================================================================
if has_phase map; then for ARM in $ARMS; do
    SFX=$(arm_sfx "$ARM"); ADIR=$(adapter_dir "$ARM"); MDIR=$(map_dir "$ARM")
    [ -d "$ADIR" ] || { echo "ERROR: Stage-1 adapter $ADIR missing (run stage1 first)." >&2; exit 1; }
    run_stage "Stage 2a MAP [$ARM] -> $MDIR" "$MDIR/layer_${MAP_MARKER_LAYER}_weights.pt" "logs/obqa-gemma-${SFX}-map-${RUN_TAG}.log" \
        "${OUT_DIR}/train_info_${SFX}.txt" "validation loss|Early stopping|complete" -- \
        "${LAUNCH[@]}" scripts/python/router-tuning.py --model_shortcode "$MODEL_SHORTCODE" --dataset_shortcode "$DATASET_SHORTCODE" \
            --base_adapter_path "$ADIR" --map_suffix "$SFX" --target_mode "$ARM" --system_prompt "$(arm_prompt "$ARM")" \
            --epochs "$MAP_EPOCHS" --batch_size "$MAP_BATCH_RANK" --grad_accum_steps "$MAP_ACCUM" --lr "$MAP_LR" \
            --early_stop_patience "$MAP_PATIENCE" --eval_every "$MAP_EVAL_EVERY" --max_seq_len "$MAX_SEQ_LEN" --seed "$SEED"
done; fi

# ============================================================================
if has_phase fcvr; then for LS in "${LAYER_SETS[@]}"; do set_layers "$LS"; for ARM in $ARMS; do for B in "${BETAS[@]}"; do
    SFX=$(arm_sfx "$ARM"); ADIR=$(adapter_dir "$ARM"); FDIR=$(fcvr_dir "$ARM" "$B" "$LS")
    [ -d "$ADIR" ] || { echo "ERROR: Stage-1 adapter $ADIR missing (run stage1 first)." >&2; exit 1; }
    run_stage "Stage 2 FCVR [$ARM] beta=$B layers=$LS (${LAYERS[*]}) -> $FDIR" "$FDIR/layer_${LAST_LAYER}_weights.pt" \
        "logs/obqa-gemma-${SFX}-fcvr-beta${B}-${LS}-${RUN_TAG}.log" \
        "${OUT_DIR}/train_info_${SFX}.txt" "validation loss|Early stopping|complete" -- \
        "${LAUNCH[@]}" scripts/python/fcvr-tuning.py --model_shortcode "$MODEL_SHORTCODE" --dataset_shortcode "$DATASET_SHORTCODE" \
            --base_adapter_path "$ADIR" --target_mode "$ARM" --system_prompt "$(arm_prompt "$ARM")" \
            --swap_layers "${LAYERS[@]}" --load_layers --train_layers "${LAYERS[@]}" \
            --epochs "$FCVR_EPOCHS" --batch_size "$FCVR_BATCH_RANK" --grad_accum_steps "$GRAD_ACCUM" --lr "$FCVR_LR" \
            --warmup_ratio "$WARMUP_RATIO" --early_stop_patience "$FCVR_PATIENCE" --eval_every "$FCVR_EVAL_EVERY" \
            --max_seq_len "$MAX_SEQ_LEN" --beta "$B" --seed "$SEED" --run_suffix "$(fcvr_sfx "$ARM" "$B" "$LS")" \
            --prior_source "$PRIOR_SOURCE" --kl_mask "$KL_MASK"
done; done; done; fi

# ============================================================================
if has_phase eval; then for SPLIT in val test; do
    run_eval zero-shot "$SPLIT" "$SEED0" zero_shot                                  # untuned model, comparison prompt
    for ARM in $ARMS; do
        SFX=$(arm_sfx "$ARM"); ADIR=$(adapter_dir "$ARM"); SP=$(arm_prompt "$ARM")
        [ -d "$ADIR" ] || { echo "ERROR: $ADIR missing -- run stage1 first." >&2; exit 1; }
        run_eval "$(tag_kvq "$ARM" "$SEED0")" "$SPLIT" "$SEED0" kvq_ft --kvq_adapter_path "$ADIR" --system_prompt "$SP"
        if [ -d "$(map_dir "$ARM")" ]; then
            run_eval "$(tag_det "$ARM" "$SEED0")" "$SPLIT" "$SEED0" det --kvq_adapter_path "$ADIR" --map_suffix "$SFX" --system_prompt "$SP"
        else echo "  (no $(map_dir "$ARM") -> skipping det row for $ARM)"; fi
        for LS in "${LAYER_SETS[@]}"; do set_layers "$LS"; for B in "${BETAS[@]}"; do
            if [ -d "$(fcvr_dir "$ARM" "$B" "$LS")" ]; then for SEED in "${EVAL_SEEDS[@]}"; do
                run_eval "$(tag_fcvr "$ARM" "$B" "$SEED" "$LS")" "$SPLIT" "$SEED" fcvr --kvq_adapter_path "$ADIR" --system_prompt "$SP" \
                    --swap_layers "${LAYERS[@]}" --run_suffix "$(fcvr_sfx "$ARM" "$B" "$LS")" --prior_source "$PRIOR_SOURCE"
            done; else echo "  (no $(fcvr_dir "$ARM" "$B" "$LS") -> skipping fcvr beta=$B layers=$LS rows for $ARM)"; fi
        done; done
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
if has_phase ood; then for LS in "${LAYER_SETS[@]}"; do set_layers "$LS"
    STEP="ood: paired ILV OoD arms comparison (layer set $LS)"
    echo ""; echo "==== ood [$LS: ${LAYERS[*]}]: evaluate_ilv_ood_arms.py --id_dataset obqa_gen --model_shortcode ${MODEL_SHORTCODE} --tag $(ood_tag "$LS") -- $(date) ===="
    if skip_done "$(ood_stem "$LS").json"; then :; else
        OW=(); [ "${ALLOW_EXISTING:-0}" = "1" ] && OW=(--overwrite)
        # run suffixes passed explicitly (they also match the ARM_SETUP registry defaults for gemma4 = depth set)
        python evaluate_ilv_ood_arms.py --model_shortcode "$MODEL_SHORTCODE" --id_dataset "$DATASET_SHORTCODE" --split test \
            --n_per_domain "$OOD_N_PER_DOMAIN" --data_seed "$SEED0" --sampling_seed "$SEED0" \
            --num_samples "$S" --batch_size "$EVAL_BATCH" --swap_layers "${LAYERS[@]}" --tag "$(ood_tag "$LS")" \
            --arm_a_adapter "$(adapter_dir letter)" --arm_b_adapter "$(adapter_dir answer_explanation)" \
            --arm_a_run_suffix "$(fcvr_sfx letter "${BETAS[0]}" "$LS")" \
            --arm_b_run_suffix "$(fcvr_sfx answer_explanation "${BETAS[0]}" "$LS")" "${OW[@]}"
    fi
done; fi

echo ""; echo "############################################################"
echo "# ALL DONE -- $(date)"
echo "# Table: ${REPORT_DIR}/letter_arms_test.md (and _val.md) | evals: ${OUT_DIR}/ | OoD: $(for LS in "${LAYER_SETS[@]}"; do printf '%s ' "$(ood_stem "$LS").*"; done)| log: ${LOG}"
echo "############################################################"
notify info "OBQA-gemma arms run finished (${PHASES})" \
    "Host $(hostname), run ${RUN_TAG}. Arms: ${ARMS}. Betas: ${BETAS[*]}. Table: ${REPORT_DIR}/letter_arms_test.md. Log: ${LOG}"

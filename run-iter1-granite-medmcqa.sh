#!/bin/bash
# ============================================================================
# MedMCQA driver -- Granite-MoE / medmcqa_gen (branch MedMCQA)
#
# Same protocol as run-iter1-granite-medexqa.sh (supervisor iteration 1), with the
# training set switched from MedExQA (740 rows) to MedMCQA gold explanations
# (30k train / 1,000 val / 1,000 test, see utils/data.py::_load_medmcqa_gen) and the
# Stage-1 / Stage-2 training folded into the driver (no MedMCQA weights exist yet).
#
# Phases (env PHASES, comma-separated, default all in this order):
#   preflight  deps; split manifest written/verified (splits/medmcqa_gen-derived-seed42.csv);
#              data inspection (funnel, subject histogram, token lengths -> results/data/);
#              label-mask check; up-front collision check of every output path.
#   stage1     Stage-1 KVQ adapter -> ./adapters/granite-medmcqa_gen   (kvq-tuning-granite-medmcqa.sh)
#   stage2     FCVR beta sweep {0.01, 0.1}, pretrained prior, KL mask attention
#              -> ./router_weights/fcvr/fcvr-granite-medmcqa_gen-pretrained-prior-beta<b>/
#   val        SELECTION runs on the val split (N_VAL examples): stochastic S=35 readouts
#              (teacher-forced + --generate mnt256) x seeds x betas -> labels (NLI + metrics)
#              -> audit CSV (hand-label 25-50 rows) -> abstention_report --select (frozen files).
#   test       ONE evaluation per frozen configuration on the test split (N_TEST examples)
#              -> labels -> abstention_report --evaluate -> --aggregate across seeds.
#   baselines  det arm (Stage-1 stock routers) and untrained-FCVR arm on val+test; trains beta=0.
#   prior      Stage-2a MAP router tuning (router-tuning-granite-medmcqa.sh) -> FCVR map-prior -> val+test.
#   kl         KL-mask ablations: none (legacy) and answer-only -> val+test.
#   ood        input-level bridge test on every arm: ID = medmcqa_gen; OoD = obqa (far),
#              medexqa (NEAR: medical, same explanation task, different source), mmlu_law (far).
#              medmcqa_med is REFUSED for this anchor (same corpus -> leakage).
#   report     baseline_ladder_report.py --source medmcqa_gen for val and test.
#
# Outputs never collide with the MedExQA runs: token_analysis files are prefixed
# step1_medmcqa_gen_*, OoD files input_ood_medmcqa_gen_*, abstention files live in
# results/abstention/medmcqa_gen/, audit CSVs are audit_medmcqa_gen_*.csv, ladder
# reports ladder_medmcqa_gen_<split>.md, weights dirs contain "medmcqa_gen".
#
# COST (rough, single GPU): Stage-1 ~1.5-3 h/epoch; each FCVR run ~2-4 h/epoch
# (early stopping usually ends earlier); each --generate readout on 1,000 examples
# ~1-1.5 h (greedy, <=256 new tokens, S=35 routing is ~free). val+test at 2 betas x
# 3 seeds = 12 generate readouts. Use SEEDS="42" and/or N_VAL/N_TEST=500 to cut this.
#
# Usage on quail-1 (tmux):
#   source ~/.venvs/moe_env/bin/activate
#   pip install sacrebleu rouge-score nltk bert-score      # once (metric suite)
#   PHASES=preflight            bash run-iter1-granite-medmcqa.sh   # inspect results/data/medmcqa_gen_inspect.json
#   git add -f splits/medmcqa_gen-derived-seed42.csv                # freeze the split
#   PHASES=stage1,stage2        bash run-iter1-granite-medmcqa.sh
#   PHASES=val                  bash run-iter1-granite-medmcqa.sh   # inspect audit CSV + frozen files
#   PHASES=test                 bash run-iter1-granite-medmcqa.sh   # once
#   PHASES=baselines,prior,kl,ood,report bash run-iter1-granite-medmcqa.sh
#   SKIP_NLI=1 to skip the NLI judge; SEEDS="42" for one inference seed; ALLOW_EXISTING=1 to resume.
# ============================================================================

set -Eeo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_SHORTCODE="granite"
DATASET_SHORTCODE="medmcqa_gen"
SOURCE="medmcqa_gen"                                   # analyze_token_signals --source
KVQ_ADAPTER_PATH="./adapters/granite-medmcqa_gen"      # Stage-1 adapter (trained in phase stage1)
LAYERS=(5 6 7 8 19 20 28 29 30 31)                     # Susceptible-10
BETAS=(${BETAS:-0.01 0.1})                             # Stage-2 beta sweep (phase stage2)
SEEDS=(${SEEDS:-42 43 44})                             # inference seeds for stochastic readouts
S=35                                                   # MC samples (paper)
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"                # gold explanations are <=160 words -> fits
N_VAL="${N_VAL:-1000}"                                 # readout examples from the 1,000-row val split
N_TEST="${N_TEST:-1000}"                               # readout examples from the 1,000-row test split
N_OOD="${N_OOD:-500}"                                  # examples per dataset in the OoD bridge test
OOD_SETS=(obqa medexqa mmlu_law)                       # NEVER medmcqa_med here (same corpus)
MAP_SUFFIX="iter1"
PHASES="${PHASES:-preflight,stage1,stage2,val,test,baselines,prior,kl,ood,report}"
NLI_MODEL="${NLI_MODEL:-MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli}"
WANDB_PROJECT="${WANDB_PROJECT:-moe-uncertainty}"
ABST_DIR="results/abstention/${DATASET_SHORTCODE}"
TOK_DIR="results/token_analysis"
# Stage-2 HP (identical to fcvr-tuning-granite-medmcqa.sh)
S2_EPOCHS="${S2_EPOCHS:-5}"; S2_BS=4; S2_ACCUM=4; S2_LR=1e-4; S2_WARMUP=0.05
S2_PATIENCE="${S2_PATIENCE:-3}"; S2_EVAL_EVERY="${S2_EVAL_EVERY:-500}"; S2_MAX_SEQ_LEN=768; S2_SEED=42

RUN_TAG="$(date +%Y%m%d-%H%M%S)"
mkdir -p logs "$TOK_DIR" results/input_level_ood "$ABST_DIR" results/labels results/reports results/data
LOG="logs/iter1-${MODEL_SHORTCODE}-${DATASET_SHORTCODE}-${RUN_TAG}.log"
exec > >(tee -a "$LOG") 2>&1

echo "############################################################"
echo "# medmcqa_gen run ${RUN_TAG}   phases=${PHASES}"
echo "# Repo:   $REPO_ROOT"
echo "# Branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?')  Commit: $(git rev-parse --short HEAD 2>/dev/null || echo '?')"
echo "# Python: $(which python)"
echo "# Log:    $LOG"
echo "# N_VAL=${N_VAL} N_TEST=${N_TEST} N_OOD=${N_OOD} SEEDS=${SEEDS[*]} BETAS=${BETAS[*]} MAX_NEW_TOKENS=${MAX_NEW_TOKENS}"
echo "############################################################"

notify() {
    local level="$1" title="$2" text="$3"
    python - "$level" "$title" "$text" <<'PY' || echo "(W&B alert failed -- check WANDB_API_KEY)"
import os, sys
level, title, text = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    import wandb
    run = wandb.init(project=os.environ.get("WANDB_PROJECT", "moe-uncertainty"),
                     name=f"medmcqa-watchdog-{os.environ.get('RUN_TAG', '')}", job_type="alert", reinit=True)
    wandb.alert(title=title, text=text, level=wandb.AlertLevel.ERROR if level == "error" else wandb.AlertLevel.INFO)
    run.finish()
except Exception as e:
    print(f"W&B alert failed: {type(e).__name__}: {e}")
PY
}
export RUN_TAG WANDB_PROJECT
on_error() {
    local exit_code=$? line=$1
    echo "############################################################"
    echo "# FAILED at line ${line} (exit ${exit_code}) -- ${STEP:-unknown step}   log: ${LOG}"
    echo "############################################################"
    notify error "medmcqa_gen run FAILED: ${STEP:-unknown step}" "Host $(hostname), run ${RUN_TAG}, exit ${exit_code} at line ${line}. Log: ${LOG}"
    exit "$exit_code"
}
trap 'on_error $LINENO' ERR

has_phase() { [[ ",${PHASES}," == *",$1,"* ]]; }
NLI_FLAG=""; [ "${SKIP_NLI:-0}" = "1" ] && NLI_FLAG="--no_nli"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
# seq_file SPLIT TAG  -> labelled seqlevel path
seq_file() { local st=""; [ "$1" = "val" ] && st="_val"; echo "${TOK_DIR}/step1_${SOURCE}${st}_generate_$2_seqlevel_labeled.jsonl"; }

# readouts ARM SUFFIX PRIOR MAPSFX SPLIT SEED TAG   (SUFFIX/PRIOR ignored for det)
readouts() {
    local arm="$1" suffix="$2" prior="$3" mapsfx="$4" split="$5" seed="$6" tag="$7"
    local n; [ "$split" = "val" ] && n=$N_VAL || n=$N_TEST
    local common=(--model_shortcode "$MODEL_SHORTCODE" --dataset_shortcode "$DATASET_SHORTCODE"
                  --kvq_adapter_path "$KVQ_ADAPTER_PATH" --arm "$arm" --routing stochastic --num_samples "$S"
                  --split "$split" --num_examples "$n" --seed "$seed" --tag "$tag")
    if [ "$arm" != "det" ]; then
        common+=(--swap_layers "${LAYERS[@]}" --prior_source "$prior")
        [ "$arm" = "fcvr" ] && common+=(--run_suffix "$suffix")
        [ -n "$mapsfx" ] && common+=(--map_suffix "$mapsfx")
    fi
    if [ "$arm" != "det" ] && [ "$split" = "val" ] && [ "$seed" = "${SEEDS[0]}" ]; then
        STEP="teacher-forced readout ${tag} (${split})"
        python analyze_token_signals.py "${common[@]}" --source "$SOURCE"
    fi
    STEP="generate readout ${tag} (${split})"
    python analyze_token_signals.py "${common[@]}" --source "$SOURCE" --generate --max_new_tokens "$MAX_NEW_TOKENS"
    local split_tok=""; [ "$split" = "val" ] && split_tok="_val"
    local seq="${TOK_DIR}/step1_${SOURCE}${split_tok}_generate_${tag}_seqlevel.jsonl"
    STEP="labels ${tag} (${split})"
    local audit=(); [ "$split" = "val" ] && audit=(--audit_export 40)
    python label_generation_correctness.py --input "$seq" --nli_model "$NLI_MODEL" $NLI_FLAG "${audit[@]}"
}

# select_eval NAME SEED [PRIMARY_SCORE]  -> abstention select on val, evaluate on test (one seed)
select_eval() {
    local name="$1" seed="$2" primary="${3:-ilv_online_mean_last10}"
    STEP="abstention select/evaluate ${name}"
    python abstention_report.py --select --tag "${name}-S35" --input "$(seq_file val "$(tag_of "$name" "$seed")")" \
        --primary_score "$primary" --out_dir "$ABST_DIR"
    python abstention_report.py --evaluate --frozen "${ABST_DIR}/frozen_${name}-S35.json" --eval_tag "s${seed}" \
        --input "$(seq_file test "$(tag_of "$name" "$seed")")" --out_dir "$ABST_DIR"
}

# ood ARM SUFFIX PRIOR MAPSFX SPLIT TAG
ood() {
    local arm="$1" suffix="$2" prior="$3" mapsfx="$4" split="$5" tag="$6"
    local common=(--model_shortcode "$MODEL_SHORTCODE" --dataset_shortcode "$DATASET_SHORTCODE"
                  --kvq_adapter_path "$KVQ_ADAPTER_PATH" --arm "$arm" --routing stochastic --num_samples "$S"
                  --split "$split" --num_examples "$N_OOD" --seed 42 --ood_datasets "${OOD_SETS[@]}")
    if [ "$arm" != "det" ]; then
        common+=(--swap_layers "${LAYERS[@]}" --prior_source "$prior")
        [ "$arm" = "fcvr" ] && common+=(--run_suffix "$suffix")
        [ -n "$mapsfx" ] && common+=(--map_suffix "$mapsfx")
    fi
    STEP="ood canonical ${tag} (${split})"
    python fcvr_input_level_ood_check.py "${common[@]}" --inner_format generation --format_control --tag "${tag}"
    STEP="ood native ${tag} (${split})"
    python fcvr_input_level_ood_check.py "${common[@]}" --inner_format native --tag "${tag}-native"
}

# train_fcvr SUFFIX BETA PRIOR KLMASK MAPSFX
train_fcvr() {
    local suffix="$1" beta="$2" prior="$3" klmask="$4" mapsfx="$5"
    local w="./router_weights/fcvr/fcvr-${MODEL_SHORTCODE}-${DATASET_SHORTCODE}-${suffix}"
    if [ -e "$w" ]; then echo "  (skip train: $w exists)"; return; fi
    STEP="train fcvr ${suffix}"
    local extra=(); [ -n "$mapsfx" ] && extra=(--map_suffix "$mapsfx")
    python scripts/python/fcvr-tuning.py \
        --model_shortcode "$MODEL_SHORTCODE" --dataset_shortcode "$DATASET_SHORTCODE" \
        --base_adapter_path "$KVQ_ADAPTER_PATH" \
        --swap_layers "${LAYERS[@]}" --load_layers --train_layers "${LAYERS[@]}" \
        --epochs "$S2_EPOCHS" --batch_size "$S2_BS" --grad_accum_steps "$S2_ACCUM" --lr "$S2_LR" \
        --warmup_ratio "$S2_WARMUP" --early_stop_patience "$S2_PATIENCE" --eval_every "$S2_EVAL_EVERY" \
        --max_seq_len "$S2_MAX_SEQ_LEN" --beta "$beta" --seed "$S2_SEED" \
        --run_suffix "$suffix" --prior_source "$prior" --kl_mask "$klmask" "${extra[@]}"
}

# arm table for the ladder:  name | arm | suffix | prior | mapsfx
ARM_ROWS=(
  "beta0.01|fcvr|pretrained-prior-beta0.01|pretrained|"
  "beta0.1|fcvr|pretrained-prior-beta0.1|pretrained|"
  "det|det|||"
  "untrained|untrained||pretrained|"
  "beta0|fcvr|iter1-beta0|pretrained|"
  "mapprior-beta0.01|fcvr|iter1-map-prior-beta0.01|map|${MAP_SUFFIX}"
  "klnone-beta0.01|fcvr|iter1-klnone-beta0.01|pretrained|"
  "klans-beta0.01|fcvr|iter1-klans-beta0.01|pretrained|"
)
tag_of() { echo "$1-S35-s$2"; }

# ---------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------
if has_phase preflight; then
    STEP="preflight: deps"
    python - <<'PY'
import importlib
for m in ("sklearn", "scipy", "sacrebleu", "rouge_score", "nltk", "bert_score", "datasets"):
    try: importlib.import_module(m); print(f"  ok  {m}")
    except Exception as e: print(f"  MISSING {m}: {e!r}  (pip install sacrebleu rouge-score nltk bert-score scikit-learn scipy datasets)")
PY
    STEP="preflight: split manifest"
    python write-medmcqa-split-manifest.py
    STEP="preflight: data inspection"
    python medmcqa-gen-inspect.py --max_new_tokens "$MAX_NEW_TOKENS" --max_seq_len "$S2_MAX_SEQ_LEN"
    STEP="preflight: label mask check (medmcqa_gen)"
    python check-answer-only-labels.py --dataset_shortcode "$DATASET_SHORTCODE"
    STEP="preflight: inputs"
    # Only enforce trained inputs when a phase of THIS invocation consumes them
    # (PHASES=preflight alone on a fresh checkout must succeed before stage1 exists).
    NEEDS_WEIGHTS=0
    for ph in val test baselines prior kl ood report; do has_phase "$ph" && NEEDS_WEIGHTS=1; done
    if [ "$NEEDS_WEIGHTS" = 1 ] && ! has_phase stage1; then
        [ -d "$KVQ_ADAPTER_PATH" ] || { echo "ERROR: adapter $KVQ_ADAPTER_PATH missing (run PHASES=stage1)"; exit 1; }
    fi
    if [ "$NEEDS_WEIGHTS" = 1 ] && ! has_phase stage2 && ! has_phase stage1; then
        for B in "${BETAS[@]}"; do
            for L in "${LAYERS[@]}"; do
                f="./router_weights/fcvr/fcvr-${MODEL_SHORTCODE}-${DATASET_SHORTCODE}-pretrained-prior-beta${B}/layer_${L}_weights.pt"
                [ -f "$f" ] || { echo "ERROR: missing FCVR weights $f (run PHASES=stage2)"; exit 1; }
            done
        done
    fi
    STEP="preflight: collision check"
    COLL=()
    if has_phase stage1 && [ -e "$KVQ_ADAPTER_PATH" ]; then COLL+=("$KVQ_ADAPTER_PATH"); fi
    for row in "${ARM_ROWS[@]}"; do IFS='|' read -r name arm suffix prior mapsfx <<<"$row"
        for split in val test; do st=""; [ "$split" = "val" ] && st="_val"
            for seed in "${SEEDS[@]}"; do t=$(tag_of "$name" "$seed")
                for f in "${TOK_DIR}/step1_${SOURCE}${st}_generate_${t}.json" "${TOK_DIR}/step1_${SOURCE}${st}_teacher_forced_${t}.json"; do
                    [ -e "$f" ] && COLL+=("$f"); done
            done
        done
        case "$suffix" in iter1-*) w="./router_weights/fcvr/fcvr-${MODEL_SHORTCODE}-${DATASET_SHORTCODE}-${suffix}"; [ -e "$w" ] && COLL+=("$w");; esac
        case "$suffix" in pretrained-prior-*) if has_phase stage2; then w="./router_weights/fcvr/fcvr-${MODEL_SHORTCODE}-${DATASET_SHORTCODE}-${suffix}"; [ -e "$w" ] && COLL+=("$w"); fi;; esac
    done
    [ -e "./router_weights/base/${MODEL_SHORTCODE}_${DATASET_SHORTCODE}-${MAP_SUFFIX}" ] && COLL+=("./router_weights/base/${MODEL_SHORTCODE}_${DATASET_SHORTCODE}-${MAP_SUFFIX}")
    if [ ${#COLL[@]} -gt 0 ]; then
        echo "WARNING: these outputs already exist (a resumed run will SKIP training dirs but readouts would overwrite):"
        printf '  %s\n' "${COLL[@]}"
        if [ "${ALLOW_EXISTING:-0}" != "1" ]; then echo "Refusing. Set ALLOW_EXISTING=1 to resume (existing readout files WILL be overwritten by same-tag reruns)."; exit 1; fi
    fi
    echo "preflight OK -- remember: git add -f splits/medmcqa_gen-derived-seed42.csv"
fi

# ---------------------------------------------------------------------------
# stage1: Stage-1 KVQ adapter
# ---------------------------------------------------------------------------
if has_phase stage1; then
    if [ -d "$KVQ_ADAPTER_PATH" ]; then
        echo "  (skip stage1: $KVQ_ADAPTER_PATH exists)"
    else
        STEP="stage1 KVQ adapter"; bash kvq-tuning-granite-medmcqa.sh
    fi
    notify info "medmcqa_gen stage1 done" "Adapter at ${KVQ_ADAPTER_PATH}. Log: ${LOG}"
fi

# ---------------------------------------------------------------------------
# stage2: FCVR beta sweep (pretrained prior, KL mask attention)
# ---------------------------------------------------------------------------
if has_phase stage2; then
    [ -d "$KVQ_ADAPTER_PATH" ] || { echo "ERROR: adapter $KVQ_ADAPTER_PATH missing (run PHASES=stage1)"; exit 1; }
    for B in "${BETAS[@]}"; do train_fcvr "pretrained-prior-beta${B}" "$B" pretrained attention ""; done
    notify info "medmcqa_gen stage2 done" "FCVR betas ${BETAS[*]} trained. Log: ${LOG}"
fi

# ---------------------------------------------------------------------------
# val  (selection)  -- pretrained-prior weights, seeds x betas
# ---------------------------------------------------------------------------
if has_phase val; then
    for B in "${BETAS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            readouts fcvr "pretrained-prior-beta${B}" pretrained "" val "$seed" "$(tag_of "beta${B}" "$seed")"
        done
        STEP="abstention select beta${B}"
        python abstention_report.py --select --tag "beta${B}-S35" --out_dir "$ABST_DIR" \
            --input "$(seq_file val "$(tag_of "beta${B}" "${SEEDS[0]}")")"
    done
    echo ">>> VAL PHASE DONE. Inspect results/labels/audit_medmcqa_gen_*.csv (hand-label 25-50 rows, then"
    echo ">>> label_generation_correctness.py --audit_import) and ${ABST_DIR}/frozen_*.json BEFORE running PHASES=test."
    notify info "medmcqa_gen val phase done" "Inspect audit CSVs + frozen files, then run PHASES=test. Log: ${LOG}"
fi

# ---------------------------------------------------------------------------
# test  (one evaluation per frozen configuration)
# ---------------------------------------------------------------------------
if has_phase test; then
    for B in "${BETAS[@]}"; do
        [ -f "${ABST_DIR}/frozen_beta${B}-S35.json" ] || { echo "ERROR: run PHASES=val first (frozen_beta${B}-S35.json missing)"; exit 1; }
        for seed in "${SEEDS[@]}"; do
            t=$(tag_of "beta${B}" "$seed")
            readouts fcvr "pretrained-prior-beta${B}" pretrained "" test "$seed" "$t"
            STEP="abstention evaluate ${t}"
            python abstention_report.py --evaluate --frozen "${ABST_DIR}/frozen_beta${B}-S35.json" --eval_tag "s${seed}" \
                --input "$(seq_file test "$t")" --out_dir "$ABST_DIR"
        done
        STEP="abstention aggregate beta${B}"
        python abstention_report.py --aggregate "${ABST_DIR}/eval_beta${B}-S35_s*.json" --tag "beta${B}-S35" --out_dir "$ABST_DIR"
    done
fi

# ---------------------------------------------------------------------------
# baselines: det, untrained, beta=0
# ---------------------------------------------------------------------------
if has_phase baselines; then
    seed="${SEEDS[0]}"
    for split in val test; do
        readouts det "" "" "" "$split" "$seed" "$(tag_of det "$seed")"
        readouts untrained "" pretrained "" "$split" "$seed" "$(tag_of untrained "$seed")"
    done
    train_fcvr "iter1-beta0" 0.0 pretrained attention ""
    for split in val test; do readouts fcvr "iter1-beta0" pretrained "" "$split" "$seed" "$(tag_of beta0 "$seed")"; done
    select_eval det "$seed" entropy_max_BASELINE
    select_eval untrained "$seed"
    select_eval beta0 "$seed"
fi

# ---------------------------------------------------------------------------
# prior: MAP router stage -> FCVR map-prior
# ---------------------------------------------------------------------------
if has_phase prior; then
    seed="${SEEDS[0]}"
    if [ ! -d "./router_weights/base/${MODEL_SHORTCODE}_${DATASET_SHORTCODE}-${MAP_SUFFIX}" ]; then
        STEP="MAP router tuning"; MAP_SUFFIX="$MAP_SUFFIX" bash router-tuning-granite-medmcqa.sh
    fi
    train_fcvr "iter1-map-prior-beta0.01" 0.01 map attention "$MAP_SUFFIX"
    for split in val test; do readouts fcvr "iter1-map-prior-beta0.01" map "$MAP_SUFFIX" "$split" "$seed" "$(tag_of mapprior-beta0.01 "$seed")"; done
    select_eval mapprior-beta0.01 "$seed"
fi

# ---------------------------------------------------------------------------
# kl: KL-mask ablations (main arm = attention, trained in stage2)
# ---------------------------------------------------------------------------
if has_phase kl; then
    seed="${SEEDS[0]}"
    train_fcvr "iter1-klnone-beta0.01" 0.01 pretrained none ""
    train_fcvr "iter1-klans-beta0.01" 0.01 pretrained answer ""
    for name in klnone-beta0.01 klans-beta0.01; do
        for split in val test; do readouts fcvr "iter1-${name}" pretrained "" "$split" "$seed" "$(tag_of "$name" "$seed")"; done
        select_eval "$name" "$seed"
    done
fi

# ---------------------------------------------------------------------------
# ood: every arm whose weights exist, val + test
# ---------------------------------------------------------------------------
if has_phase ood; then
    for row in "${ARM_ROWS[@]}"; do IFS='|' read -r name arm suffix prior mapsfx <<<"$row"
        if [ "$arm" = "fcvr" ] && [ ! -d "./router_weights/fcvr/fcvr-${MODEL_SHORTCODE}-${DATASET_SHORTCODE}-${suffix}" ]; then
            echo "  (ood skip ${name}: weights not trained)"; continue; fi
        for split in val test; do ood "$arm" "$suffix" "$prior" "$mapsfx" "$split" "$(tag_of "$name" "${SEEDS[0]}")"; done
    done
fi

# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
if has_phase report; then
    STEP="ladder report"
    TAGS=(); for row in "${ARM_ROWS[@]}"; do IFS='|' read -r name _ <<<"$row"; TAGS+=("$(tag_of "$name" "${SEEDS[0]}")"); done
    for split in val test; do
        python baseline_ladder_report.py --source "$SOURCE" --split "$split" --tags "${TAGS[@]}" || echo "(ladder ${split}: some arms missing)"
    done
fi

echo "############################################################"
echo "# medmcqa_gen run ${RUN_TAG} finished phases=${PHASES}   log: ${LOG}"
echo "############################################################"
notify info "medmcqa_gen run finished (${PHASES})" "Host $(hostname), run ${RUN_TAG}. Log: ${LOG}"

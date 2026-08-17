#!/bin/bash
# ============================================================================
# codex-recom-iter1 driver -- Granite-MoE / MedExQA (supervisor iteration 1)
#
# One resumable driver for every GPU run of the iteration. Phases (env PHASES,
# comma-separated, default all in this order):
#
#   preflight  adapters/weights/deps present; label-mask check (medexqa);
#              split manifest written/verified; up-front collision check.
#   val        SELECTION runs on the 50 val examples: stochastic S=35 readouts
#              (teacher-forced + --generate mnt256) x seeds {42,43,44} for the
#              existing beta0.01 / beta0.1 weights -> labels (NLI + metrics)
#              -> audit CSV (hand-label 25-50 rows) -> abstention_report --select
#              (freezes sign check / thresholds / val-fitted models).
#   test       ONE evaluation per frozen configuration on the 175 test examples
#              (same readouts, same seeds) -> labels -> abstention_report --evaluate
#              -> --aggregate across seeds.  Never run before `val` is inspected.
#   baselines  det arm (Stage-1 stock routers) and untrained-FCVR arm on val+test;
#              trains beta=0 (pretrained prior, kl_mask attention).
#   prior      Stage-2a MAP router tuning (router-tuning-granite-medexqa.sh) ->
#              FCVR iter1-map-prior-beta0.01 -> val+test readouts (1 seed).
#   kl         FCVR iter1-klattn-beta0.01 and iter1-klans-beta0.01 -> val+test.
#   ood        input-level bridge test on every arm (canonical template +
#              format control + native comparison), val and test.
#   report     baseline_ladder_report.py for val and test.
#
# Every output path is checked for collisions BEFORE any GPU time is spent;
# nothing here overwrites an existing adapter / weight dir / result file. All
# new artefacts carry an `iter1` run suffix or an `-S35-s<seed>` tag; the
# existing new-exp2 files (results/token_analysis/step1_medexqa_generate_beta*,
# router_weights/fcvr/fcvr-granite-medexqa-pretrained-prior-beta*) are inputs.
#
# Hyperparameters: Stage-2 identical to fcvr-tuning-granite-medexqa.sh
# (4 ep, bs 4 x accum 4, lr 1e-4, warmup .05, patience 2, seed 42); MAP stage
# 3 ep / bs 4 / lr 1e-4 / patience 2 (user decision 2026-08-17).
#
# Usage on quail-1 (tmux):
#   source ~/.venvs/moe_env/bin/activate
#   pip install sacrebleu rouge-score nltk bert-score      # once (metric suite)
#   PHASES=preflight,val bash run-iter1-granite-medexqa.sh # inspect audit CSV + frozen files
#   PHASES=test          bash run-iter1-granite-medexqa.sh # then the rest:
#   PHASES=baselines,prior,kl,ood,report bash run-iter1-granite-medexqa.sh
#   SKIP_NLI=1 to skip the NLI judge; SEEDS="42" to run one inference seed.
# ============================================================================

set -Eeo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_SHORTCODE="granite"
DATASET_SHORTCODE="medexqa"
KVQ_ADAPTER_PATH="./adapters/granite-medexqa"          # existing Stage-1 adapter (input)
LAYERS=(5 6 7 8 19 20 28 29 30 31)                     # Susceptible-10
BETAS=(0.01 0.1)                                       # existing pretrained-prior weights
SEEDS=(${SEEDS:-42 43 44})                             # inference seeds for stochastic readouts
S=35                                                   # MC samples (paper)
MAX_NEW_TOKENS=256
N_VAL=50
N_TEST=175
MAP_SUFFIX="iter1"
PHASES="${PHASES:-preflight,val,test,baselines,prior,kl,ood,report}"
NLI_MODEL="${NLI_MODEL:-MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli}"
WANDB_PROJECT="${WANDB_PROJECT:-moe-uncertainty}"
# Stage-2 HP (identical to fcvr-tuning-granite-medexqa.sh)
S2_EPOCHS=4; S2_BS=4; S2_ACCUM=4; S2_LR=1e-4; S2_WARMUP=0.05; S2_PATIENCE=2; S2_SEED=42

RUN_TAG="$(date +%Y%m%d-%H%M%S)"
mkdir -p logs results/token_analysis results/input_level_ood results/abstention results/labels results/reports
LOG="logs/iter1-${MODEL_SHORTCODE}-${DATASET_SHORTCODE}-${RUN_TAG}.log"
exec > >(tee -a "$LOG") 2>&1

echo "############################################################"
echo "# iter1 run ${RUN_TAG}   phases=${PHASES}"
echo "# Repo:   $REPO_ROOT"
echo "# Branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?')  Commit: $(git rev-parse --short HEAD 2>/dev/null || echo '?')"
echo "# Python: $(which python)"
echo "# Log:    $LOG"
echo "############################################################"

notify() {
    local level="$1" title="$2" text="$3"
    python - "$level" "$title" "$text" <<'PY' || echo "(W&B alert failed -- check WANDB_API_KEY)"
import os, sys
level, title, text = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    import wandb
    run = wandb.init(project=os.environ.get("WANDB_PROJECT", "moe-uncertainty"),
                     name=f"iter1-watchdog-{os.environ.get('RUN_TAG', '')}", job_type="alert", reinit=True)
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
    notify error "iter1 run FAILED: ${STEP:-unknown step}" "Host $(hostname), run ${RUN_TAG}, exit ${exit_code} at line ${line}. Log: ${LOG}"
    exit "$exit_code"
}
trap 'on_error $LINENO' ERR

has_phase() { [[ ",${PHASES}," == *",$1,"* ]]; }
NLI_FLAG=""; [ "${SKIP_NLI:-0}" = "1" ] && NLI_FLAG="--no_nli"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
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
        python analyze_token_signals.py "${common[@]}" --source medexqa
    fi
    STEP="generate readout ${tag} (${split})"
    python analyze_token_signals.py "${common[@]}" --source medexqa --generate --max_new_tokens "$MAX_NEW_TOKENS"
    local split_tok=""; [ "$split" = "val" ] && split_tok="_val"
    local seq="results/token_analysis/step1_medexqa${split_tok}_generate_${tag}_seqlevel.jsonl"
    STEP="labels ${tag} (${split})"
    local audit=(); [ "$split" = "val" ] && audit=(--audit_export 40)
    python label_generation_correctness.py --input "$seq" --nli_model "$NLI_MODEL" $NLI_FLAG "${audit[@]}"
}

# ood ARM SUFFIX PRIOR MAPSFX SPLIT TAG
ood() {
    local arm="$1" suffix="$2" prior="$3" mapsfx="$4" split="$5" tag="$6"
    local n; [ "$split" = "val" ] && n=$N_VAL || n=$N_TEST
    local common=(--model_shortcode "$MODEL_SHORTCODE" --dataset_shortcode "$DATASET_SHORTCODE"
                  --kvq_adapter_path "$KVQ_ADAPTER_PATH" --arm "$arm" --routing stochastic --num_samples "$S"
                  --split "$split" --num_examples "$n" --seed 42 --ood_datasets obqa medmcqa_med mmlu_law)
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
        --warmup_ratio "$S2_WARMUP" --early_stop_patience "$S2_PATIENCE" --beta "$beta" --seed "$S2_SEED" \
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
  "klattn-beta0.01|fcvr|iter1-klattn-beta0.01|pretrained|"
  "klans-beta0.01|fcvr|iter1-klans-beta0.01|pretrained|"
)
tag_of() { echo "$1-S35-s$2"; }

# ---------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------
if has_phase preflight; then
    STEP="preflight: inputs"
    [ -d "$KVQ_ADAPTER_PATH" ] || { echo "ERROR: adapter $KVQ_ADAPTER_PATH missing"; exit 1; }
    for B in "${BETAS[@]}"; do
        for L in "${LAYERS[@]}"; do
            f="./router_weights/fcvr/fcvr-${MODEL_SHORTCODE}-${DATASET_SHORTCODE}-pretrained-prior-beta${B}/layer_${L}_weights.pt"
            [ -f "$f" ] || { echo "ERROR: missing FCVR weights $f"; exit 1; }
        done
    done
    STEP="preflight: deps"
    python - <<'PY'
import importlib
for m in ("sklearn", "scipy", "sacrebleu", "rouge_score", "nltk", "bert_score"):
    try: importlib.import_module(m); print(f"  ok  {m}")
    except Exception as e: print(f"  MISSING {m}: {e!r}  (pip install sacrebleu rouge-score nltk bert-score scikit-learn scipy)")
PY
    STEP="preflight: split manifest"
    python write-medexqa-split-manifest.py
    STEP="preflight: label mask check (medexqa)"
    python check-answer-only-labels.py --dataset_shortcode medexqa
    STEP="preflight: collision check"
    COLL=()
    for row in "${ARM_ROWS[@]}"; do IFS='|' read -r name arm suffix prior mapsfx <<<"$row"
        for split in val test; do st=""; [ "$split" = "val" ] && st="_val"
            for seed in "${SEEDS[@]}"; do t=$(tag_of "$name" "$seed")
                for f in "results/token_analysis/step1_medexqa${st}_generate_${t}.json" "results/token_analysis/step1_medexqa${st}_teacher_forced_${t}.json"; do
                    [ -e "$f" ] && COLL+=("$f"); done
            done
        done
        case "$suffix" in iter1-*) w="./router_weights/fcvr/fcvr-${MODEL_SHORTCODE}-${DATASET_SHORTCODE}-${suffix}"; [ -e "$w" ] && COLL+=("$w");; esac
    done
    [ -e "./router_weights/base/${MODEL_SHORTCODE}_${DATASET_SHORTCODE}-${MAP_SUFFIX}" ] && COLL+=("./router_weights/base/${MODEL_SHORTCODE}_${DATASET_SHORTCODE}-${MAP_SUFFIX}")
    if [ ${#COLL[@]} -gt 0 ]; then
        echo "WARNING: these outputs already exist (a resumed run will SKIP training dirs but readouts would overwrite):"
        printf '  %s\n' "${COLL[@]}"
        if [ "${ALLOW_EXISTING:-0}" != "1" ]; then echo "Refusing. Set ALLOW_EXISTING=1 to resume (existing readout files WILL be overwritten by same-tag reruns)."; exit 1; fi
    fi
    echo "preflight OK"
fi

# ---------------------------------------------------------------------------
# val  (selection)  -- existing pretrained-prior weights, seeds x betas
# ---------------------------------------------------------------------------
if has_phase val; then
    for B in "${BETAS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            readouts fcvr "pretrained-prior-beta${B}" pretrained "" val "$seed" "$(tag_of "beta${B}" "$seed")"
        done
        STEP="abstention select beta${B}"
        python abstention_report.py --select --tag "beta${B}-S35" \
            --input "results/token_analysis/step1_medexqa_val_generate_$(tag_of "beta${B}" "${SEEDS[0]}")_seqlevel_labeled.jsonl"
    done
    echo ">>> VAL PHASE DONE. Inspect results/labels/audit_*.csv (hand-label 25-50 rows, then"
    echo ">>> label_generation_correctness.py --audit_import) and results/abstention/frozen_*.json BEFORE running PHASES=test."
    notify info "iter1 val phase done" "Inspect audit CSVs + frozen files, then run PHASES=test. Log: ${LOG}"
fi

# ---------------------------------------------------------------------------
# test  (one evaluation per frozen configuration)
# ---------------------------------------------------------------------------
if has_phase test; then
    for B in "${BETAS[@]}"; do
        [ -f "results/abstention/frozen_beta${B}-S35.json" ] || { echo "ERROR: run PHASES=val first (frozen_beta${B}-S35.json missing)"; exit 1; }
        for seed in "${SEEDS[@]}"; do
            t=$(tag_of "beta${B}" "$seed")
            readouts fcvr "pretrained-prior-beta${B}" pretrained "" test "$seed" "$t"
            STEP="abstention evaluate ${t}"
            python abstention_report.py --evaluate --frozen "results/abstention/frozen_beta${B}-S35.json" --eval_tag "s${seed}" \
                --input "results/token_analysis/step1_medexqa_generate_${t}_seqlevel_labeled.jsonl"
        done
        STEP="abstention aggregate beta${B}"
        python abstention_report.py --aggregate "results/abstention/eval_beta${B}-S35_s*.json" --tag "beta${B}-S35"
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
    for name in det untrained beta0; do
        STEP="abstention select/evaluate ${name}"
        python abstention_report.py --select --tag "${name}-S35" --input "results/token_analysis/step1_medexqa_val_generate_$(tag_of "$name" "$seed")_seqlevel_labeled.jsonl" \
            --primary_score "$([ "$name" = det ] && echo entropy_max_BASELINE || echo ilv_online_mean_last10)"
        python abstention_report.py --evaluate --frozen "results/abstention/frozen_${name}-S35.json" --eval_tag "s${seed}" \
            --input "results/token_analysis/step1_medexqa_generate_$(tag_of "$name" "$seed")_seqlevel_labeled.jsonl"
    done
fi

# ---------------------------------------------------------------------------
# prior: MAP router stage -> FCVR map-prior
# ---------------------------------------------------------------------------
if has_phase prior; then
    seed="${SEEDS[0]}"
    if [ ! -d "./router_weights/base/${MODEL_SHORTCODE}_${DATASET_SHORTCODE}-${MAP_SUFFIX}" ]; then
        STEP="MAP router tuning"; MAP_SUFFIX="$MAP_SUFFIX" bash router-tuning-granite-medexqa.sh
    fi
    train_fcvr "iter1-map-prior-beta0.01" 0.01 map attention "$MAP_SUFFIX"
    for split in val test; do readouts fcvr "iter1-map-prior-beta0.01" map "$MAP_SUFFIX" "$split" "$seed" "$(tag_of mapprior-beta0.01 "$seed")"; done
    STEP="abstention select/evaluate mapprior"
    python abstention_report.py --select --tag "mapprior-beta0.01-S35" --input "results/token_analysis/step1_medexqa_val_generate_$(tag_of mapprior-beta0.01 "$seed")_seqlevel_labeled.jsonl"
    python abstention_report.py --evaluate --frozen "results/abstention/frozen_mapprior-beta0.01-S35.json" --eval_tag "s${seed}" \
        --input "results/token_analysis/step1_medexqa_generate_$(tag_of mapprior-beta0.01 "$seed")_seqlevel_labeled.jsonl"
fi

# ---------------------------------------------------------------------------
# kl: attention-masked and answer-only KL
# ---------------------------------------------------------------------------
if has_phase kl; then
    seed="${SEEDS[0]}"
    train_fcvr "iter1-klattn-beta0.01" 0.01 pretrained attention ""
    train_fcvr "iter1-klans-beta0.01" 0.01 pretrained answer ""
    for name in klattn-beta0.01 klans-beta0.01; do
        for split in val test; do readouts fcvr "iter1-${name}" pretrained "" "$split" "$seed" "$(tag_of "$name" "$seed")"; done
        STEP="abstention select/evaluate ${name}"
        python abstention_report.py --select --tag "${name}-S35" --input "results/token_analysis/step1_medexqa_val_generate_$(tag_of "$name" "$seed")_seqlevel_labeled.jsonl"
        python abstention_report.py --evaluate --frozen "results/abstention/frozen_${name}-S35.json" --eval_tag "s${seed}" \
            --input "results/token_analysis/step1_medexqa_generate_$(tag_of "$name" "$seed")_seqlevel_labeled.jsonl"
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
    for split in val test; do python baseline_ladder_report.py --split "$split" --tags "${TAGS[@]}" || echo "(ladder ${split}: some arms missing)"; done
fi

echo "############################################################"
echo "# iter1 run ${RUN_TAG} finished phases=${PHASES}   log: ${LOG}"
echo "############################################################"
notify info "iter1 run finished (${PHASES})" "Host $(hostname), run ${RUN_TAG}. Log: ${LOG}"

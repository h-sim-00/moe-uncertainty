#!/bin/bash
# ============================================================================
# OoD detection from ILV at the answer position AND over the explanation
# (teacher-forced gold vs model-generated), arm A vs arm B, ID = OBQA
# (branch OBQA-comparison). Drives evaluate_ood_expl_readout.py, one stage per
# invocation, each with its own output stem under results/ood_expl_readout/:
#
#   stage1  final-position ilv_last on arc_c arc_e medexqa (paper protocol; must
#           reproduce results/ilv_ood_arms/obqa-arms_test_data-s42_mc-s42.*)
#   tf      teacher-forced explanation trace on medexqa scienceqa ecqa aqua_rat
#           (+ the final-position read-out on the same rows)
#   gen     generated-explanation trace on the same rows (predicted letter +
#           forced "\nExplanation:" + greedy, <= MAX_NEW_TOKENS); computes the
#           gen-tf paired deltas if the tf per-example file exists
#
# READ-ONLY w.r.t. every trained artefact (adapters/, router_weights/); the
# python script refuses to overwrite its own outputs unless ALLOW_EXISTING=1.
#
# Usage on quail-1 (venv moe_env; tmux so an ssh drop does not kill it):
#     tmux new -s ood-expl
#     SMOKE=1 bash run-ood-expl-readout.sh        # ~10 min end-to-end sanity run
#     bash run-ood-expl-readout.sh                # full: stage1 -> tf -> gen
#   after a crash:   RESUME=1 bash run-ood-expl-readout.sh   (skips finished stages)
#
# Env knobs (all optional):
#   STAGES="stage1,tf,gen"   SPLIT=test   N_PER_DOMAIN=500  TRACE_N_ID=500  TRACE_N_OOD=500
#   MAX_NEW_TOKENS=256  MAX_SEQ_TOKENS=2048  PERTOKEN_SCOPE=target|all
#   DATA_SEED=42  SAMPLING_SEED=42  S=35  EVAL_BATCH=8  N_BOOT=2000  ROUTING=stochastic
#   LAYERS="5 6 7 8 19 20 28 29 30 31"  TAG=obqa-ood-expl  OUT_DIR=results/ood_expl_readout
#   STAGE1_OOD="arc_c arc_e medexqa"  TRACE_OOD="medexqa scienceqa ecqa aqua_rat"
#   CROSSCHECK_STAGE1=<saved ilv_ood_arms perexample.jsonl>  CROSSCHECK_TF=<saved analyze_obqa_gen trace perexample.jsonl>
#   SMOKE=1 (tiny caps, tag smoke, plus a deterministic-routing gen alignment check)
#   RESUME=1  ALLOW_EXISTING=1  MOE_RAW_DATA_DIR=<dir for the ECQA/CommonsenseQA raw files>
#   WANDB_API_KEY=...  WANDB_PROJECT=moe-uncertainty  (only for the failure alert)
# ============================================================================
set -Eeo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
# source ~/.venvs/moe_env/bin/activate   # activate the venv BEFORE running (kept as a reminder)

STAGES="${STAGES:-stage1,tf,gen}"
SPLIT="${SPLIT:-test}"
N_PER_DOMAIN="${N_PER_DOMAIN:-500}"; TRACE_N_ID="${TRACE_N_ID:-500}"; TRACE_N_OOD="${TRACE_N_OOD:-500}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"; MAX_SEQ_TOKENS="${MAX_SEQ_TOKENS:-2048}"
PERTOKEN_SCOPE="${PERTOKEN_SCOPE:-target}"
DATA_SEED="${DATA_SEED:-42}"; SAMPLING_SEED="${SAMPLING_SEED:-42}"; S="${S:-35}"
EVAL_BATCH="${EVAL_BATCH:-8}"; N_BOOT="${N_BOOT:-2000}"; ROUTING="${ROUTING:-stochastic}"
read -r -a LAYERS <<< "${LAYERS:-5 6 7 8 19 20 28 29 30 31}"
TAG="${TAG:-obqa-ood-expl}"; OUT_DIR="${OUT_DIR:-results/ood_expl_readout}"
read -r -a STAGE1_OOD <<< "${STAGE1_OOD:-arc_c arc_e medexqa}"
read -r -a TRACE_OOD <<< "${TRACE_OOD:-medexqa scienceqa ecqa aqua_rat}"

if [ "${SMOKE:-0}" = "1" ]; then
    N_PER_DOMAIN=12; TRACE_N_ID=6; TRACE_N_OOD=4; N_BOOT=50; MAX_NEW_TOKENS=24; TAG="smoke"
fi

WANDB_PROJECT="${WANDB_PROJECT:-moe-uncertainty}"; RUN_TAG="$(date +%Y%m%d-%H%M%S)"
export RUN_TAG WANDB_PROJECT
mkdir -p logs "$OUT_DIR"
LOG="logs/ood-expl-readout-${RUN_TAG}.log"; exec > >(tee -a "$LOG") 2>&1
# shellcheck source=medmcqa-arms-lib.sh
source "$REPO_ROOT/medmcqa-arms-lib.sh"      # notify / on_error / skip_done / has_phase
PHASES="$STAGES"
trap 'on_error $LINENO' ERR

stem() { echo "${OUT_DIR}/${TAG}_${SPLIT}_data-s${DATA_SEED}_mc-s${SAMPLING_SEED}_$1"; }   # <stage>

common=(--split "$SPLIT" --n_per_domain "$N_PER_DOMAIN" --trace_n_id "$TRACE_N_ID" --trace_n_ood "$TRACE_N_OOD"
        --max_new_tokens "$MAX_NEW_TOKENS" --max_seq_tokens "$MAX_SEQ_TOKENS" --pertoken_scope "$PERTOKEN_SCOPE"
        --data_seed "$DATA_SEED" --sampling_seed "$SAMPLING_SEED" --num_samples "$S" --batch_size "$EVAL_BATCH"
        --n_boot "$N_BOOT" --routing "$ROUTING" --swap_layers "${LAYERS[@]}" --output_dir "$OUT_DIR" --tag "$TAG")
OW=(); [ "${ALLOW_EXISTING:-0}" = "1" ] && OW=(--overwrite)

echo "############################################################"
echo "# OoD explanation read-out ${RUN_TAG}  branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?') commit=$(git rev-parse --short HEAD 2>/dev/null || echo '?')"
echo "# stages=$STAGES split=$SPLIT n_per_domain=$N_PER_DOMAIN trace_n=$TRACE_N_ID/$TRACE_N_OOD max_new_tokens=$MAX_NEW_TOKENS"
echo "# data_seed=$DATA_SEED sampling_seed=$SAMPLING_SEED S=$S routing=$ROUTING layers=${LAYERS[*]} tag=$TAG"
echo "# stage1 OoD: ${STAGE1_OOD[*]} | trace OoD: ${TRACE_OOD[*]} | SMOKE=${SMOKE:-0} RESUME=${RESUME:-0} ALLOW_EXISTING=${ALLOW_EXISTING:-0}"
echo "# python=$(which python) | log: $LOG"
echo "############################################################"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

# ---- stage1 ------------------------------------------------------------------
if has_phase stage1; then
    STEP="stage1"; echo ""; echo "==== stage1 (final-position ILV: ${STAGE1_OOD[*]}) -- $(date) ===="
    if ! skip_done "$(stem stage1).json"; then
        XC=(); [ -n "${CROSSCHECK_STAGE1:-}" ] && XC=(--crosscheck "$CROSSCHECK_STAGE1")
        python evaluate_ood_expl_readout.py --stage stage1 --ood_datasets "${STAGE1_OOD[@]}" "${common[@]}" "${OW[@]}" "${XC[@]}"
    fi
fi

# ---- tf ----------------------------------------------------------------------
if has_phase tf; then
    STEP="tf"; echo ""; echo "==== tf (teacher-forced explanation trace: ${TRACE_OOD[*]}) -- $(date) ===="
    if ! skip_done "$(stem tf).json"; then
        XC=(); [ -n "${CROSSCHECK_TF:-}" ] && XC=(--crosscheck "$CROSSCHECK_TF")
        python evaluate_ood_expl_readout.py --stage tf --ood_datasets "${TRACE_OOD[@]}" "${common[@]}" "${OW[@]}" "${XC[@]}"
    fi
fi

# ---- gen ---------------------------------------------------------------------
if has_phase gen; then
    STEP="gen"; echo ""; echo "==== gen (generated explanation trace: ${TRACE_OOD[*]}) -- $(date) ===="
    if ! skip_done "$(stem gen).json"; then
        python evaluate_ood_expl_readout.py --stage gen --ood_datasets "${TRACE_OOD[@]}" "${common[@]}" "${OW[@]}"
    fi
    if [ "${SMOKE:-0}" = "1" ]; then
        # Alignment check: under posterior-mean routing the online (decoding-time)
        # and post-hoc ILV must agree on the explanation rows.
        STEP="gen-deterministic-check"; echo ""; echo "==== gen alignment check (deterministic routing) -- $(date) ===="
        python evaluate_ood_expl_readout.py --stage gen --ood_datasets "${TRACE_OOD[0]}" "${common[@]}" \
            --routing deterministic --tag "${TAG}-det" "${OW[@]}"
        python - "$(stem gen | sed "s/${TAG}_/${TAG}-det_/")_perexample.jsonl" <<'PY'
import json, sys, math
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
rows = [r for r in rows if r.get("kind") != "readout" and r.get("n_expl_tokens", 0) > 0]
bad = [r for r in rows if not r.get("online_ok")]
d = [r["online_posthoc_expl_maxabsdiff"] for r in rows if r.get("online_posthoc_expl_maxabsdiff") is not None
     and not math.isnan(r["online_posthoc_expl_maxabsdiff"])]
print(f"deterministic-routing alignment: {len(rows)} traced rows, online_ok failures={len(bad)}, "
      f"max |ilv_online - ilv_posthoc| over explanation rows = {max(d) if d else float('nan'):.4g}")
if bad or (d and max(d) > 1e-2):
    raise SystemExit("ALIGNMENT CHECK FAILED: online recorder rows do not match the post-hoc forward")
print("alignment OK")
PY
    fi
fi

echo ""; echo "==== all requested stages done -- $(date) ===="
ls -la "$OUT_DIR" | grep "${TAG}_" || true
notify info "ood-expl-readout finished" "Host $(hostname), run ${RUN_TAG}, stages ${STAGES}. Log: ${LOG}" || true

#!/bin/bash
# ============================================================================
# MedMCQA end-to-end sequence wrapper (branch MedMCQA) -- drives
# run-iter1-granite-medmcqa.sh through the mandated run order on quail-1:
#
#   1. preflight     PHASES=preflight            (deps, split manifest, data inspection,
#                                                 label-mask check, collision check)
#   2. freeze-split  git add -f splits/medmcqa_gen-derived-seed42.csv   (NO commit -- you do that)
#   3. train         PHASES=stage1,stage2        (KVQ adapter, FCVR beta sweep)
#   4. val           PHASES=val                  (selection readouts, audit CSVs, frozen files)
#      >>> the script STOPS here: hand-label the audit CSV(s), then re-run this script <<<
#   5. audit-import  label_generation_correctness.py --audit_import for every audit CSV
#                    that has human labels (re-labels the same file with --overwrite; labels are
#                    deterministic, only the "audit" block is added to the summary)
#   6. test          PHASES=test                 (ONE evaluation per frozen configuration)
#   7. rest          PHASES=baselines,prior,kl,ood,report
#
# Progress is checkpointed in logs/medmcqa-sequence.state (one completed step per line);
# re-running the script skips finished steps, so after the val stop you just run it again.
#
# Usage on quail-1 (tmux, moe_env):
#   bash run-medmcqa-sequence.sh                       # runs 1-4, stops for hand-labelling
#   ... fill human_* columns in results/labels/audit_medmcqa_gen_val_generate_*.csv ...
#   bash run-medmcqa-sequence.sh                       # runs 5-7
#
# Knobs (all optional, exported to the driver):
#   SEEDS="42" N_VAL=500 N_TEST=500 SKIP_NLI=1 BETAS="0.01 0.1" ...   as in the driver
#   FROM=<step>        force start at this step (ignore state for it and later steps)
#   STOP_AFTER=<step>  stop after this step (default: stops after 'val' on the first pass)
#   SKIP_AUDIT=1       proceed to test even if no audit CSV carries hand labels (NOT recommended)
#   RESET=1            wipe the state file and start from step 1
# ============================================================================

set -Eeo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

DRIVER="run-iter1-granite-medmcqa.sh"
SPLIT_CSV="splits/medmcqa_gen-derived-seed42.csv"
AUDIT_GLOB="results/labels/audit_medmcqa_gen_val_generate_*.csv"
TOK_DIR="results/token_analysis"
NLI_MODEL="${NLI_MODEL:-MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli}"
STATE="logs/medmcqa-sequence.state"
STEPS=(preflight freeze-split train val audit-import test rest)

mkdir -p logs
[ "${RESET:-0}" = "1" ] && rm -f "$STATE"
touch "$STATE"
LOG="logs/medmcqa-sequence-$(date +%Y%m%d-%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "############################################################"
echo "# medmcqa sequence   branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?') commit=$(git rev-parse --short HEAD 2>/dev/null || echo '?')"
echo "# state: $STATE   done so far: [$(paste -sd, "$STATE" 2>/dev/null)]"
echo "# log:   $LOG"
echo "############################################################"

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
step_index() { local i; for i in "${!STEPS[@]}"; do [ "${STEPS[$i]}" = "$1" ] && { echo "$i"; return; }; done; echo "ERROR: unknown step '$1' (valid: ${STEPS[*]})" >&2; exit 2; }
is_done() { grep -qx "$1" "$STATE"; }
mark_done() { is_done "$1" || echo "$1" >> "$STATE"; }
FROM_IDX=-1; [ -n "${FROM:-}" ] && FROM_IDX=$(step_index "$FROM")
STOP_IDX=99; [ -n "${STOP_AFTER:-}" ] && STOP_IDX=$(step_index "$STOP_AFTER")

# should_run STEP -> 0 if the step must execute now
should_run() {
    local idx; idx=$(step_index "$1")
    if [ "$FROM_IDX" -ge 0 ] && [ "$idx" -ge "$FROM_IDX" ]; then return 0; fi   # forced from here on
    if [ "$FROM_IDX" -ge 0 ] && [ "$idx" -lt "$FROM_IDX" ]; then return 1; fi   # before FROM: treat as done
    is_done "$1" && { echo ">>> skip $1 (already done; RESET=1 or FROM=$1 to redo)"; return 1; }
    return 0
}
finish_step() {
    mark_done "$1"
    local idx; idx=$(step_index "$1")
    if [ "$idx" -ge "$STOP_IDX" ]; then echo ">>> STOP_AFTER=$STOP_AFTER reached. Log: $LOG"; exit 0; fi
}
banner() { echo; echo "============================================================"; echo "== STEP: $1"; echo "============================================================"; }
run_driver() { echo "+ PHASES=$1 bash $DRIVER"; PHASES="$1" bash "$DRIVER"; }

on_error() {
    local ec=$? line=$1
    echo "############################################################"
    echo "# SEQUENCE FAILED at line $line (exit $ec) during step '${CUR:-?}'.  Log: $LOG"
    echo "# Fix the issue and re-run: bash run-medmcqa-sequence.sh   (finished steps are skipped)"
    echo "############################################################"
    exit "$ec"
}
trap 'on_error $LINENO' ERR

# ---------------------------------------------------------------------------
# 1. preflight
# ---------------------------------------------------------------------------
CUR=preflight
if should_run preflight; then
    banner preflight
    run_driver preflight
    [ -f "$SPLIT_CSV" ] || { echo "ERROR: preflight did not produce $SPLIT_CSV"; exit 1; }
    echo ">>> preflight OK. Data inspection: results/data/medmcqa_gen_inspect.json"
    finish_step preflight
fi

# ---------------------------------------------------------------------------
# 2. freeze the split manifest (git add -f; committing is left to you)
# ---------------------------------------------------------------------------
CUR=freeze-split
if should_run freeze-split; then
    banner freeze-split
    [ -f "$SPLIT_CSV" ] || { echo "ERROR: $SPLIT_CSV missing -- run FROM=preflight"; exit 1; }
    echo "+ git add -f $SPLIT_CSV"
    git add -f "$SPLIT_CSV"
    git status --short -- "$SPLIT_CSV"
    echo ">>> split manifest staged. Remember to commit it (git commit -m 'medmcqa_gen: freeze derived split (seed 42)')."
    finish_step freeze-split
fi

# ---------------------------------------------------------------------------
# 3. train: stage1 (KVQ adapter) + stage2 (FCVR beta sweep)
# ---------------------------------------------------------------------------
CUR=train
if should_run train; then
    banner "train (stage1,stage2)"
    run_driver stage1,stage2
    finish_step train
fi

# ---------------------------------------------------------------------------
# 4. val (selection) -- then STOP for the hand-label audit
# ---------------------------------------------------------------------------
CUR=val
if should_run val; then
    banner val
    run_driver val
    finish_step val
    echo
    echo "############################################################"
    echo "# VAL DONE -- STOPPING FOR THE HAND-LABEL AUDIT."
    echo "#"
    echo "# 1. Open the audit sheet(s) and fill the human_* columns for 25-50 rows:"
    ls -1 $AUDIT_GLOB 2>/dev/null | sed 's/^/#      /' || echo "#      (no audit CSV found under results/labels/ ?!)"
    echo "#    columns: human_option_letter (A-D), human_option_correct (1/0),"
    echo "#             human_expl_ok (1/0), human_expl_contradicts (1/0), notes"
    echo "#    (one sheet is enough -- e.g. the beta0.01 seed-42 one; each has 40 rows)"
    echo "# 2. Inspect results/abstention/medmcqa_gen/frozen_beta*-S35.json (val-selected thresholds)."
    echo "# 3. Re-run:  bash run-medmcqa-sequence.sh     -> audit-import, test (ONCE), rest"
    echo "############################################################"
    exit 0
fi

# ---------------------------------------------------------------------------
# 5. audit-import: fold the hand labels into the val label summaries
# ---------------------------------------------------------------------------
CUR=audit-import
if should_run audit-import; then
    banner audit-import
    # Which audit CSVs carry human labels? (any row with human_option_correct in {0,1})
    FILLED=()
    for csv in $AUDIT_GLOB; do
        [ -f "$csv" ] || continue
        if python - "$csv" <<'PY'
import csv, sys
with open(sys.argv[1], newline="", encoding="utf-8") as f:
    n = sum(1 for r in csv.DictReader(f)
            if (r.get("human_option_correct") or "").strip() in ("0", "1")
            or (r.get("human_expl_ok") or "").strip() in ("0", "1"))
print(f"  {sys.argv[1]}: {n} hand-labelled rows")
sys.exit(0 if n > 0 else 1)
PY
        then FILLED+=("$csv"); fi
    done
    if [ ${#FILLED[@]} -eq 0 ]; then
        echo "ERROR: no audit CSV under results/labels/ has hand labels yet."
        echo "       Fill human_* columns in one of: $AUDIT_GLOB"
        echo "       (or SKIP_AUDIT=1 to go to test without the audit -- not recommended)."
        [ "${SKIP_AUDIT:-0}" = "1" ] || exit 1
        echo ">>> SKIP_AUDIT=1: continuing without audit import."
    else
        NLI_FLAG=""; [ "${SKIP_NLI:-0}" = "1" ] && NLI_FLAG="--no_nli"
        for csv in "${FILLED[@]}"; do
            # audit_medmcqa_gen_val_generate_<tag>.csv -> step1_medmcqa_gen_val_generate_<tag>_seqlevel.jsonl
            tag="$(basename "$csv" .csv)"; tag="${tag#audit_}"
            seq="${TOK_DIR}/step1_${tag}_seqlevel.jsonl"
            [ -f "$seq" ] || { echo "ERROR: $seq not found for $csv"; exit 1; }
            echo "+ label_generation_correctness.py --input $seq --audit_import $csv --overwrite"
            python label_generation_correctness.py --input "$seq" --nli_model "$NLI_MODEL" $NLI_FLAG \
                --audit_import "$csv" --overwrite
        done
        echo ">>> audit vs human summaries written to ${TOK_DIR}/step1_*_labels_summary.json (key 'audit')"
    fi
    finish_step audit-import
fi

# ---------------------------------------------------------------------------
# 6. test: ONE evaluation per frozen configuration
# ---------------------------------------------------------------------------
CUR=test
if should_run test; then
    banner test
    if is_done test && [ "$FROM_IDX" -lt 0 ]; then echo "ERROR: test already ran once (protocol: evaluate test ONCE). Use FROM=test to override."; exit 1; fi
    run_driver test
    finish_step test
fi

# ---------------------------------------------------------------------------
# 7. the rest: baselines, prior, kl, ood, report
# ---------------------------------------------------------------------------
CUR=rest
if should_run rest; then
    banner "rest (baselines,prior,kl,ood,report)"
    run_driver baselines,prior,kl,ood,report
    finish_step rest
fi

echo
echo "############################################################"
echo "# medmcqa sequence COMPLETE.  Reports: results/reports/ladder_medmcqa_gen_{val,test}.md"
echo "# Log: $LOG"
echo "############################################################"

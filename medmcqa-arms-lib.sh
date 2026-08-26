# Helpers sourced by run-overnight-medmcqa-arms.sh (not meant to be run directly).
# Everything that is "plumbing" lives here so the driver reads as the experiment.

# ---- W&B alert + failure trap ------------------------------------------------
notify() {   # <info|error> <title> <text>
    local level="$1" title="$2" text="$3"
    python - "$level" "$title" "$text" <<'PY' || echo "(W&B alert failed -- check WANDB_API_KEY)"
import os, sys
level, title, text = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    import wandb
    run = wandb.init(project=os.environ.get("WANDB_PROJECT", "moe-uncertainty"),
                     name=f"overnight-watchdog-{os.environ.get('RUN_TAG', '')}", job_type="alert", reinit=True)
    wandb.alert(title=title, text=text, level=wandb.AlertLevel.ERROR if level == "error" else wandb.AlertLevel.INFO)
    run.finish()
    print(f"W&B alert sent: [{level}] {title}")
except Exception as e:
    print(f"W&B alert failed: {type(e).__name__}: {e}")
PY
}

on_error() {
    local exit_code=$? line=$1
    echo ""; echo "############################################################"
    echo "# FAILED at line ${line} (exit ${exit_code}) -- ${STEP:-unknown step}"
    echo "# Full log: ${LOG}   (fix, then: RESUME=1 bash $(basename "$0"))"
    echo "############################################################"
    notify error "$(basename "$0" .sh) FAILED: ${STEP:-unknown step}" \
        "Host $(hostname), run ${RUN_TAG}, exit ${exit_code} at line ${line}. Log: ${LOG}"
    exit "$exit_code"
}

# ---- phases / arms / artefact paths -------------------------------------------
has_phase() { [[ ",$PHASES," == *",$1,"* ]]; }

# target_mode -> artefact suffix. Mirrors utils.data.TARGET_MODE_SUFFIX (python).
arm_sfx() {
    case "$1" in
        letter) echo "armA-letter" ;;
        answer_explanation) echo "armB-ansexp" ;;
        *) echo "ERROR: unknown arm '$1' (letter|answer_explanation)" >&2; return 1 ;;
    esac
}
adapter_dir() { echo "./adapters/${MODEL_SHORTCODE}-${DATASET_SHORTCODE}-$(arm_sfx "$1")"; }
map_dir()     { echo "./router_weights/base/${MODEL_SHORTCODE}_${DATASET_SHORTCODE}-$(arm_sfx "$1")"; }
fcvr_sfx()    { echo "$(arm_sfx "$1")-${PRIOR_SOURCE}-prior-beta$2"; }           # <arm> <beta>
fcvr_dir()    { echo "./router_weights/fcvr/fcvr-${MODEL_SHORTCODE}-${DATASET_SHORTCODE}-$(fcvr_sfx "$1" "$2")"; }
eval_json()   { echo "${OUT_DIR}/$1_$2.json"; }                                     # <tag> <split>

# evaluate_letter.py tags (the report pairs arm A/B on an EXACT remainder match)
tag_kvq()  { echo "$(arm_sfx "$1")_kvq-s$2"; }                  # <arm> <seed>
tag_det()  { echo "$(arm_sfx "$1")_det-s$2"; }                  # <arm> <seed>
tag_fcvr() { echo "$(arm_sfx "$1")_fcvr-beta$2_S${S}-s$3"; }    # <arm> <beta> <seed>

# skip_done <path>: true (=> skip) if RESUME=1 and the artefact exists
skip_done() {
    if [ "${RESUME:-0}" = "1" ] && [ -e "$1" ]; then echo "  [RESUME] exists, skipping: $1"; return 0; fi
    return 1
}
n_arg() { [ "$1" -gt 0 ] 2>/dev/null && echo "--n $1" || true; }

# every output path this run would write (collision sweep)
all_outputs() {
    local ARM B SPLIT SEED
    has_phase eval && for SPLIT in val test; do echo "$(eval_json zero-shot "$SPLIT")"; done
    for ARM in $ARMS; do
        has_phase stage1 && echo "$(adapter_dir "$ARM")"
        has_phase map && echo "$(map_dir "$ARM")"
        for B in "${BETAS[@]}"; do has_phase fcvr && echo "$(fcvr_dir "$ARM" "$B")"; done
        if has_phase eval; then
            for SPLIT in val test; do
                echo "$(eval_json "$(tag_kvq "$ARM" "$SEED0")" "$SPLIT")"
                echo "$(eval_json "$(tag_det "$ARM" "$SEED0")" "$SPLIT")"
                for B in "${BETAS[@]}"; do for SEED in "${EVAL_SEEDS[@]}"; do
                    echo "$(eval_json "$(tag_fcvr "$ARM" "$B" "$SEED")" "$SPLIT")"
                done; done
            done
        fi
    done
}

collision_sweep() {
    local P COLLISIONS=()
    while IFS= read -r P; do [ -n "$P" ] && [ -e "$P" ] && COLLISIONS+=("$P"); done < <(all_outputs)
    if [ ${#COLLISIONS[@]} -eq 0 ]; then echo "OK: no collisions. Will write:"; all_outputs | sed 's/^/  /'; return 0; fi
    if [ "${RESUME:-0}" = "1" ]; then
        echo "RESUME=1: the following outputs exist and will be SKIPPED (not overwritten):"; printf '  %s\n' "${COLLISIONS[@]}"
    elif [ "${ALLOW_EXISTING:-0}" = "1" ]; then
        echo "ALLOW_EXISTING=1: the following outputs exist and WILL BE OVERWRITTEN:"; printf '  %s\n' "${COLLISIONS[@]}"
    else
        echo "ERROR: these output paths already exist -- refusing to overwrite:" >&2; printf '  %s\n' "${COLLISIONS[@]}" >&2
        echo "Use RESUME=1 to skip finished steps, ALLOW_EXISTING=1 to overwrite, or move them." >&2
        return 1
    fi
}

# ---- stage runners -------------------------------------------------------------
# run_stage <name> <done-marker> <log> <train-info-file> <grep-pattern> -- <command...>
# Runs the command (tee'd to <log>) unless the marker exists under RESUME=1; then
# appends the matching log lines (optimizer steps, val curve, early stopping) to
# the train-info file so the report can cite them.
run_stage() {
    local NAME="$1" MARKER="$2" SLOG="$3" INFO="$4" PAT="$5"; shift 5; [ "$1" = "--" ] && shift
    STEP="$NAME"
    echo ""; echo "==== $NAME -- $(date) ===="
    if skip_done "$MARKER"; then return 0; fi
    "$@" 2>&1 | tee "$SLOG"
    [ -e "$MARKER" ] || { echo "ERROR: $NAME finished but $MARKER is missing." >&2; return 1; }
    { echo "== $NAME ($(date)) =="; grep -E "$PAT" "$SLOG"; echo; } >> "$INFO" || true
    echo "$NAME done -- $(date)"
}

# run_eval <tag> <split> <seed> <method> [extra evaluate_letter.py args...]
run_eval() {
    local TAG="$1" SPLIT="$2" SEED="$3" METHOD="$4"; shift 4
    local N; if [ "$SPLIT" = "val" ]; then N="$N_VAL"; else N="$N_TEST"; fi
    STEP="eval $TAG [$SPLIT]"
    echo ""; echo "---- eval $TAG [$SPLIT] method=$METHOD seed=$SEED -- $(date) ----"
    if skip_done "$(eval_json "$TAG" "$SPLIT")"; then return 0; fi
    local OW=(); [ "${ALLOW_EXISTING:-0}" = "1" ] && OW=(--overwrite)
    # shellcheck disable=SC2046
    python evaluate_letter.py --model_shortcode "$MODEL_SHORTCODE" --dataset_shortcode "$DATASET_SHORTCODE" \
        --split "$SPLIT" --method "$METHOD" --batch_size "$EVAL_BATCH" --num_samples "$S" --seed "$SEED" \
        --tag "$TAG" --out_dir "$OUT_DIR" $(n_arg "$N") "${OW[@]}" "$@"
}

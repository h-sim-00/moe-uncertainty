#!/bin/bash
# Submit the OBQA-gemma pipeline as an afterok chain on Isambard-AI (24 h QoS):
#
#   prep (1 GPU, 4 h; downloads + preflight)
#   -> train[depth]        (4 GPUs, 23.5 h; stage1 + map + fcvr on 5 6 7 8 18 19 26 27 28 29)
#   -> train[depth]-resume (afterany insurance; RESUME=1 no-op when done)
#   -> eval[depth]         (1 GPU; letter evals, report, OoD arms)
#   -> oodexpl[depth]      (1 GPU; nine explanation-ILV aggregates, tf + gen)
#
#   bash submit-obqa-gemma-chain.sh                    # full chain
#   FROM=train bash submit-obqa-gemma-chain.sh         # skip prep (already done)
#   FROM=eval  bash submit-obqa-gemma-chain.sh         # eval onwards
#   FROM=oodexpl bash submit-obqa-gemma-chain.sh       # read-out only
#   INITIAL_DEP=afterok:<jobid> FROM=train bash ...    # attach to an already-submitted prep job
#   DRY=1 bash submit-obqa-gemma-chain.sh              # print the sbatch lines only
#
# Gemma 4 has a single layer set ('depth'; Granite's literal indices exceed its
# 30 layers), so unlike submit-obqa-qwen-chain.sh there is no per-set fork.
# LAYER_SETS="depth other" still works if you define LAYERS_other="..." for the jobs.
#
# Run from the Gemma WORKTREE (see sbatch-obqa-gemma.sh header):
#   cd /projects/u6qd/moe-uncertainty-gemma && bash submit-obqa-gemma-chain.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
FROM="${FROM:-prep}"
DRY="${DRY:-0}"
TRAIN_GPUS="${TRAIN_GPUS:-4}"
LAYER_SETS="${LAYER_SETS:-depth}"             # order = execution order
REPO="${REPO:-$(pwd)}"
export LAYER_SETS REPO
mkdir -p "$REPO/logs" 2>/dev/null || [ "$DRY" = "1" ] || { echo "cannot create $REPO/logs (#SBATCH --output dir must exist)" >&2; exit 1; }

submit() {   # <stage> <gpus> <time> <dependency or ""> [job-name suffix] [extra --export vars] -> job id
    local STAGE="$1" GPUS="$2" TIME="$3" DEP="$4" SUFFIX="${5:-}" EXTRA="${6:-}"
    local NAME="obqa-gemma-${STAGE}${SUFFIX:+-$SUFFIX}"
    local ARGS=(--parsable --job-name="$NAME" --nodes=1 --time="$TIME" --export="ALL,STAGE=${STAGE},REPO=${REPO}${EXTRA:+,$EXTRA}")
    if [ "$GPUS" -ge 4 ]; then ARGS+=(--gpus-per-node="$GPUS"); else ARGS+=(--gpus="$GPUS"); fi
    [ -n "$DEP" ] && ARGS+=(--dependency="$DEP")
    if [ "$DRY" = "1" ]; then echo "sbatch ${ARGS[*]} sbatch-obqa-gemma.sh" >&2; echo "DRY-$NAME"; return; fi
    sbatch "${ARGS[@]}" sbatch-obqa-gemma.sh
}

# Ordered step list: prep, then train/eval/oodexpl per layer set.
steps=(prep)
for LS in $LAYER_SETS; do steps+=("train:$LS" "eval:$LS" "oodexpl:$LS"); done
first_set="${LAYER_SETS%% *}"
# FROM = prep | <stage> (first layer set) | <stage>-<set>
WANT="${FROM/-/:}"; case "$WANT" in prep|*:*) ;; *) WANT="$WANT:$first_set" ;; esac
start=-1
for i in "${!steps[@]}"; do [ "${steps[$i]}" = "$WANT" ] && { start=$i; break; }; done
[ "$start" -ge 0 ] || { echo "FROM=$FROM not found; valid: ${steps[*]//:/-}" >&2; exit 2; }

DEP="${INITIAL_DEP:-}"  # optional existing dependency, then the next step in the strict chain
for ((i=start; i<${#steps[@]}; i++)); do
    s="${steps[$i]}"; STAGE="${s%%:*}"; LS="${s#*:}"
    case "$STAGE" in
        prep)
            J=$(submit prep 1 04:00:00 "$DEP")
            echo "submitted prep job $J (dependency: ${DEP:-none})"; DEP="afterok:$J" ;;
        train)
            J=$(submit train "$TRAIN_GPUS" 23:30:00 "$DEP" "$LS" "LAYER_SETS=$LS")
            echo "submitted train[$LS]        job $J (dependency: ${DEP:-none})"
            # insurance re-run: continues an unfinished stage if the first hit the 24 h wall
            J2=$(submit train "$TRAIN_GPUS" 23:30:00 "afterany:$J" "$LS-resume" "LAYER_SETS=$LS")
            echo "submitted train[$LS]-resume job $J2 (afterany $J; RESUME=1 -> no-op when done)"
            DEP="afterok:$J2" ;;
        eval)
            J=$(submit eval 1 23:30:00 "$DEP" "$LS" "LAYER_SETS=$LS")
            echo "submitted eval[$LS] job $J (dependency: ${DEP:-none})"; DEP="afterok:$J" ;;
        oodexpl)
            J=$(submit oodexpl 1 23:30:00 "$DEP" "$LS" "LAYER_SET=$LS")
            echo "submitted oodexpl[$LS] job $J (dependency: ${DEP:-none}; not a dependency of later steps)" ;;
    esac
done
echo "queue: squeue -u \$USER   | logs: ${REPO}/logs/obqa-gemma-<stage>-<set>-<jobid>.out"

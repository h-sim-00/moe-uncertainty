#!/bin/bash
# Submit the OBQA-qwen pipeline as an afterok chain on Isambard-AI (24 h QoS),
# ONE LAYER SET AT A TIME so the first set's results can be examined while the
# next set trains:
#
#   prep (1 GPU, 4 h; preflight covers every layer set)
#   -> train[literal]   (4 GPUs, 23.5 h; stage1 + map + fcvr on 5 6 7 8 19 20 28 29 30 31)
#   -> train[literal]-resume (afterany insurance; RESUME=1 no-op when done)
#   -> eval[literal]    (1 GPU; letter evals, report, OoD arms)
#   -> oodexpl[literal] (1 GPU)  ||  train[depth] (4 GPUs; stage1/map skipped by
#                                    RESUME=1, fcvr on 6 7 8 9 24 25 36 37 38 39)
#                                 -> train[depth]-resume -> eval[depth] (report
#                                    now lists both sets) -> oodexpl[depth]
#
#   bash submit-obqa-qwen-chain.sh                    # full chain
#   FROM=train bash submit-obqa-qwen-chain.sh         # skip prep (already done)
#   FROM=eval  bash submit-obqa-qwen-chain.sh         # eval[literal] onwards
#   FROM=train-depth bash submit-obqa-qwen-chain.sh   # literal all done: depth train/eval/oodexpl only
#   INITIAL_DEP=afterok:<jobid> FROM=train bash ...   # attach to an already-submitted prep job
#   LAYER_SETS="literal" bash ...                     # a single layer set end-to-end
#   DRY=1 bash submit-obqa-qwen-chain.sh              # print the sbatch lines only
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
FROM="${FROM:-prep}"
DRY="${DRY:-0}"
TRAIN_GPUS="${TRAIN_GPUS:-4}"
LAYER_SETS="${LAYER_SETS:-literal depth}"     # order = execution order
export LAYER_SETS                              # prep's collision sweep / memchecks cover every set

submit() {   # <stage> <gpus> <time> <dependency or ""> [job-name suffix] [extra --export vars] -> job id
    local STAGE="$1" GPUS="$2" TIME="$3" DEP="$4" SUFFIX="${5:-}" EXTRA="${6:-}"
    local NAME="obqa-qwen-${STAGE}${SUFFIX:+-$SUFFIX}"
    local ARGS=(--parsable --job-name="$NAME" --nodes=1 --time="$TIME" --export="ALL,STAGE=${STAGE}${EXTRA:+,$EXTRA}")
    if [ "$GPUS" -ge 4 ]; then ARGS+=(--gpus-per-node="$GPUS"); else ARGS+=(--gpus="$GPUS"); fi
    [ -n "$DEP" ] && ARGS+=(--dependency="$DEP")
    if [ "$DRY" = "1" ]; then echo "sbatch ${ARGS[*]} sbatch-obqa-qwen.sh" >&2; echo "DRY-$NAME"; return; fi
    sbatch "${ARGS[@]}" sbatch-obqa-qwen.sh
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
            # Reads only this set's finished weights: runs in parallel with the NEXT
            # set's training (both depend on eval[$LS]); nothing waits on it.
            J=$(submit oodexpl 1 23:30:00 "$DEP" "$LS" "LAYER_SET=$LS")
            echo "submitted oodexpl[$LS] job $J (dependency: ${DEP:-none}; not a dependency of later steps)" ;;
    esac
done
echo "queue: squeue -u \$USER   | logs: /projects/u6qd/moe-uncertainty/logs/obqa-qwen-<stage>-<set>-<jobid>.out"

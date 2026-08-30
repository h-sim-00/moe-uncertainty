#!/bin/bash
# Submit the OBQA-qwen pipeline as an afterok chain on Isambard-AI (24 h QoS):
#   prep (1 GPU, 4 h) -> train (4 GPUs, 23.5 h) -> train-insurance (afterany;
#   RESUME=1 no-op if everything finished, otherwise picks up the unfinished
#   stage) -> eval (1 GPU) -> oodexpl (1 GPU).
#
#   bash submit-obqa-qwen-chain.sh              # full chain
#   FROM=train bash submit-obqa-qwen-chain.sh   # skip prep (already done)
#   FROM=eval  bash submit-obqa-qwen-chain.sh   # only eval + oodexpl
#   DRY=1 bash submit-obqa-qwen-chain.sh        # print the sbatch lines only
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
FROM="${FROM:-prep}"
DRY="${DRY:-0}"
TRAIN_GPUS="${TRAIN_GPUS:-4}"

submit() {   # <stage> <gpus> <time> <dependency or ""> -> job id
    local STAGE="$1" GPUS="$2" TIME="$3" DEP="$4"
    local ARGS=(--parsable --job-name="obqa-qwen-${STAGE}" --nodes=1 --time="$TIME" --export=ALL,STAGE="$STAGE")
    if [ "$GPUS" -ge 4 ]; then ARGS+=(--gpus-per-node="$GPUS"); else ARGS+=(--gpus="$GPUS"); fi
    [ -n "$DEP" ] && ARGS+=(--dependency="$DEP")
    if [ "$DRY" = "1" ]; then echo "sbatch ${ARGS[*]} sbatch-obqa-qwen.sh" >&2; echo "DRY-$STAGE"; return; fi
    sbatch "${ARGS[@]}" sbatch-obqa-qwen.sh
}

order=(prep train eval oodexpl)
start=0; for i in "${!order[@]}"; do [ "${order[$i]}" = "$FROM" ] && start=$i; done

DEP=""; PREV=""
for ((i=start; i<${#order[@]}; i++)); do
    S="${order[$i]}"
    case "$S" in
        prep)    J=$(submit prep 1 04:00:00 "$DEP") ;;
        train)   J=$(submit train "$TRAIN_GPUS" 23:30:00 "$DEP")
                 echo "submitted train        job $J (afterok ${PREV:-none})"
                 # insurance re-run: continues an unfinished stage if the first hit the 24 h wall
                 J2=$(submit train "$TRAIN_GPUS" 23:30:00 "afterany:$J")
                 echo "submitted train-resume job $J2 (afterany $J; RESUME=1 -> no-op when done)"
                 PREV="$J2"; DEP="afterok:$J2"; continue ;;
        eval)    J=$(submit eval 1 23:30:00 "$DEP") ;;
        oodexpl) J=$(submit oodexpl 1 23:30:00 "$DEP") ;;
    esac
    echo "submitted $S job $J (dependency: ${DEP:-none})"
    PREV="$J"; DEP="afterok:$J"
done
echo "queue: squeue -u \$USER   | logs: /projects/u6qd/moe-uncertainty/logs/obqa-qwen-<stage>-<jobid>.out"

#!/bin/bash
# ============================================================================
# Isambard-AI (phase 2) batch wrapper for the OBQA-qwen (Qwen3.6-35B-A3B) run.
# One STAGE per job (the workq QoS caps jobs at 24 h; every driver phase is
# resumable with RESUME=1, so a chain of jobs = one logical run):
#
#   STAGE=prep     1 GPU   env sanity, HF model + dataset downloads, driver preflight
#   STAGE=train    4 GPUs  RESUME=1 PHASES=stage1,map,fcvr   (torchrun DDP, both arms;
#                          FCVR for every layer set in LAYER_SETS, default "literal depth")
#   STAGE=eval     1 GPU   RESUME=1 PHASES=eval,report,ood   (all layer sets, one table)
#   STAGE=oodexpl  1 GPU   run-ood-expl-readout.sh stages stage1,tf,gen (nine
#                          explanation ILV aggregates, teacher-forced + generated)
#                          for ONE layer set: LAYER_SET=literal (default) or depth
#                          -> results/ood_expl_readout/obqa-ood-expl-qwen36[-layers-depth]_*
#
# Submit by hand:   STAGE=prep sbatch --gpus=1 --time=04:00:00 sbatch-obqa-qwen.sh
#                   STAGE=train sbatch --gpus-per-node=4 --time=23:30:00 sbatch-obqa-qwen.sh
#                   STAGE=oodexpl LAYER_SET=depth sbatch --gpus=1 --time=23:30:00 sbatch-obqa-qwen.sh
# or use submit-obqa-qwen-chain.sh (afterok chain with the right GPU counts,
# ONE LAYER SET AT A TIME: train[literal] -> eval[literal] -> {oodexpl[literal]
# || train[depth] -> eval[depth] -> oodexpl[depth]}; each train/eval job gets
# LAYER_SETS=<set>, each oodexpl job LAYER_SET=<set>).
#
# Isambard notes (docs.isambard.ac.uk): partition workq (default), QoS
# workq_qos MaxWall 1-00:00:00 (default 4 h -> always pass --time), --gpus=N
# allocates N GH200 superchips (96 GB HBM + 120 GB LPDDR each, aarch64),
# logs/checkpoints on /projects (not $HOME, 100 GiB), nothing is backed up.
# ============================================================================
#SBATCH --job-name=obqa-qwen
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=23:30:00
#SBATCH --output=/projects/u6qd/moe-uncertainty/logs/%x-%j.out
#SBATCH --error=/projects/u6qd/moe-uncertainty/logs/%x-%j.err

set -e

STAGE="${STAGE:-train}"
REPO="${REPO:-/projects/u6qd/moe-uncertainty}"
CONDA_ENV="${CONDA_ENV:-qwen_env}"

# Activate our Python environment
source ~/miniforge3/bin/activate
conda activate "$CONDA_ENV"

# NCCL for the multi-GPU (torchrun) stages
module load brics/nccl 2>/dev/null || echo "(module brics/nccl not loaded -- using the wheel's bundled NCCL)"

# Keep Hugging Face downloads/cache out of $HOME
export HF_HOME="${HF_HOME:-/projects/u6qd/hf_cache}"
export MOE_RAW_DATA_DIR="${MOE_RAW_DATA_DIR:-/projects/u6qd/moe_raw_data}"     # ECQA / CommonsenseQA raw files
export WANDB_PROJECT="${WANDB_PROJECT:-moe-uncertainty-qwen36}"
export WANDB_DIR="${WANDB_DIR:-/projects/u6qd/wandb}"
export GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
# After prep everything is cached: never let a compute job stall on the network.
if [ "$STAGE" != "prep" ]; then export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"; export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"; fi
mkdir -p "$HF_HOME" "$MOE_RAW_DATA_DIR" "$WANDB_DIR" "$REPO/logs"

# Move explicitly into the repository
cd "$REPO"

echo "========================================"
echo "Job ID: $SLURM_JOB_ID   Stage: $STAGE   LAYER_SETS=${LAYER_SETS:-literal depth}   LAYER_SET(oodexpl)=${LAYER_SET:-literal}"
echo "Started: $(date)"
echo "Node: $(hostname)"
echo "Git branch/commit: $(git rev-parse --abbrev-ref HEAD) $(git rev-parse HEAD)"
echo "Python: $(which python)   conda env: $CONDA_ENV"
echo "GPUs on node (SLURM_GPUS_ON_NODE): ${SLURM_GPUS_ON_NODE:-?}   HF_HOME=$HF_HOME   WANDB_PROJECT=$WANDB_PROJECT"
echo "========================================"

nvidia-smi

python - <<'PY'
import torch, transformers, peft
print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
print("GPU:", torch.cuda.get_device_name(0))
print("transformers:", transformers.__version__, "| peft:", peft.__version__)
PY

# ----------------------------------------
# STAGE DISPATCH
# ----------------------------------------
case "$STAGE" in
    prep)
        bash setup-isambard-qwen-env.sh --downloads-only
        PHASES=preflight NGPUS=1 bash run-overnight-obqa-qwen-arms.sh
        ;;
    train)
        RESUME=1 PHASES="${PHASES:-stage1,map,fcvr}" bash run-overnight-obqa-qwen-arms.sh
        ;;
    eval)
        RESUME=1 NGPUS=1 PHASES="${PHASES:-eval,report,ood}" bash run-overnight-obqa-qwen-arms.sh
        ;;
    oodexpl)
        RESUME=1 MODEL_SHORTCODE=qwen36 LAYER_SET="${LAYER_SET:-literal}" STAGES="${STAGES:-stage1,tf,gen}" bash run-ood-expl-readout.sh
        ;;
    smoke)
        # tiny end-to-end on a few rows (see smoke-qwen36-model.py + plan Part 6)
        python smoke-qwen36-model.py
        MOE_SMOKE_N_ROWS="${MOE_SMOKE_N_ROWS:-64}" RESUME=0 ALLOW_EXISTING=1 SKIP_PREFLIGHT=1 \
            PHASES="${PHASES:-preflight,stage1,fcvr,eval}" ARMS="${ARMS:-answer_explanation}" \
            STAGE1_EPOCHS=1 FCVR_EPOCHS=1 N_VAL=8 N_TEST=8 EVAL_SEEDS="42" \
            bash run-overnight-obqa-qwen-arms.sh
        ;;
    *)
        echo "unknown STAGE=$STAGE (prep|train|eval|oodexpl|smoke)" >&2; exit 2 ;;
esac

echo "========================================"
echo "Finished: $(date)"
echo "========================================"

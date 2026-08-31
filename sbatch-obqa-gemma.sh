#!/bin/bash
# ============================================================================
# Isambard-AI (phase 2) batch wrapper for the OBQA-gemma (Gemma 4 26B-A4B-it) run.
# One STAGE per job (the workq QoS caps jobs at 24 h; every driver phase is
# resumable with RESUME=1, so a chain of jobs = one logical run):
#
#   STAGE=prep     1 GPU   env sanity (qwen_env already has gemma4), HF model +
#                          dataset downloads, driver preflight
#   STAGE=train    4 GPUs  RESUME=1 PHASES=stage1,map,fcvr   (torchrun DDP, both arms;
#                          FCVR on the single 'depth' layer set 5 6 7 8 18 19 26 27 28 29)
#   STAGE=eval     1 GPU   RESUME=1 PHASES=eval,report,ood
#   STAGE=oodexpl  1 GPU   run-ood-expl-readout.sh stages stage1,tf,gen (nine
#                          explanation ILV aggregates, teacher-forced + generated)
#                          -> results/ood_expl_readout/obqa-ood-expl-gemma4-layers-depth_*
#   STAGE=smoke    4 GPUs  smoke-gemma4-model.py + a 64-row end-to-end mini run
#
# CHECKOUT: the Qwen chain runs from /projects/u6qd/moe-uncertainty (branch
# OBQA-qwen) and every one of its stages re-imports that checkout's python, so
# this branch lives in its OWN worktree (also keeps adapters/ router_weights/
# results/ apart):
#   cd /projects/u6qd/moe-uncertainty && git fetch origin OBQA-gemma \
#     && git worktree add /projects/u6qd/moe-uncertainty-gemma OBQA-gemma
#   mkdir -p /projects/u6qd/moe-uncertainty-gemma/logs            # SBATCH --output needs the dir
#   (optional) cp splits/obqa_gen-derived-seed42.csv /projects/u6qd/moe-uncertainty-gemma/splits/
#   -- the manifest is re-derived deterministically by preflight otherwise; the
#   eligible-ID lists are tokenizer-specific (*-gemma4.txt) and written fresh.
#
# Submit by hand:   STAGE=prep sbatch --gpus=1 --time=04:00:00 sbatch-obqa-gemma.sh
#                   STAGE=train sbatch --gpus-per-node=4 --time=23:30:00 sbatch-obqa-gemma.sh
#                   STAGE=oodexpl sbatch --gpus=1 --time=23:30:00 sbatch-obqa-gemma.sh
# or use submit-obqa-gemma-chain.sh (afterok chain with the right GPU counts).
#
# Isambard notes (docs.isambard.ac.uk): partition workq (default), QoS
# workq_qos MaxWall 1-00:00:00 (default 4 h -> always pass --time), --gpus=N
# allocates N GH200 superchips (96 GB HBM + 120 GB LPDDR each, aarch64),
# logs/checkpoints on /projects (not $HOME, 100 GiB), nothing is backed up.
# ============================================================================
#SBATCH --job-name=obqa-gemma
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=23:30:00
#SBATCH --output=/projects/u6qd/moe-uncertainty-gemma/logs/%x-%j.out
#SBATCH --error=/projects/u6qd/moe-uncertainty-gemma/logs/%x-%j.err

set -e

STAGE="${STAGE:-train}"
REPO="${REPO:-/projects/u6qd/moe-uncertainty-gemma}"
CONDA_ENV="${CONDA_ENV:-qwen_env}"     # transformers 5.16.1 + peft 0.20.0: has Gemma4ForCausalLM

# Activate our Python environment
source ~/miniforge3/bin/activate
conda activate "$CONDA_ENV"

# NCCL for the multi-GPU (torchrun) stages
module load brics/nccl 2>/dev/null || echo "(module brics/nccl not loaded -- using the wheel's bundled NCCL)"

# Keep Hugging Face downloads/cache out of $HOME (shared with the Qwen run)
export HF_HOME="${HF_HOME:-/projects/u6qd/hf_cache}"
export MOE_RAW_DATA_DIR="${MOE_RAW_DATA_DIR:-/projects/u6qd/moe_raw_data}"     # ECQA / CommonsenseQA raw files
export WANDB_PROJECT="${WANDB_PROJECT:-moe-uncertainty-gemma4}"
export WANDB_DIR="${WANDB_DIR:-/projects/u6qd/wandb}"
export GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
# After prep everything is cached: never let a compute job stall on the network.
if [ "$STAGE" != "prep" ]; then export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"; export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"; fi
mkdir -p "$HF_HOME" "$MOE_RAW_DATA_DIR" "$WANDB_DIR" "$REPO/logs"

# Move explicitly into the repository (worktree)
cd "$REPO"

echo "========================================"
echo "Job ID: $SLURM_JOB_ID   Stage: $STAGE   LAYER_SETS=${LAYER_SETS:-depth}   LAYER_SET(oodexpl)=${LAYER_SET:-depth}"
echo "Started: $(date)"
echo "Node: $(hostname)"
echo "Repo: $REPO   Git branch/commit: $(git rev-parse --abbrev-ref HEAD) $(git rev-parse HEAD)"
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
from transformers import Gemma4ForCausalLM  # noqa: F401
print("Gemma4ForCausalLM importable: yes")
PY

# ----------------------------------------
# STAGE DISPATCH
# ----------------------------------------
case "$STAGE" in
    prep)
        bash setup-isambard-gemma-env.sh --downloads-only
        PHASES=preflight NGPUS=1 LAYER_SETS="${LAYER_SETS:-depth}" bash run-overnight-obqa-gemma-arms.sh
        ;;
    train)
        RESUME=1 LAYER_SETS="${LAYER_SETS:-depth}" PHASES="${PHASES:-stage1,map,fcvr}" bash run-overnight-obqa-gemma-arms.sh
        ;;
    eval)
        RESUME=1 NGPUS=1 LAYER_SETS="${LAYER_SETS:-depth}" PHASES="${PHASES:-eval,report,ood}" bash run-overnight-obqa-gemma-arms.sh
        ;;
    oodexpl)
        RESUME=1 MODEL_SHORTCODE=gemma4 LAYER_SET="${LAYER_SET:-depth}" STAGES="${STAGES:-stage1,tf,gen}" bash run-ood-expl-readout.sh
        ;;
    smoke)
        # model-layer checks, then a tiny end-to-end on a few rows (suffix-free
        # dirs are namespaced by MODEL_SHORTCODE=gemma4, so nothing else is touched)
        echo "==== smoke 1/3: model checks, HF default experts impl ===="
        time python smoke-gemma4-model.py
        # experts-kernel trial: same checks (incl. ELBO grads + generation) on the
        # grouped-GEMM path -- a kernel choice read by load_model at load time, not
        # a math change. Compare the two 'real' times; if it passes, the mini run
        # uses it, and a later train picks it up via exported GEMMA_EXPERTS_IMPL.
        echo "==== smoke 2/3: model checks, GEMMA_EXPERTS_IMPL=grouped_mm ===="
        SMOKE_IMPL=""
        if time GEMMA_EXPERTS_IMPL=grouped_mm python smoke-gemma4-model.py; then
            echo "grouped_mm experts: ALL CHECKS PASSED"
            SMOKE_IMPL=grouped_mm
        else
            echo "WARNING: grouped_mm experts smoke FAILED -- mini run falls back to the HF default impl" >&2
        fi
        echo "==== smoke 3/3: 64-row end-to-end (experts impl: ${GEMMA_EXPERTS_IMPL:-${SMOKE_IMPL:-HF default}}) ===="
        MOE_SMOKE_N_ROWS="${MOE_SMOKE_N_ROWS:-64}" RESUME=0 ALLOW_EXISTING=1 SKIP_PREFLIGHT=1 LAYER_SETS=depth \
            GEMMA_EXPERTS_IMPL="${GEMMA_EXPERTS_IMPL:-$SMOKE_IMPL}" \
            PHASES="${PHASES:-preflight,stage1,fcvr,eval}" ARMS="${ARMS:-answer_explanation}" \
            STAGE1_EPOCHS=1 FCVR_EPOCHS=1 N_VAL=8 N_TEST=8 EVAL_SEEDS="42" \
            bash run-overnight-obqa-gemma-arms.sh
        ;;
    *)
        echo "unknown STAGE=$STAGE (prep|train|eval|oodexpl|smoke)" >&2; exit 2 ;;
esac

echo "========================================"
echo "Finished: $(date)"
echo "========================================"

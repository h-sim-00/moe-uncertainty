#!/bin/bash
#SBATCH --job-name=medmcqa-arms
#SBATCH --output=/vol/bitbucket/%u/logs/slurm/slurm_%j_medmcqa-arms.log
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=2-23:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sh2419@ic.ac.uk
# ============================================================================
# SLURM wrapper for run-overnight-medmcqa-arms.sh on the DoC GPU cluster.
# (Header mirrors the original author's scripts/bash/*.sh, updated for sh2419.)
#
# Submit from gpucluster2 (after setup-doc-gpucluster.sh has been run once):
#   export WANDB_API_KEY=...            # sbatch forwards your env by default
#   sbatch sbatch-medmcqa-arms.sh
#
# Pick a different GPU at submit time (check `sinfo` for live partition names;
# T4/A16 16GB will OOM -- use a40 48GB or a100 80GB):
#   sbatch --partition=a100 sbatch-medmcqa-arms.sh
#
# Walltime is capped at 3 days cluster-wide. If the job dies or is killed at
# the cap, fix/resubmit with the driver's resume mode:
#   RESUME=1 sbatch --export=ALL sbatch-medmcqa-arms.sh
#
# Driver knobs (PHASES, ARMS, BETAS, STAGE1_BATCH, ...) pass straight through:
#   PHASES=eval,report sbatch sbatch-medmcqa-arms.sh
# ============================================================================
set -Eeo pipefail

BB="/vol/bitbucket/${USER}"
export HF_HOME="$BB/.cache/huggingface"
export WANDB_PROJECT="${WANDB_PROJECT:-moe-uncertainty}"
# existing env, verified 2026-08-24 to match quail's moe_env exactly
# (py3.12.3, torch 2.12.1, transformers 4.47.1, peft 0.19.1, datasets 5.0.0)
source "$BB/moe/bin/activate"

cd "$BB/moe-uncertainty"
echo "host=$(hostname)  gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)  branch=$(git rev-parse --abbrev-ref HEAD)"

# SLURM gives the job exactly one visible GPU; it is always cuda:0 here
# (unlike quail-1 where we pointed things at cuda:1).
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

bash run-overnight-medmcqa-arms.sh

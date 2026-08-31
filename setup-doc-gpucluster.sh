#!/bin/bash
# ============================================================================
# ONE-TIME SETUP for the DoC SLURM GPU cluster (run ON gpucluster2, not locally).
#
# Recreates the quail-1 ~/.venvs/moe_env venv (Python 3.12.3, torch 2.12.1 cu13)
# under /vol/bitbucket/sh2419 (home dirs are tiny -- everything lives here),
# and clones the repo. Mirrors the original author's layout (al1624 used
# /vol/bitbucket/al1624/.venv/moe_env + HF_HOME on bitbucket).
#
# From your Mac:
#   ssh gpucluster2.doc.ic.ac.uk          # agent forwarding is on, so the
#   bash <this script>                    #   GitHub clone uses your Mac key
#
# Easiest way to get this script + requirements file over there first:
#   scp setup-doc-gpucluster.sh requirements-quail-freeze.txt sbatch-medmcqa-arms.sh \
#       gpucluster2.doc.ic.ac.uk:~
# ============================================================================
set -Eeuo pipefail

BB="/vol/bitbucket/${USER}"
# Existing cluster env (matches quail's moe_env: py3.12.3, torch 2.12.1,
# transformers 4.47.1, peft 0.19.1 -- verified 2026-08-24). Steps 2-3 are
# skipped when it exists; a fresh quail-clone env is only built if it's gone.
VENV="${VENV:-$BB/moe}"
FALLBACK_VENV="$BB/.venvs/moe_env"
if [ ! -e "$VENV/bin/activate" ]; then VENV="$FALLBACK_VENV"; fi
PROJ="$BB/moe-uncertainty"
REPO_SSH="git@github.com:h-sim-00/moe-uncertainty.git"
BRANCH="${BRANCH:-MedMCQA-comparison}"

echo "== 1/5 bitbucket layout =="
mkdir -p "$BB"/.venvs "$BB"/.cache/huggingface "$BB"/logs/slurm

if [ -e "$VENV/bin/activate" ]; then
    echo "== 2-3/5 SKIPPED: reusing existing env $VENV =="
    source "$VENV/bin/activate"
else
    echo "== 2/5 python =="
    # quail-1 env is Python 3.12.3; insist on 3.12 so wheel pins resolve identically
    PY="$(command -v python3.12 || true)"
    if [ -z "$PY" ]; then
        PYV="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
        if [ "$PYV" = "3.12" ]; then PY="$(command -v python3)"; else
            echo "ERROR: no python3.12 found (python3 is $PYV)." >&2
            echo "Check 'ls /usr/bin/python3*' or ask CSG; do NOT install with a different minor version." >&2
            exit 1
        fi
    fi
    echo "using $PY ($($PY --version))"

    echo "== 3/5 venv + packages (takes a while; torch cu13 wheels are ~3GB) =="
    "$PY" -m venv "$VENV"
    source "$VENV/bin/activate"
    pip install --upgrade pip
    REQ="$(dirname "${BASH_SOURCE[0]}")/requirements-quail-freeze.txt"
    [ -f "$REQ" ] || REQ="$HOME/requirements-quail-freeze.txt"
    pip install -r "$REQ"
fi

echo "== 4/5 repo clone =="
if [ ! -d "$PROJ/.git" ]; then
    git clone "$REPO_SSH" "$PROJ"   # needs your forwarded ssh agent (github key)
fi
git -C "$PROJ" fetch origin
git -C "$PROJ" checkout "$BRANCH"
git -C "$PROJ" pull --ff-only origin "$BRANCH"

echo "== 5/5 sanity check on a GPU node (30s) =="
# gpucluster2/3 banner: install envs on a lab PC, submit jobs here. /vol/bitbucket
# is shared, so steps 1-4 can run on any lab machine; this step needs SLURM.
if command -v srun >/dev/null; then
    srun --gres=gpu:1 --time=0:05:00 bash -c \
        "source $VENV/bin/activate && python -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))'" \
        || echo "WARNING: GPU check failed. If torch reports CUDA unavailable, the node driver may be too old for cu13 wheels -- tell Claude and we pin the cu126 torch build instead."
else
    echo "no srun on this host -- run this from gpucluster2 to finish the check:"
    echo "  srun --gres=gpu:1 --time=0:05:00 bash -c 'source $VENV/bin/activate && python -c \"import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))\"'"
fi

echo
echo "DONE. Next: copy sbatch-medmcqa-arms.sh into $PROJ (or it's already in the repo),"
echo "export WANDB_API_KEY, and submit with:  sbatch $PROJ/sbatch-medmcqa-arms.sh"

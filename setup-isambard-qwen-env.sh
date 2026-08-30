#!/bin/bash
# ============================================================================
# One-time Isambard-AI setup for the Qwen3.6-35B-A3B run (branch OBQA-qwen).
#
#   bash setup-isambard-qwen-env.sh                  # env + downloads
#   bash setup-isambard-qwen-env.sh --env-only       # just the conda env
#   bash setup-isambard-qwen-env.sh --downloads-only # just model/dataset caches (used by STAGE=prep)
#
# Run inside a job, per the Isambard docs ("create environments on compute
# nodes"; login nodes must not run intensive work):
#   srun --nodes=1 --gpus=1 --time=02:00:00 --pty bash -c 'bash setup-isambard-qwen-env.sh'
#
# Env: qwen_env = a CLONE of moe_env (torch 2.12.1+cu126, SM90-compatible, and
# every other pin unchanged) with transformers 5.16.1 + peft 0.20.0 on top.
# moe_env stays untouched so the Granite pipeline remains byte-for-byte runnable.
# ============================================================================
set -euo pipefail
MODE="${1:-all}"
REPO="${REPO:-/projects/u6qd/moe-uncertainty}"
export HF_HOME="${HF_HOME:-/projects/u6qd/hf_cache}"
export MOE_RAW_DATA_DIR="${MOE_RAW_DATA_DIR:-/projects/u6qd/moe_raw_data}"
SRC_ENV="${SRC_ENV:-moe_env}"; NEW_ENV="${NEW_ENV:-qwen_env}"
TRANSFORMERS_PIN="${TRANSFORMERS_PIN:-5.16.1}"; PEFT_PIN="${PEFT_PIN:-0.20.0}"
mkdir -p "$HF_HOME" "$MOE_RAW_DATA_DIR"

source ~/miniforge3/bin/activate

if [ "$MODE" = "all" ] || [ "$MODE" = "--env-only" ]; then
    if conda env list | grep -qE "^${NEW_ENV}\s"; then
        echo "conda env ${NEW_ENV} already exists -- skipping the clone (delete it to rebuild)"
    else
        echo "==== cloning ${SRC_ENV} -> ${NEW_ENV} ===="
        conda create -y -n "$NEW_ENV" --clone "$SRC_ENV"
    fi
    conda activate "$NEW_ENV"
    echo "==== installing transformers==${TRANSFORMERS_PIN} peft==${PEFT_PIN} ===="
    pip install --no-cache-dir "transformers==${TRANSFORMERS_PIN}" "peft==${PEFT_PIN}" "huggingface_hub[cli]"
    python - <<'PY'
import torch, transformers, peft
from transformers import Qwen3_5MoeForCausalLM, AutoTokenizer   # noqa: F401  (transformers>=5 only)
print("OK:", "torch", torch.__version__, torch.version.cuda, "| transformers", transformers.__version__, "| peft", peft.__version__)
print("CUDA available:", torch.cuda.is_available())
PY
    pip freeze > "$REPO/requirements-isambard-qwen-freeze.txt"
    echo "wrote $REPO/requirements-isambard-qwen-freeze.txt"
fi

if [ "$MODE" = "all" ] || [ "$MODE" = "--downloads-only" ]; then
    conda activate "$NEW_ENV"
    cd "$REPO"
    export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
    echo "==== model: Qwen/Qwen3.6-35B-A3B -> $HF_HOME (~72 GB) ===="
    hf download Qwen/Qwen3.6-35B-A3B --exclude "*.md" "*.png" "*.jpg" || huggingface-cli download Qwen/Qwen3.6-35B-A3B
    echo "==== eval datasets (HF cache + raw ECQA/CommonsenseQA files) ===="
    python - <<'PY'
from utils import setup_environment
from utils.data import load_exp_dataset
setup_environment()
# training / ID set + every OoD domain used by the arms / explanation read-out
for code in ["obqa_gen", "obqa", "arc_c", "arc_e", "sciq", "mmlu_law", "medexqa", "medmcqa_gen",
             "scienceqa", "ecqa", "aqua_rat"]:
    try:
        rows = load_exp_dataset(code, seed=42, split="test")
        print(f"cached {code}: {len(list(rows))} test rows")
    except Exception as e:   # noqa: BLE001 -- report, keep going
        print(f"WARNING: {code} failed: {type(e).__name__}: {e}")
PY
    echo "downloads done -- $(date)"
fi

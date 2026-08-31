#!/bin/bash
# ============================================================================
# One-time Isambard-AI setup for the Gemma 4 26B-A4B-it run (branch OBQA-gemma).
#
#   bash setup-isambard-gemma-env.sh                  # env check + downloads
#   bash setup-isambard-gemma-env.sh --env-check      # just verify the conda env
#   bash setup-isambard-gemma-env.sh --downloads-only # just model/dataset caches (used by STAGE=prep)
#
# Run inside a job, per the Isambard docs (login nodes must not run intensive work):
#   srun --nodes=1 --gpus=1 --time=02:00:00 --pty bash -c 'bash setup-isambard-gemma-env.sh'
#
# Env: NO new environment. The Qwen run's `qwen_env` (clone of moe_env with
# transformers 5.16.1 + peft 0.20.0) already ships Gemma4ForCausalLM, so it is
# reused; moe_env stays untouched (Granite byte-for-byte). Override with
# CONDA_ENV=... if you ever build a separate one.
# ============================================================================
set -euo pipefail
MODE="${1:-all}"
REPO="${REPO:-/projects/u6qd/moe-uncertainty-gemma}"
export HF_HOME="${HF_HOME:-/projects/u6qd/hf_cache}"
export MOE_RAW_DATA_DIR="${MOE_RAW_DATA_DIR:-/projects/u6qd/moe_raw_data}"
CONDA_ENV="${CONDA_ENV:-qwen_env}"
MODEL_ID="google/gemma-4-26B-A4B-it"
mkdir -p "$HF_HOME" "$MOE_RAW_DATA_DIR"

# Define `conda` without implicitly activating the first positional argument
# (bin/activate would forward this script's mode flag to `conda activate`).
source ~/miniforge3/etc/profile.d/conda.sh
conda env list | grep -qE "^${CONDA_ENV}\s" || { echo "ERROR: conda env ${CONDA_ENV} not found (run setup-isambard-qwen-env.sh --env-only first)" >&2; exit 1; }
conda activate "$CONDA_ENV"

if [ "$MODE" = "all" ] || [ "$MODE" = "--env-check" ]; then
    echo "==== env check: ${CONDA_ENV} ===="
    python - <<'PY'
import torch, transformers, peft
from transformers import Gemma4ForCausalLM, AutoTokenizer   # noqa: F401  (transformers>=5.5)
print("OK:", "torch", torch.__version__, torch.version.cuda, "| transformers", transformers.__version__, "| peft", peft.__version__)
print("CUDA available:", torch.cuda.is_available())
PY
    pip freeze > "$REPO/requirements-isambard-gemma-freeze.txt" 2>/dev/null && echo "wrote $REPO/requirements-isambard-gemma-freeze.txt" || true
fi

if [ "$MODE" = "all" ] || [ "$MODE" = "--downloads-only" ]; then
    cd "$REPO"
    export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
    echo "==== model: ${MODEL_ID} -> $HF_HOME (~52 GB bf16; HF_TOKEN from the env if the repo asks for it) ===="
    hf download "$MODEL_ID" || huggingface-cli download "$MODEL_ID"
    echo "==== tokenizer sanity (eos override, letter tokens, chat-template tail) ===="
    python - <<'PY'
from utils import setup_environment
from model import load_tokenizer
from utils.prompt import multiple_choice_prompt_engineer, MCQ_SYSTEM_INSTRUCTION
setup_environment()
tok = load_tokenizer("gemma4")
p = multiple_choice_prompt_engineer({"question": "Question: q?\nChoices:\nA. x\nB. y\nAnswer:", "answer": "A", "id": "x"},
                                    tokenizer=tok, system_instruction=MCQ_SYSTEM_INSTRUCTION)["question"]
assert p.endswith("<|turn>model\n<|channel>thought\n<channel|>"), p[-80:]
print("tokenizer OK: eos", repr(tok.eos_token), tok.eos_token_id, "| pad", repr(tok.pad_token), tok.pad_token_id,
      "| letters", [tok.convert_tokens_to_ids(c) for c in "ABCD"], "| prompt tail", repr(p[-45:]))
PY
    echo "==== eval datasets (HF cache + raw ECQA/CommonsenseQA files; already cached by the Qwen prep -> fast) ===="
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

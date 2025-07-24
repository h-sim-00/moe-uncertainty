#!/bin/bash
#SBATCH --job-name=eval_0-shot
#SBATCH --output=/vol/bitbucket/al1624/FIP/albus-bayesian-moe-router/logs/slurm/slurm_%j_eval_0-shot.log
#SBATCH --partition=AMD7-A100-T
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00 # This should be a relatively fast process
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=al1624@ic.ac.uk

# --- Setup ---
export HF_HOME="/vol/bitbucket/al1624/.cache/huggingface"
export HF_HUB_ENABLE_HF_TRANSFER=1

source /vol/bitbucket/al1624/.venv/moe_env/bin/activate
cd /vol/bitbucket/al1624/FIP/albus-bayesian-moe-router/
echo "Current working directory: $(pwd)"

cp ./scripts/python/eval_0-shot.py .

# --- Define Parameters ---
MODEL_SHORTCODE="qwen"

# --- Run 0-shot evaluation on all datasets ---
echo "----------------------------------------------------"
echo "0-shot evaluation on all datasets"
echo "----------------------------------------------------"
python eval_0-shot.py \
    --model_shortcode $MODEL_SHORTCODE

echo "0-shot Evaluation completed for model: $MODEL_SHORTCODE"
rm eval_0-shot.py
echo "Temporary script file removed."
echo "All tasks completed successfully."
# --- End of Script ---

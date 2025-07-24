#!/bin/bash
#SBATCH --job-name=eval_kvq-ft
#SBATCH --output=/vol/bitbucket/al1624/FIP/albus-bayesian-moe-router/logs/slurm/slurm_%j_eval_kvq-ft.log
#SBATCH --partition=gpgpuB
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=al1624@ic.ac.uk

# --- Setup ---
export HF_HOME="/vol/bitbucket/al1624/.cache/huggingface"

source /vol/bitbucket/al1624/.venv/moe_env/bin/activate
cd /vol/bitbucket/al1624/FIP/albus-bayesian-moe-router/
echo "Current working directory: $(pwd)"

cp ./scripts/python/eval_kvq-ft.py eval_kvq-ft.py

# --- Define Parameters ---
# Define the base model to evaluate
MODEL_SHORTCODE="granite"

# Define the seeds corresponding to the trained adapters
SEEDS=(42 43 44 45 46)

# --- Run Evaluation for Each Fine-tuned Adapter ---
echo "===================================================="
echo "Starting evaluation of fine-tuned adapters for model: $MODEL_SHORTCODE"
echo "===================================================="

for SEED in "${SEEDS[@]}"; do

    echo "----------------------------------------------------"
    echo "Evaluating adapter: $MODEL_SHORTCODE with seed: $SEED"
    echo "----------------------------------------------------"
    
    python eval_kvq-ft.py \
        --model_shortcode $MODEL_SHORTCODE \
        --seed $SEED

    echo "Evaluation completed for adapter: $MODEL_SHORTCODE with seed: $SEED"
done

rm eval_kvq-ft.py
echo "Temporary script file removed."
echo "All tasks completed successfully."
# --- End of Script ---
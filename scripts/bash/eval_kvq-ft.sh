#!/bin/bash
#SBATCH --job-name=eval_kvq-ft
#SBATCH --output=/vol/bitbucket/al1624/FIP/albus-bayesian-moe-router/logs/slurm/slurm_%j_eval_kvq-ft.log
#SBATCH --partition=gpgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00 # Evaluation should be faster than training
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=al1624@ic.ac.uk

# --- Setup ---
source ~/.bashrc
source /vol/bitbucket/al1624/.venv/moe_env/bin/activate
cd /vol/bitbucket/al1624/FIP/albus-bayesian-moe-router/
echo "Current working directory: $(pwd)"

cp ./scripts/python/eval_kvq-ft.py eval_kvq-ft.py

# --- Define Parameters ---
# Define the base model to evaluate
MODEL_SHORTCODE="granite"

# Define the seeds corresponding to the trained adapters
SEEDS=(42)

# --- Run Evaluation for Each Fine-tuned Adapter ---
echo "===================================================="
echo "Starting evaluation of fine-tuned adapters for model: $MODEL_SHORTCODE"
echo "===================================================="

for SEED in "${SEEDS[@]}"; do
    # Construct the path to the specific adapter for the current seed
    ADAPTER_PATH="./adapters/kvq_ft_${MODEL_SHORTCODE}_seed-${SEED}"

    echo "----------------------------------------------------"
    echo "Evaluating adapter: $ADAPTER_PATH"
    echo "----------------------------------------------------"
    
    # Check if the adapter directory exists before trying to run
    if [ ! -d "$ADAPTER_PATH" ]; then
        echo "Error: Adapter path not found: $ADAPTER_PATH"
        continue
    fi

    python eval_kvq-ft.py \
        --model_shortcode $MODEL_SHORTCODE \
        --adapter_path $ADAPTER_PATH

    echo "Evaluation completed for adapter: $ADAPTER_PATH"
done

rm eval_kvq-ft.py
echo "Temporary script file removed."
echo "All tasks completed successfully."
# --- End of Script ---
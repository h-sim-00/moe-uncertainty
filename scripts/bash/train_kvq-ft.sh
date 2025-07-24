#!/bin/bash
#SBATCH --job-name=train_kvq-ft
#SBATCH --output=/vol/bitbucket/al1624/FIP/albus-bayesian-moe-router/logs/slurm/slurm_%j_train_kvq-ft.log
#SBATCH --partition=AMD7-A100-T
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=al1624@ic.ac.uk

# --- Setup ---
export HF_HOME="/vol/bitbucket/al1624/.cache/huggingface"

source /vol/bitbucket/al1624/.venv/moe_env/bin/activate
cd /vol/bitbucket/al1624/FIP/albus-bayesian-moe-router/
echo "Current working directory: $(pwd)"

# Copy the training script to the current directory
# This is necessary to ensure the script can be executed in the current context
cp ./scripts/python/train_kvq-ft.py .

# --- Define Parameters ---
MODEL_SHORTCODE="granite" 

# Define training hyperparameters
EPOCHS=5
BATCH_SIZE=8

# Define the seeds to run the experiment with
SEEDS=(42 43 44 45 46)

# --- Run Fine-tuning for Each Seed ---
echo "===================================================="
echo "Starting fine-tuning runs for model: $MODEL_SHORTCODE"
echo "===================================================="

for SEED in "${SEEDS[@]}"; do
    echo "----------------------------------------------------"
    echo "Running fine-tuning with SEED: $SEED"
    echo "----------------------------------------------------"
    
    python train_kvq-ft.py \
        --model_shortcode $MODEL_SHORTCODE \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --seed $SEED

    echo "Fine-tuning completed for SEED: $SEED"
done

echo "All fine-tuning tasks completed successfully."
# Clean up temporary script file
rm train_kvq-ft.py
echo "Temporary script file removed."
# --- End of Script ---
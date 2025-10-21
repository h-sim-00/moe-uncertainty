#!/bin/bash
#SBATCH --job-name=kvq-tuning
#SBATCH --output=/vol/bitbucket/al1624/projects/bayesian-moe-router/logs/slurm/slurm_%j_train_kvq-ft.log
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
cd /vol/bitbucket/al1624/projects/bayesian-moe-router/
echo "Current working directory: $(pwd)"

# Copy the training script to the current directory
# This is necessary to ensure the script can be executed in the current context
cp ./scripts/python/kvq-tuning.py .

# --- Define Parameters ---
# MODEL_SHORTCODE="deepseek" 
# DATASET_SHORTCODES=("arc_c" "obqa" "arc_e" "sciq" "medmcqa_med" "mmlu_law")

MODEL_SHORTCODE="qwen" 
DATASET_SHORTCODES=("arc_c" "obqa" "arc_e" "sciq" "medmcqa_med" "mmlu_law")

# MODEL_SHORTCODE="granite" 
# DATASET_SHORTCODES=("arc_e" "mmlu_law")

# Define training hyperparameters
EPOCHS=10
BATCH_SIZE=8

# Define the seeds to run the experiment with
SEED=42

# --- Run Fine-tuning for Each Seed ---
echo "===================================================="
echo "Starting fine-tuning runs for model: $MODEL_SHORTCODE"
echo "===================================================="

for DATASET_SHORTCODE in "${DATASET_SHORTCODES[@]}"; do
    echo "----------------------------------------------------"
    echo "Running fine-tuning for DATASET: $DATASET_SHORTCODE"
    echo "----------------------------------------------------"
    
    python kvq-tuning.py \
        --model_shortcode $MODEL_SHORTCODE \
        --dataset_shortcode $DATASET_SHORTCODE \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --seed $SEED

    echo "Fine-tuning completed for DATASET: $DATASET_SHORTCODE"
done

echo "All fine-tuning tasks completed successfully."

rm kvq-tuning.py
echo "Temporary script file removed."
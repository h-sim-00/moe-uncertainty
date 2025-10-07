#!/bin/bash
#SBATCH --job-name=router-tuning
#SBATCH --output=/vol/bitbucket/al1624/FIP/albus-bayesian-moe-router/logs/slurm/slurm_%j_router_tuning.log
#SBATCH --partition=gpgpu
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
cp ./scripts/python/router-tuning.py .

# --- Define Parameters ---
MODEL_SHORTCODE=("granite")
DATASET_SHORTCODES=("arc_c")

# Define training hyperparameters
EPOCHS=10
BATCH_SIZE=8
SEED=42

# --- Run Fine-tuning for Each Combination ---
echo "===================================================="
echo "Starting MAP Router Tuning Runs"
echo "===================================================="

for DATASET_SHORTCODE in "${DATASET_SHORTCODES[@]}"; do

    BASE_ADAPTER_PATH="./adapters/${MODEL_SHORTCODE}-${DATASET_SHORTCODE}"

    echo "----------------------------------------------------"
    echo "Running: ${MODEL_SHORTCODE} on DATASET: ${DATASET_SHORTCODE}"
    echo "Using base adapter: ${BASE_ADAPTER_PATH}"
    echo "----------------------------------------------------"

    python router-tuning.py \
        --model_shortcode "$MODEL_SHORTCODE" \
        --dataset_shortcode "$DATASET_SHORTCODE" \
        --base_adapter_path "$BASE_ADAPTER_PATH" \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --seed "$SEED"

    echo "Router tuning completed for DATASET: $DATASET_SHORTCODE"
done

echo "All fine-tuning tasks completed successfully."

# Clean up temporary script file
rm router-tuning.py
echo "Temporary script file removed."
# --- End of Script ---
#!/bin/bash
#SBATCH --job-name=train_swag
#SBATCH --output=/vol/bitbucket/al1624/FIP/albus-bayesian-moe-router/logs/slurm/slurm_%j_train_swag.log
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

# --- Define Parameters ---
MODELS=("granite")
SEEDS=(42)
EPOCHS=10          # Initial fine-tuning epochs before SWAG collection
SWA_EPOCHS=5       # Epochs for SWAG weight collection
SWA_LR=0.01        # Learning rate for the SWAG phase
BATCH_SIZE=8

# --- Run Fine-tuning and SWAG Collection for Each Combination ---
echo "===================================================="
echo "Starting SWAG Router Training & Collection Runs"
echo "===================================================="

for MODEL_SHORTCODE in "${MODELS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo "----------------------------------------------------"
        echo "Running: ${MODEL_SHORTCODE} | Seed: ${SEED}"
        echo "----------------------------------------------------"

        python ./scripts/python/train_swag_router.py \
            --model_shortcode "$MODEL_SHORTCODE" \
            --seed "$SEED" \
            --epochs "$EPOCHS" \
            --swa_epochs "$SWA_EPOCHS" \
            --swa_lr "$SWA_LR" \
            --batch_size "$BATCH_SIZE"

        echo "SWAG training completed for ${MODEL_SHORTCODE} | Seed: ${SEED}"
    done
done

echo "All SWAG tasks completed successfully."
# --- End of Script ---
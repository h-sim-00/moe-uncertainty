#!/bin/bash
#SBATCH --job-name=train_laplace
#SBATCH --output=/vol/bitbucket/al1624/FIP/albus-bayesian-moe-router/logs/slurm/slurm_%j_train_laplace.log
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

cp ./scripts/python/train_laplace.py .

# --- Define Parameters ---
MODELS=("granite")
SEEDS=(42)
EPOCHS=10
BATCH_SIZE=8

# --- Run Fine-tuning and Fitting for Each Combination ---
echo "===================================================="
echo "Starting Laplace Router Training & Fitting Runs"
echo "===================================================="

for MODEL_SHORTCODE in "${MODELS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo "----------------------------------------------------"
        echo "Running: ${MODEL_SHORTCODE} | Seed: ${SEED}"
        echo "----------------------------------------------------"

        python train_laplace.py \
            --model_shortcode "$MODEL_SHORTCODE" \
            --seed "$SEED" \
            --epochs "$EPOCHS" \
            --batch_size "$BATCH_SIZE"

        echo "Training and fitting completed for ${MODEL_SHORTCODE} | Seed: ${SEED}"
    done
done

rm train_laplace.py

echo "All Laplace tasks completed successfully."
# --- End of Script ---
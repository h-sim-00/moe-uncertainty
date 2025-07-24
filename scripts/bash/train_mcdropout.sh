#!/bin/bash
#SBATCH --job-name=train_mcdropout
#SBATCH --output=/vol/bitbucket/al1624/FIP/albus-bayesian-moe-router/logs/slurm/slurm_%j_train_mcdropout.log
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
DROPOUT_RATES=(0.01 0.05 0.1 0.25)
EPOCHS=10
BATCH_SIZE=8

cp ./scripts/python/train_mcdropout.py .

# --- Run Fine-tuning for Each Combination ---
echo "===================================================="
echo "Starting MCDropout Router Fine-tuning Runs"
echo "===================================================="

for MODEL_SHORTCODE in "${MODELS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        for DOR in "${DROPOUT_RATES[@]}"; do
            echo "----------------------------------------------------"
            echo "Training: ${MODEL_SHORTCODE} | Seed: ${SEED} | Dropout: ${DOR}"
            echo "----------------------------------------------------"

            python train_mcdropout.py \
                --model_shortcode "$MODEL_SHORTCODE" \
                --seed "$SEED" \
                --dropout_rate "$DOR" \
                --epochs "$EPOCHS" \
                --batch_size "$BATCH_SIZE"

            echo "Training completed for ${MODEL_SHORTCODE} | Seed: ${SEED} | Dropout: ${DOR}"
        done
    done
done

# --- Cleanup ---
rm train_mcdropout.py
echo "Temporary files cleaned up."

echo "All fine-tuning tasks completed successfully."
# --- End of Script ---
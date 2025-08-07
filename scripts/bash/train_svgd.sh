#!/bin/bash
#SBATCH --job-name=train_svgd
#SBATCH --output=/vol/bitbucket/al1624/FIP/albus-bayesian-moe-router/logs/slurm/slurm_%j_train_svgd.log
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

cp ./scripts/python/train_svgd.py .

# --- Define Parameters ---
MODELS=("granite")
SEEDS=(42)
EPOCHS=5
BATCH_SIZE=4
LEARNING_RATE=1e-4
NUM_PARTICLES=20
WEIGHT_DECAY=0.01

# --- Run Fine-tuning for Each Combination ---
echo "===================================================="
echo "Starting SVGD Router Fine-tuning Runs"
echo "===================================================="

for MODEL_SHORTCODE in "${MODELS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo "----------------------------------------------------"
        echo "Running: ${MODEL_SHORTCODE} | Seed: ${SEED}"
        echo "----------------------------------------------------"

        python train_svgd.py \
            --model_shortcode "$MODEL_SHORTCODE" \
            --seed "$SEED" \
            --epochs "$EPOCHS" \
            --batch_size "$BATCH_SIZE" \
            --lr "$LEARNING_RATE" \
            --num_particles "$NUM_PARTICLES" \
            --weight_decay "$WEIGHT_DECAY"

        echo "SVGD training completed for ${MODEL_SHORTCODE} | Seed: ${SEED}"
    done
done

rm train_svgd.py

echo "All SVGD training tasks completed successfully."
# --- End of Script ---
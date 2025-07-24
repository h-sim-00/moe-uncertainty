#!/bin/bash
#SBATCH --job-name=train_mfvi
#SBATCH --output=/vol/bitbucket/al1624/FIP/albus-bayesian-moe-router/logs/slurm/slurm_%j_train_mfvi.log
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
MODELS=("granite" "deepseek" "qwen")
SEEDS=(42 43 44 45 46)
EPOCHS=5
BATCH_SIZE=4
LEARNING_RATE=1e-4

# --- Run Fine-tuning for Each Combination ---
echo "===================================================="
echo "Starting MFVI Router Fine-tuning Runs"
echo "===================================================="

for MODEL_SHORTCODE in "${MODELS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo "----------------------------------------------------"
        echo "Running: ${MODEL_SHORTCODE} | Seed: ${SEED}"
        echo "----------------------------------------------------"

        # Define the path to the required Stage 1 adapter
        BASE_ADAPTER_PATH="./adapters/kvq_ft_${MODEL_SHORTCODE}_seed-${SEED}"

        python ./scripts/python/train_mfvi_router.py \
            --model_shortcode "$MODEL_SHORTCODE" \
            --base_adapter_path "$BASE_ADAPTER_PATH" \
            --seed "$SEED" \
            --epochs "$EPOCHS" \
            --batch_size "$BATCH_SIZE" \
            --lr "$LEARNING_RATE"

        echo "MFVI training completed for ${MODEL_SHORTCODE} | Seed: ${SEED}"
    done
done

echo "All MFVI training tasks completed successfully."
# --- End of Script ---
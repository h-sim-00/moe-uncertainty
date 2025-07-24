#!/bin/bash
#SBATCH --job-name=eval_swag
#SBATCH --output=/vol/bitbucket/al1624/FIP/albus-bayesian-moe-router/logs/slurm/slurm_%j_eval_swag.log
#SBATCH --partition=gpgpuB
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
NUM_SAMPLES=(10 20 30)
BATCH_SIZE=8

# --- Run Evaluation for Each Combination ---
echo "===================================================="
echo "Starting Evaluation of SWAG Routers"
echo "===================================================="

for MODEL_SHORTCODE in "${MODELS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        for N_SAMPLES in "${NUM_SAMPLES[@]}"; do
            echo "----------------------------------------------------"
            echo "Evaluating: ${MODEL_SHORTCODE} | Seed: ${SEED} | Samples: ${N_SAMPLES}"
            echo "----------------------------------------------------"

            python ./scripts/python/evaluate_swag_router.py \
                --model_shortcode "$MODEL_SHORTCODE" \
                --num_samples "$N_SAMPLES" \
                --batch_size "$BATCH_SIZE" \
                --seed "$SEED" 

            echo "Evaluation completed."
        done
    done
done

echo "All evaluation tasks completed successfully."
# --- End of Script ---
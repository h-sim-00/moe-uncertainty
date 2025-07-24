#!/bin/bash
#SBATCH --job-name=eval_mfvi
#SBATCH --output=/vol/bitbucket/al1624/FIP/albus-bayesian-moe-router/logs/slurm/slurm_%j_eval_mfvi.log
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
MODELS=("granite" "deepseek" "qwen")
SEEDS=(42 43 44 45 46)
NUM_SAMPLES=(10 20 30)
BATCH_SIZE=8

# --- Run Evaluation for Each Combination ---
echo "===================================================="
echo "Starting Evaluation of MFVI Routers"
echo "===================================================="

for MODEL_SHORTCODE in "${MODELS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        for N_SAMPLES in "${NUM_SAMPLES[@]}"; do
            echo "----------------------------------------------------"
            echo "Evaluating: ${MODEL_SHORTCODE} | Seed: ${SEED} | Samples: ${N_SAMPLES}"
            echo "----------------------------------------------------"

            # Path to the Stage 1 adapter
            BASE_ADAPTER_PATH="./adapters/kvq_ft_${MODEL_SHORTCODE}_seed-${SEED}"

            # Path to the trained MFVI router weights
            RUN_NAME="MFVI_${MODEL_SHORTCODE}_seed-${SEED}"
            ROUTER_WEIGHTS_PATH="./models/routers/${RUN_NAME}/router_weights.pt"
            
            # Define a unique output file for this specific evaluation run
            RESULTS_CSV_PATH="./results/eval_${RUN_NAME}_n_samples-${N_SAMPLES}.csv"

            python ./scripts/python/evaluate_mfvi_router.py \
                --model_shortcode "$MODEL_SHORTCODE" \
                --base_adapter_path "$BASE_ADAPTER_PATH" \
                --router_weights_path "$ROUTER_WEIGHTS_PATH" \
                --num_samples "$N_SAMPLES" \
                --batch_size "$BATCH_SIZE" \
                --results_csv_path "$RESULTS_CSV_PATH"

            echo "Evaluation completed."
        done
    done
done

echo "All evaluation tasks completed successfully."
# --- End of Script ---
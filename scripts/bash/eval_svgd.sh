#!/bin/bash
#SBATCH --job-name=eval_svgd
#SBATCH --output=/vol/bitbucket/al1624/FIP/albus-bayesian-moe-router/logs/slurm/slurm_%j_eval_svgd.log
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

cp ./scripts/python/eval_svgd.py .

# --- Define Parameters ---
MODELS=("granite")
SEEDS=(42)
BATCH_SIZE=8

# --- Run Evaluation for Each Combination ---
echo "===================================================="
echo "Starting Evaluation of SVGD Routers"
echo "===================================================="

for MODEL_SHORTCODE in "${MODELS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo "----------------------------------------------------"
        echo "Evaluating: ${MODEL_SHORTCODE} | Seed: ${SEED}"
        echo "----------------------------------------------------"

        python eval_svgd.py \
            --model_shortcode "$MODEL_SHORTCODE" \
            --batch_size "$BATCH_SIZE" \
            --seed "$SEED" 

        echo "Evaluation completed."
    done
done

rm eval_svgd.py

echo "All evaluation tasks completed successfully."
# --- End of Script ---
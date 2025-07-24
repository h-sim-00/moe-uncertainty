#!/bin/bash
#SBATCH --job-name=eval_mcdropout
#SBATCH --output=/vol/bitbucket/al1624/FIP/albus-bayesian-moe-router/logs/slurm/slurm_%j_eval_mcdropout.log
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
DROPOUT_RATES=(0.01 0.05 0.1 0.25)
NUM_SAMPLES=(10 20 30)
BATCH_SIZE=8

cp ./scripts/python/eval_mcdropout.py .

# --- Run Evaluation for Each Combination ---
echo "===================================================="
echo "Starting Evaluation of MCDropout Routers"
echo "===================================================="

for MODEL_SHORTCODE in "${MODELS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        for DOR in "${DROPOUT_RATES[@]}"; do
            for N_SAMPLES in "${NUM_SAMPLES[@]}"; do
                echo "----------------------------------------------------"
                echo "Evaluating: ${MODEL_SHORTCODE} | Seed: ${SEED} | Dropout: ${DOR} | Samples: ${N_SAMPLES}"
                echo "----------------------------------------------------"


                python eval_mcdropout.py \
                    --model_shortcode "$MODEL_SHORTCODE" \
                    --seed "$SEED" \
                    --dropout_rate "$DOR" \
                    --num_samples "$N_SAMPLES" \
                    --batch_size "$BATCH_SIZE" \

                echo "Evaluation completed."
            done
        done
    done
done

# --- Cleanup ---
rm eval_mcdropout.py
echo "Temporary files cleaned up."

echo "All evaluation tasks completed successfully."
# --- End of Script ---
#!/bin/bash
#SBATCH --job-name=flops-analysis
#SBATCH --output=/vol/bitbucket/al1624/FIP/albus-bayesian-moe-router/logs/slurm/slurm_%j_flops-analysis.log
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

cp ./scripts/python/flops_analysis.py .

pip install fvcore

# --- Define Parameters ---
MODELS=("granite")
DATASETS=("obqa")
SEED=42

# --- Run FLOPs Analysis for Each Combination ---
echo "===================================================="
echo "Starting FLOPs Analysis Runs"
echo "===================================================="

for MODEL_SHORTCODE in "${MODELS[@]}"; do
    for DATASET_SHORTCODE in "${DATASETS[@]}"; do
        echo "----------------------------------------------------"
        echo "Analyzing: ${MODEL_SHORTCODE} trained on ${DATASET_SHORTCODE}"
        echo "----------------------------------------------------"

        # Path to the Stage 1 KVQ adapter
        KVQ_ADAPTER_PATH="./adapters/${MODEL_SHORTCODE}-${DATASET_SHORTCODE}"
        
        # Path to the MAP router weights needed for initialization
        MAP_RUN_NAME="${MODEL_SHORTCODE}_${DATASET_SHORTCODE}"
        MAP_ROUTER_PATH="./router_weights/base/${MAP_RUN_NAME}"

        # Define a unique output file for this run's results
        RESULTS_JSON_PATH="./results/flops_analysis_${MAP_RUN_NAME}.json"

        python flops_analysis.py \
            --model_shortcode "$MODEL_SHORTCODE" \
            --kvq_adapter_path "$KVQ_ADAPTER_PATH" \
            --map_router_path "$MAP_ROUTER_PATH" \
            --output_json_path "$RESULTS_JSON_PATH"

        echo "FLOPs analysis completed for ${MODEL_SHORTCODE} on ${DATASET_SHORTCODE}"
    done
done

rm flops_analysis.py
echo "Temporary script file removed."

echo "All FLOPs analysis tasks completed successfully."
# --- End of Script ---
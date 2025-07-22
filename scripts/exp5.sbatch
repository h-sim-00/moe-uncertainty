#!/bin/bash
#SBATCH --job-name=exp3-propagation-analysis
#SBATCH --output=/vol/bitbucket/al1624/FIP/albus-bayesian-moe-router/logs/slurm_%j_exp5_propagation_analysis.log
#SBATCH --partition=gpgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00 # Analysis should be faster than training
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=al1624@ic.ac.uk

# --- Setup ---
source ~/.bashrc
source /vol/bitbucket/al1624/.venv/moe_env/bin/activate
cd /vol/bitbucket/al1624/FIP/albus-bayesian-moe-router/

echo "Current working directory: $(pwd)"
echo "Starting Heuristic Propagation Analysis..."

# --- Define Parameters for the Analysis Run ---

ADAPTER_PATH="None" 
DATASET="arc_easy"
NUM_PROMPTS=50

# List of temperatures to run
TEMPS=(0.3 0.5 1.5 2.0)

# Build the adapter argument only if the path is not "None"
ADAPTER_ARG=""
if [ "$ADAPTER_PATH" != "None" ] && [ -n "$ADAPTER_PATH" ]; then
    ADAPTER_ARG="--adapter_path $ADAPTER_PATH"
fi

echo "----------------------------------------------------"
echo "Running analysis with the following settings:"
echo "Adapter: $ADAPTER_PATH"
echo "Dataset: $DATASET"
echo "Num Prompts: $NUM_PROMPTS"
echo "Temperatures: ${TEMPS[@]}"
echo "----------------------------------------------------"

for INTERVENTION_TEMP in "${TEMPS[@]}"; do
    echo "Running with intervention temperature: $INTERVENTION_TEMP"
    python exp5-prop-analysis.py \
        --dataset_name "$DATASET" \
        --num_prompts "$NUM_PROMPTS" \
        --intervention_temp "$INTERVENTION_TEMP" \
        $ADAPTER_ARG

    if [ $? -ne 0 ]; then
        echo "Propagation analysis FAILED for temperature $INTERVENTION_TEMP."
    else
        echo "Propagation analysis SUCCEEDED for temperature $INTERVENTION_TEMP. Heatmap saved to heuristic_propagation_heatmap.pdf"
    fi
done

echo "Job finished."

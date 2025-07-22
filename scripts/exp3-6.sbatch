#!/bin/bash
#SBATCH --job-name=exp3-laplace-fit
#SBATCH --output=/vol/bitbucket/al1624/FIP/albus-bayesian-moe-router/logs/slurm_%j_exp3_laplace_fit.log
#SBATCH --partition=gpgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00 # Fitting should be faster than training
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=al1624@ic.ac.uk

# --- Setup ---
source ~/.bashrc
source /vol/bitbucket/al1624/.venv/moe_env/bin/activate
cd /vol/bitbucket/al1624/FIP/albus-bayesian-moe-router/

echo "Current working directory: $(pwd)"
echo "Starting dedicated Laplace fitting workflow..."

# --- Define Parameters ---
PARAM_NUM="3b"
# The specific layers for which you have already trained MAP adapters
DATASETS_TO_EVAL=("arc_easy" "medmcqa") # ID and OoD datasets
TARGET_LAYERS=(4 20) 

# --- Loop through each pre-trained adapter and fit Laplace ---
for target_layer in "${TARGET_LAYERS[@]}"; do
    
    # Define paths based on the target layer
    # Assumes your MAP adapters were saved with this naming convention
    ADAPTER_PATH="./peft_adapters/laplace_map_layer_${target_layer}"
    LAPLACE_OUTPUT_FILE="./laplace_fits/laplace_layer_${target_layer}.pt"

    # Check if the required MAP adapter directory exists before trying to fit
    if [ ! -d "$ADAPTER_PATH" ]; then
        echo "SKIPPING Layer $target_layer: Pre-trained adapter not found at $ADAPTER_PATH"
        continue
    fi

    echo "----------------------------------------------------"
    echo "Starting Laplace Fit for TARGET LAYER: $target_layer"
    echo "Using adapter from: $ADAPTER_PATH"
    echo "----------------------------------------------------"
    
    python exp3-6-laplace-fit.py \
        --param_num "$PARAM_NUM" \
        --adapter_path "$ADAPTER_PATH" \
        --dataset_name "arc_easy" \
        --output_file "$LAPLACE_OUTPUT_FILE"

    if [ $? -ne 0 ]; then
        echo "Laplace fitting FAILED for TARGET LAYER: $target_layer"
    else
        echo "Laplace fitting SUCCEEDED for TARGET LAYER: $target_layer"
    fi
done

echo "Laplace fitting workflow finished for all target layers."

# =================================================================================
#                         STEP 2: EVALUATE EACH FITTED MODEL
# =================================================================================

echo -e "\n\n--- Starting Evaluation Phase ---"

for target_layer in "${TARGET_LAYERS[@]}"; do
    
    LAPLACE_FILE_PATH="./laplace_fits/laplace_layer_${target_layer}.pt"
    ADAPTER_PATH="./peft_adapters/laplace_map_layer_${target_layer}"

    if [ ! -f "$LAPLACE_FILE_PATH" ]; then
        echo "SKIPPING EVAL: Laplace file not found at $LAPLACE_FILE_PATH"
        continue
    fi
    
    for dataset in "${DATASETS_TO_EVAL[@]}"; do
        echo "----------------------------------------------------"
        echo "Evaluating Layer $target_layer on Dataset $dataset"
        echo "----------------------------------------------------"
        
        python exp3-5-laplace-evaluation.py \
            --laplace_path "$LAPLACE_FILE_PATH" \
            --adapter_path "$ADAPTER_PATH" \
            --dataset_name "$dataset" \
            --num_samples 20 \
            --batch_size 4
    done
done

echo "Full Laplace workflow finished."

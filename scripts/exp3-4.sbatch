#!/bin/bash
#SBATCH --job-name=exp3-laplace-workflow
#SBATCH --output=/vol/bitbucket/al1624/FIP/albus-bayesian-moe-router/logs/slurm_%j_exp3_laplace_workflow.log
#SBATCH --partition=gpgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=al1624@ic.ac.uk

# --- Setup ---
source ~/.bashrc
source /vol/bitbucket/al1624/.venv/moe_env/bin/activate
cd /vol/bitbucket/al1624/FIP/albus-bayesian-moe-router/

echo "Current working directory: $(pwd)"
echo "Starting FULL Laplace workflow for targeted layers..."

# --- Define Parameters ---
PARAM_NUM="3b"
TARGET_LAYERS=(4 20) # The specific layers we want to test
DATASETS_TO_EVAL=("arc_easy" "medmcqa") # ID and OoD datasets

# =================================================================================
#                         STEP 1: TRAIN & FIT FOR EACH TARGET LAYER
# =================================================================================

for target_layer in "${TARGET_LAYERS[@]}"; do
    echo "----------------------------------------------------"
    echo "Starting Train & Fit process for TARGET LAYER: $target_layer"
    echo "----------------------------------------------------"
    
    python exp3-4-laplace-train-and-fit.py \
        --param_num "$PARAM_NUM" \
        --target_layer "$target_layer"\
        --dataset_name arc_easy \
        --epochs 20 

    if [ $? -ne 0 ]; then
        echo "Train & Fit FAILED for TARGET LAYER: $target_layer"
        # Decide if you want to exit on failure or continue with the next layer
        # exit 1 
    else
        echo "Train & Fit SUCCEEDED for TARGET LAYER: $target_layer"
    fi
done

echo "All Laplace models have been fitted."

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

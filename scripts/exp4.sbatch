#!/bin/bash
#SBATCH --job-name=exp4-swag-workflow
#SBATCH --output=/vol/bitbucket/al1624/FIP/albus-bayesian-moe-router/logs/slurm_%j_exp4_swag_workflow.log
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
echo "Starting FULL SWAG workflow for targeted layers..."

# --- Define Parameters ---
PARAM_NUM="3b"
TARGET_LAYERS=(4 20) # The specific layers we want to test
DATASETS_TO_EVAL=("arc_easy" "medmcqa") # ID and OoD datasets

# =================================================================================
#                         STEP 1: TRAIN & COLLECT SWAG WEIGHTS
# =================================================================================

# for target_layer in "${TARGET_LAYERS[@]}"; do
#     echo "----------------------------------------------------"
#     echo "Starting SWAG Training process for TARGET LAYER: $target_layer"
#     echo "----------------------------------------------------"
    
#     python exp4-1-swag-finetuning.py \
#         --param_num "$PARAM_NUM" \
#         --target_layer "$target_layer"

#     if [ $? -ne 0 ]; then
#         echo "SWAG Training FAILED for TARGET LAYER: $target_layer"
#     else
#         echo "SWAG Training SUCCEEDED for TARGET LAYER: $target_layer"
#     fi
# done

# echo "All SWAG models have been trained and weights collected."

# =================================================================================
#                         STEP 2: EVALUATE EACH SWAG MODEL
# =================================================================================

echo -e "\n\n--- Starting SWAG Evaluation Phase ---"

for target_layer in "${TARGET_LAYERS[@]}"; do
    
    SWAG_FIT_DIR="./swag_fits/swag_layer_${target_layer}"

    if [ ! -f "${SWAG_FIT_DIR}/swag_weights.pt" ]; then
        echo "SKIPPING EVAL for Layer $target_layer: SWAG weights file not found."
        continue
    fi
    
    for dataset in "${DATASETS_TO_EVAL[@]}"; do
        echo "----------------------------------------------------"
        echo "Evaluating Layer $target_layer on Dataset $dataset"
        echo "----------------------------------------------------"
        
        python exp4-2-swag-evaluation.py \
            --param_num "$PARAM_NUM" \
            --swag_fit_dir "$SWAG_FIT_DIR" \
            --dataset_name "$dataset" \
            --target_layer "$target_layer" \
            --num_samples 20 \
            --batch_size 4
    done
done

echo "Full SWAG workflow finished."

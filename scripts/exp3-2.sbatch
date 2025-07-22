#!/bin/bash
#SBATCH --job-name=exp3-2-laplace-fit                # Job name
#SBATCH --output=/vol/bitbucket/al1624/FIP/albus-bayesian-moe-router/logs/slurm_%j_exp3-2_laplace_fit.log # Log file
#SBATCH --partition=gpgpuB                         # Partition
#SBATCH --gres=gpu:1                              # Request 1 GPU
#SBATCH --cpus-per-task=4                         # CPU cores
#SBATCH --mem=48G                                 # System RAM
#SBATCH --time=04:00:00                           # Max runtime
#SBATCH --mail-type=END,FAIL                      # Email notifications
#SBATCH --mail-user=al1624@ic.ac.uk               # Your email

# --- Setup ---
source ~/.bashrc
source /vol/bitbucket/al1624/.venv/moe_env/bin/activate
cd /vol/bitbucket/al1624/FIP/albus-bayesian-moe-router/

echo "Current working directory: $(pwd)"
echo "Starting Python Laplace fitting script..."

# --- Define Paths ---
# Path to the adapter trained in Phase 1
TARGET_LAYERS=(4 20) # The specific layers we want to test
ADAPTER_PATH="./peft_adapters/3b_router_map_arc_easy"
# Path where the final fitted Laplace object will be saved
OUTPUT_FILE="./laplace_objects/3b_router_laplace_arc_easy.pt"

# --- Execute the Fitting Script ---
python exp3-2-laplace-fit.py \
    --param_num 3b \
    --adapter_path "$ADAPTER_PATH" \
    --dataset_name arc_easy \
    --output_file "$OUTPUT_FILE"



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

echo "Laplace fitting job finished."

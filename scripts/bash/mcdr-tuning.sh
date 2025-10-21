#!/bin/bash
#SBATCH --job-name=mcdr_tuning_prog
#SBATCH --output=/vol/bitbucket/al1624/FIP/albus-bayesian-moe-router/logs/slurm/slurm_%j_mcdr_tuning_prog.log
#SBATCH --partition=gpgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00 # Increased time for multiple sequential runs
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=al1624@ic.ac.uk

# --- Setup ---
export HF_HOME="/vol/bitbucket/al1624/.cache/huggingface"
source /vol/bitbucket/al1624/.venv/moe_env/bin/activate
cd /vol/bitbucket/al1624/FIP/albus-bayesian-moe-router/
echo "Current working directory: $(pwd)"

cp ./scripts/python/mcdr-tuning.py .

# --- Define Parameters ---
MODELS=("qwen")
DATASETS=("arc_c")

# Define training hyperparameters
EPOCHS=5
BATCH_SIZE=8
SEED=42
DROPOUT_RATE=0.05
LR=1e-5

# --- Run Progressive Fine-tuning for Each Combination ---
echo "===================================================="
echo "Starting Progressive MCDropout Router Fine-tuning"
echo "===================================================="

for MODEL_SHORTCODE in "${MODELS[@]}"; do
    for DATASET_SHORTCODE in "${DATASETS[@]}"; do
        
        # Path to the Stage 1 adapter for the current model
        BASE_ADAPTER_PATH="./adapters/${MODEL_SHORTCODE}-${DATASET_SHORTCODE}"

        echo "----------------------------------------------------"
        echo "Running Progressive Training for: ${MODEL_SHORTCODE} on ${DATASET_SHORTCODE}"
        echo "----------------------------------------------------"

        # Loop backwards from the last layer (31) to the 27th layer
        for i in $(seq 31 -1 27); do
            
            # The layer to train in this step
            TRAIN_LAYER=$i
            
            # Build the list of all layers to be swapped to MCDR
            # This includes the current training layer and all subsequent layers
            # SWAP_LAYERS=$i
            SWAP_LAYERS=""
            for j in $(seq $i 31); do
                SWAP_LAYERS+="$j "
            done

            # Build the list of previously trained layers to load weights for
            # This includes all layers after the current training layer
            LOAD_LAYERS=""
            for j in $(seq $(($i + 1)) 31); do
                LOAD_LAYERS+="$j "
            done

            echo "    Step: Training Layer ${TRAIN_LAYER}"
            echo "    - Swapping Layers: [${SWAP_LAYERS}]"
            echo "    - Loading Pre-trained MCDR Layers: [${LOAD_LAYERS}]"

            # If LOAD_LAYERS is empty, the python script will handle it gracefully
            python mcdr-tuning.py \
                --model_shortcode "$MODEL_SHORTCODE" \
                --dataset_shortcode "$DATASET_SHORTCODE" \
                --base_adapter_path "$BASE_ADAPTER_PATH" \
                --swap_layers $SWAP_LAYERS \
                --load_layers $LOAD_LAYERS \
                --train_layers $TRAIN_LAYER \
                --epochs "$EPOCHS" \
                --batch_size "$BATCH_SIZE" \
                --seed "$SEED" \
                --dropout_rate "$DROPOUT_RATE" \
                --lr "$LR"
            
            echo "    Step for layer ${TRAIN_LAYER} completed."
        done
        echo "Progressive training completed for ${MODEL_SHORTCODE} on ${DATASET_SHORTCODE}"
    done
done

echo "All fine-tuning tasks completed successfully."

# Clean up
rm ./mcdr-tuning.py
echo "Temporary files cleaned up."

# --- End of Script ---
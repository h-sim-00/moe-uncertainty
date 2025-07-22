#!/bin/bash
#SBATCH --job-name=exp2-finetune-sweep              # Job name
#SBATCH --output=/vol/bitbucket/al1624/FIP/albus-bayesian-moe-router/logs/slurm_%j_exp2-1_finetune_sweep.log # Standard output and error log
#SBATCH --partition=gpgpu                         # Partition/queue to use
#SBATCH --gres=gpu:1                              # Request 1 GPU with specific memory
#SBATCH --cpus-per-task=2                         # Number of CPU cores per task
#SBATCH --mem=32G                                 # System memory required (RAM)
#SBATCH --time=24:00:00                           # Max runtime hh:mm:ss (Increased for sweep)
#SBATCH --mail-type=END,FAIL                      # Send email on job end or failure
#SBATCH --mail-user=al1624@ic.ac.uk               # Your email address

# --- Setup ---
source ~/.bashrc
source /vol/bitbucket/al1624/.venv/moe_env/bin/activate
cd /vol/bitbucket/al1624/FIP/albus-bayesian-moe-router/

echo "Current working directory: $(pwd)"
echo "Starting hyperparameter sweep for fine-tuning..."

# --- Define Hyperparameter Search Space ---
# LEARNING_RATES=(2e-4 5e-5 2e-5)
echo "Starting TARGETED hyperparameter sweep for fine-tuning..."

# --- Define Hyperparameter Search Space ---
LEARNING_RATES=(2e-4)
LORA_DROPOUTS=(0.1)
ROUTER_DROPOUTS=(0.05)
PARAM_NUM="3b"
FINETUNE_MODE="router"
DATASET_NAME="arc_easy"
TARGET_LAYERS=(4 20) 
EPOCHS=20

# --- Loop through all hyperparameter combinations ---
# NEW: Outer loop to iterate over the target layers
for target_layer in "${TARGET_LAYERS[@]}"; do
    for lr in "${LEARNING_RATES[@]}"; do
        for lora_dropout in "${LORA_DROPOUTS[@]}"; do
            for router_dropout in "${ROUTER_DROPOUTS[@]}"; do

                echo "----------------------------------------------------"
                echo "Starting run for TARGET LAYER: $target_layer with LR: $lr, LoRA Dropout: $lora_dropout, Router Dropout: $router_dropout"
                echo "----------------------------------------------------"

                # NEW: Experiment name now includes the target layer for clarity
                exp_name="${PARAM_NUM}_${FINETUNE_MODE}_layer_${target_layer}_lr_${lr}_ldrop_${lora_dropout}_rdrop_${router_dropout}"

                # Execute the Fine-tuning Script with the current set of hyperparameters
                python exp2-1-mc-dropout-finetuning.py \
                    --param_num "$PARAM_NUM" \
                    --finetune_mode "$FINETUNE_MODE" \
                    --dataset_name "$DATASET_NAME" \
                    --output_dir_base "./peft_adapters" \
                    --epochs "$EPOCHS" \
                    --learning_rate "$lr" \
                    --lora_dropout "$lora_dropout" \
                    --router_dropout_rate "$router_dropout" \
                    --target_layer "$target_layer" # NEW: Pass the target layer to the python script
                    
                # Check exit code
                if [ $? -ne 0 ]; then
                    echo "Run FAILED for TARGET LAYER: $target_layer"
                else
                    echo "Run SUCCEEDED for TARGET LAYER: $target_layer"
                fi
            done
        done
    done
done

echo "Hyperparameter sweep finished."
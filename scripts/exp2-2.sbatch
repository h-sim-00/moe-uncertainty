#!/bin/bash
#SBATCH --job-name=exp2-2-id-ood-eval               # Job name
#SBATCH --output=/vol/bitbucket/al1624/FIP/albus-bayesian-moe-router/logs/slurm_%j_exp2-2_id_ood_eval.log # Standard output and error log
#SBATCH --partition=gpgpu                         # Partition/queue to use
#SBATCH --gres=gpu:1                              # Request 1 GPU
#SBATCH --cpus-per-task=2                         # Number of CPU cores per task
#SBATCH --mem=32G                                 # System memory required (RAM)
#SBATCH --time=08:00:00                           # Max runtime hh:mm:ss
#SBATCH --mail-type=END,FAIL                      # Send email on job end or failure
#SBATCH --mail-user=al1624@ic.ac.uk               # Your email address

# --- Setup ---
source ~/.bashrc
source /vol/bitbucket/al1624/.venv/moe_env/bin/activate
cd /vol/bitbucket/al1624/FIP/albus-bayesian-moe-router/

echo "Current working directory: $(pwd)"
echo "Starting ID vs OOD evaluation..."

# --- Define Parameters ---
# IMPORTANT: Change this path to point to the specific adapter you trained and want to evaluate.
# --- Define Parameters to locate trained adapters ---
# These should match the parameters used in your training sweep script
LEARNING_RATES=(0.0002)
LORA_DROPOUTS=(0.1)
ROUTER_DROPOUTS=(0.05)
PARAM_NUM="3b"
FINETUNE_MODE="router"
TARGET_LAYERS=(4 20) # The specific layers you trained adapters for
EPOCHS=5 # Should match the training epochs

# --- Define Inference Parameters ---
NUM_SAMPLES=20
BATCH_SIZE=4

# --- Loop through all trained adapters ---
for target_layer in "${TARGET_LAYERS[@]}"; do
    for lr in "${LEARNING_RATES[@]}"; do
        for lora_dropout in "${LORA_DROPOUTS[@]}"; do
            for router_dropout in "${ROUTER_DROPOUTS[@]}"; do

                # --- Dynamically construct the path to the specific adapter ---
                # This naming convention MUST match the one from your training script
                ADAPTER_SUBDIR="mcdropout_${PARAM_NUM}_${FINETUNE_MODE}_lr_${lr}_ldrop_${lora_dropout}_rdrop_${router_dropout}_r_64_target_layer_${target_layer}"
                ADAPTER_PATH="./peft_adapters/${ADAPTER_SUBDIR}"

                # Check if the adapter directory actually exists before proceeding
                if [ ! -d "$ADAPTER_PATH" ]; then
                    echo "SKIPPING: Adapter path not found at $ADAPTER_PATH"
                    continue
                fi

                echo "===================================================="
                echo "Evaluating Adapter: $ADAPTER_SUBDIR"
                echo "===================================================="

                # --- 1. Run on In-Distribution (ID) Data ---
                echo "----------------------------------------------------"
                echo "Running evaluation on ID dataset (arc_easy)"
                echo "----------------------------------------------------"
                python exp2-2-mc-dropout-id-ood-evaluation.py \
                    --param_num "$PARAM_NUM" \
                    --adapter_path "$ADAPTER_PATH" \
                    --dataset_name arc_easy \
                    --num_samples "$NUM_SAMPLES" \
                    --router_dropout_rate "$router_dropout" \
                    --target_layer "$target_layer" \
                    --batch_size "$BATCH_SIZE"

                # --- 2. Run on Out-of-Distribution (OOD) Data ---
                echo "----------------------------------------------------"
                echo "Running evaluation on OOD dataset (medmcqa)"
                echo "----------------------------------------------------"
                python exp2-2-mc-dropout-id-ood-evaluation.py \
                    --param_num "$PARAM_NUM" \
                    --adapter_path "$ADAPTER_PATH" \
                    --dataset_name medmcqa \
                    --num_samples "$NUM_SAMPLES" \
                    --router_dropout_rate "$router_dropout" \
                    --target_layer "$target_layer" \
                    --batch_size "$BATCH_SIZE"

            done
        done
    done
done

echo "Targeted ID vs OOD evaluation finished."
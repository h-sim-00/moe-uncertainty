#!/bin/bash
#SBATCH --job-name=vtsr_tuning_prog
#SBATCH --output=/vol/bitbucket/al1624/FIP/albus-bayesian-moe-router/logs/slurm/slurm_%j_vtsr_tuning_prog.log
#SBATCH --partition=gpgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=al1624@ic.ac.uk

# --- Setup ---
export HF_HOME="/vol/bitbucket/al1624/.cache/huggingface"
source /vol/bitbucket/al1624/.venv/moe_env/bin/activate
cd /vol/bitbucket/al1624/FIP/albus-bayesian-moe-router/
echo "Current working directory: $(pwd)"

cp ./scripts/python/vtsr-tuning.py .

# --- Define Parameters ---
MODELS=("granite")
DATASET_SHORTCODES=("obqa" "sciq" "medmcqa_med")

# Define training hyperparameters
EPOCHS=10
BATCH_SIZE=8
SEED=42
TEMPERATURE_MODE="shared"  # Options: "shared", "per_expert"

# --- Run Progressive Fine-tuning for Each Combination ---
echo "===================================================="
echo "Starting Progressive VTSR Fine-tuning (Mode: ${TEMPERATURE_MODE})"
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
            
            TRAIN_LAYER=$i
            
            # Build the list of all layers to be swapped to VTSR
            SWAP_LAYERS=($i)
            # for j in $(seq $i 31); do SWAP_LAYERS+=($j); done

            # Build the list of previously trained layers to load weights for
            LOAD_LAYERS=($i)
            # for j in $(seq $(($i + 1)) 31); do LOAD_LAYERS+=($j); done

            echo "    Step: Training Layer ${TRAIN_LAYER}"
            echo "    - Swapping Layers: [${SWAP_LAYERS[@]}]"
            echo "    - Loading Pre-trained VTSR Layers: [${LOAD_LAYERS[@]}]"

            # This assumes your vtsr-tuning.py script is modified to accept
            # the layer-wise arguments, similar to the MFVR script.
            python vtsr-tuning.py \
                --model_shortcode "$MODEL_SHORTCODE" \
                --dataset_shortcode "$DATASET_SHORTCODE" \
                --base_adapter_path "$BASE_ADAPTER_PATH" \
                --temperature_mode "$TEMPERATURE_MODE" \
                --swap_layers "${SWAP_LAYERS[@]}" \
                --load_layers "${LOAD_LAYERS[@]}" \
                --train_layers "$TRAIN_LAYER" \
                --epochs "$EPOCHS" \
                --batch_size "$BATCH_SIZE" \
                --seed "$SEED"
            
            echo "    Step for layer ${TRAIN_LAYER} completed."
        done
        echo "Progressive training completed for ${MODEL_SHORTCODE} on ${DATASET_SHORTCODE}"
    done
done

rm vtsr-tuning.py
echo "Temporary script file removed."

echo "All fine-tuning tasks completed successfully."
# --- End of Script ---
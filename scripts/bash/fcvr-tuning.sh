#!/bin/bash
#SBATCH --job-name=fcvr_tuning_prog
#SBATCH --output=/vol/bitbucket/al1624/projects/bayesian-moe-router/logs/slurm/slurm_%j_fcvr_tuning_prog.log
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
cd /vol/bitbucket/al1624/projects/bayesian-moe-router/
echo "Current working directory: $(pwd)"

cp ./scripts/python/fcvr-tuning.py .


# --- Define Parameters ---
MODEL_SHORTCODE="granite"
DATASET_SHORTCODES=("arc_c")
SEED=42
EPOCHS=10
BATCH_SIZE=4
LEARNING_RATE=1e-5
BETA=0.01

for DATASET_SHORTCODE in "${DATASET_SHORTCODES[@]}"; do
    echo "===================================================="
    echo "Starting Progressive FCVR Fine-tuning"
    echo "Model: ${MODEL_SHORTCODE} | Dataset: ${DATASET_SHORTCODE}"
    echo "===================================================="

    BASE_ADAPTER_PATH="./adapters/${MODEL_SHORTCODE}-${DATASET_SHORTCODE}"
    
    # Loop backwards from the last layer (31) to the 27th layer
    for i in $(seq 31 -1 27); do
        TRAIN_LAYER=$i

        # Only swap layers to train
        # SWAP_LAYERS=($i)
        SWAP_LAYERS=()
        for j in $(seq $i 31); do SWAP_LAYERS+=($j); done

        # Do not load any layers
        # LOAD_LAYERS=($i)
        LOAD_LAYERS=()
        for j in $(seq $(($i + 1)) 31); do LOAD_LAYERS+=($j); done

        echo "----------------------------------------------------"
        echo "Step: Training Layer ${TRAIN_LAYER}"
        echo "  - Swapping Layers: [${SWAP_LAYERS[@]}]"
        echo "  - Loading Pre-trained FCVR Layers: [${LOAD_LAYERS[@]}]"
        echo "----------------------------------------------------"

        python fcvr-tuning.py \
            --model_shortcode "$MODEL_SHORTCODE" \
            --dataset_shortcode "$DATASET_SHORTCODE" \
            --base_adapter_path "$BASE_ADAPTER_PATH" \
            --swap_layers "${SWAP_LAYERS[@]}" \
            --load_layers "${LOAD_LAYERS[@]}" \
            --train_layers "$TRAIN_LAYER" \
            --epochs "$EPOCHS" \
            --batch_size "$BATCH_SIZE" \
            --lr "$LEARNING_RATE" \
            --beta "$BETA" \
            --seed "$SEED"
    done
done

echo "All progressive training tasks completed successfully."

rm fcvr-tuning.py
echo "Temporary script file removed."

# --- End of Script ---
#!/bin/bash
#SBATCH --job-name=fcvr-eval
#SBATCH --output=/vol/bitbucket/al1624/projects/bayesian-moe-router/logs/slurm/slurm_%j_fcvr-eval.log
#SBATCH --partition=gpgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00 # Increased time for the large number of evaluations
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=al1624@ic.ac.uk

# --- Setup ---
export HF_HOME="/vol/bitbucket/al1624/.cache/huggingface"
source /vol/bitbucket/al1624/.venv/moe_env/bin/activate
cd /vol/bitbucket/al1624/projects/bayesian-moe-router/
echo "Current working directory: $(pwd)"

cp ./scripts/python/evaluate.py .

# --- Define Parameters ---
MODEL_SHORTCODE="granite"
ID_DATASETS=("obqa" "medmcqa_med" "sciq")
SEED=42
BATCH_SIZE=8
NUM_SAMPLES=35

# Define the layer configurations to test
LAYER_CONFIGS=(
    "31"
    "30"
    "29"
    "28"
    "27"
    "30 31"
    "29 30 31"
    "28 29 30 31"
    "27 28 29 30 31"
)

# ====================================================
# Task 1: In-Distribution (ID) Calibration
# ====================================================
echo "===================================================="
echo "Starting Task 1: ID Calibration for FCVR"
echo "===================================================="

for DATASET_SHORTCODE in "${ID_DATASETS[@]}"; do
    # Path to the Stage 1 adapter
    BASE_ADAPTER_PATH="./adapters/${MODEL_SHORTCODE}-${DATASET_SHORTCODE}"
    
    # Path to the FCVR weights trained on this specific dataset
    RUN_NAME="fcvr-${MODEL_SHORTCODE}-${DATASET_SHORTCODE}"
    ROUTER_WEIGHTS_PATH="./router_weights/fcvr/${RUN_NAME}"

    for LAYER_SET in "${LAYER_CONFIGS[@]}"; do
        echo "----------------------------------------------------"
        echo "Eval ID Calib: ${MODEL_SHORTCODE} | Data: ${DATASET_SHORTCODE} | Layers: ${LAYER_SET}"
        echo "----------------------------------------------------"

        # Create a clean filename for the layer set
        LAYER_FILENAME=$(echo "$LAYER_SET" | tr ' ' '-')
        RESULTS_JSON_PATH="./results/fcvr/id_calib_${RUN_NAME}_layers-${LAYER_FILENAME}.json"

        python evaluate.py \
            --method "fcvr" \
            --task "id_calibration" \
            --model_shortcode "$MODEL_SHORTCODE" \
            --dataset_shortcode "$DATASET_SHORTCODE" \
            --kvq_adapter_path "$BASE_ADAPTER_PATH" \
            --router_weights_path "$ROUTER_WEIGHTS_PATH" \
            --output_json_path "$RESULTS_JSON_PATH" \
            --swap_layers $LAYER_SET \
            --num_samples "$NUM_SAMPLES" \
            --batch_size "$BATCH_SIZE" \
            --seed "$SEED"
    done
done

# ====================================================
# Task 2: Out-of-Distribution (OOD) Detection
# ====================================================
echo "===================================================="
echo "Starting Task 2: OOD Detection for FCVR (ID Model: obqa)"
echo "===================================================="

ID_DATASET_OOD="obqa"
BASE_ADAPTER_PATH="./adapters/${MODEL_SHORTCODE}-${ID_DATASET_OOD}"
RUN_NAME="fcvr-${MODEL_SHORTCODE}-${ID_DATASET_OOD}"
ROUTER_WEIGHTS_PATH="./router_weights/fcvr/${RUN_NAME}"

for LAYER_SET in "${LAYER_CONFIGS[@]}"; do
    echo "----------------------------------------------------"
    echo "Eval OOD: ${MODEL_SHORTCODE} | ID Data: ${ID_DATASET_OOD} | Layers: ${LAYER_SET}"
    echo "----------------------------------------------------"

    LAYER_FILENAME=$(echo "$LAYER_SET" | tr ' ' '-')
    RESULTS_JSON_PATH="./results/fcvr/ood_detect_${RUN_NAME}_layers-${LAYER_FILENAME}.json"

    python evaluate.py \
        --method "fcvr" \
        --task "ood_detection" \
        --model_shortcode "$MODEL_SHORTCODE" \
        --dataset_shortcode "$ID_DATASET_OOD" \
        --kvq_adapter_path "$BASE_ADAPTER_PATH" \
        --router_weights_path "$ROUTER_WEIGHTS_PATH" \
        --output_json_path "$RESULTS_JSON_PATH" \
        --swap_layers $LAYER_SET \
        --num_samples "$NUM_SAMPLES" \
        --batch_size "$BATCH_SIZE" \
        --seed "$SEED"
done

echo "All FCVR evaluation tasks completed successfully."
# --- End of Script ---
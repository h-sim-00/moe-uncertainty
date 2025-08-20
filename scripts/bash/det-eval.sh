#!/bin/bash
#SBATCH --job-name=det-eval
#SBATCH --output=/vol/bitbucket/al1624/FIP/albus-bayesian-moe-router/logs/slurm/slurm_%j_det-eval.log
#SBATCH --partition=gpgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=al1624@ic.ac.uk

# --- Setup ---
export HF_HOME="/vol/bitbucket/al1624/.cache/huggingface"
source /vol/bitbucket/al1624/.venv/moe_env/bin/activate
cd /vol/bitbucket/al1624/FIP/albus-bayesian-moe-router/
echo "Current working directory: $(pwd)"

# Copy the evaluation script to the current directory
cp ./scripts/python/evaluate.py .

# --- Define Parameters ---
MODEL_SHORTCODE="granite"
ID_DATASETS=("obqa" "medmcqa_med" "sciq")
SEED=42
BATCH_SIZE=8

# ====================================================
# Task 1: In-Distribution (ID) Calibration
# ====================================================
echo "===================================================="
echo "Starting Task 1: ID Calibration for ${MODEL_SHORTCODE}"
echo "===================================================="

for DATASET_SHORTCODE in "${ID_DATASETS[@]}"; do
    echo "----------------------------------------------------"
    echo "Evaluating ID Calibration on: ${DATASET_SHORTCODE}"
    echo "----------------------------------------------------"

    # Updated Path Logic
    BASE_ADAPTER_PATH="./adapters/${MODEL_SHORTCODE}-${DATASET_SHORTCODE}"
    RUN_NAME="${MODEL_SHORTCODE}_${DATASET_SHORTCODE}"
    ROUTER_WEIGHTS_PATH="./router_weights/base/${RUN_NAME}"
    RESULTS_JSON_PATH="./results/det/id_calib_det_${RUN_NAME}.json"

    python evaluate.py \
        --method "det" \
        --task "id_calibration" \
        --model_shortcode "$MODEL_SHORTCODE" \
        --dataset_shortcode "$DATASET_SHORTCODE" \
        --kvq_adapter_path "$BASE_ADAPTER_PATH" \
        --router_weights_path "$ROUTER_WEIGHTS_PATH" \
        --output_json_path "$RESULTS_JSON_PATH" \
        --batch_size "$BATCH_SIZE" \
        --seed "$SEED"

    echo "ID Calibration completed."
done

# ====================================================
# Task 2: Out-of-Distribution (OOD) Detection
# ====================================================
echo "===================================================="
echo "Starting Task 2: OOD Detection for ${MODEL_SHORTCODE} (ID Model: obqa)"
echo "===================================================="

# For OOD detection, the ID model is always the one trained on 'obqa'
ID_DATASET_OOD="obqa"

# Updated Path Logic
BASE_ADAPTER_PATH="./adapters/${MODEL_SHORTCODE}-${ID_DATASET_OOD}"
RUN_NAME="${MODEL_SHORTCODE}_${ID_DATASET_OOD}"
ROUTER_WEIGHTS_PATH="./models/routers/base/${RUN_NAME}"
RESULTS_JSON_PATH="./results/ood_detect_det_${RUN_NAME}.json"

python evaluate.py \
    --method "det" \
    --task "ood_detection" \
    --model_shortcode "$MODEL_SHORTCODE" \
    --dataset_shortcode "$ID_DATASET_OOD" \
    --kvq_adapter_path "$BASE_ADAPTER_PATH" \
    --router_weights_path "$ROUTER_WEIGHTS_PATH" \
    --output_json_path "$RESULTS_JSON_PATH" \
    --batch_size "$BATCH_SIZE" \
    --seed "$SEED"

echo "OOD Detection completed."

# Clean up the temporary script file
rm evaluate.py
echo "Temporary script file removed."

echo "All evaluation tasks completed successfully."
# --- End of Script ---
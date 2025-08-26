#!/bin/bash
#SBATCH --job-name=ts-combo-eval
#SBATCH --output=/vol/bitbucket/al1624/FIP/albus-bayesian-moe-router/logs/slurm/slurm_%j_ts-combo-eval.log
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

cp ./scripts/python/evaluate.py .

# --- Define Parameters ---
MODEL_SHORTCODE="granite"
ID_DATASETS=("obqa" "medmcqa_med" "sciq")
SEED=42
BATCH_SIZE=8

# --- Define Experiment-Specific Parameters ---
TEMPERATURE=0.3

# --- Run Evaluation for Each Combination ---
echo "===================================================="
echo "Starting Temperature Sampling Evaluation (Layer Combinations)"
echo "===================================================="


for DATASET_SHORTCODE in "${ID_DATASETS[@]}"; do
    # The MAP router is specific to the dataset it was tuned on
    BASE_ADAPTER_PATH="./adapters/${MODEL_SHORTCODE}-${DATASET_SHORTCODE}"
    RUN_NAME="${MODEL_SHORTCODE}_${DATASET_SHORTCODE}"
    ROUTER_WEIGHTS_PATH="./router_weights/base/${RUN_NAME}"

    # Loop from swapping 1 layer (31) up to swapping 10 layers (22-31)
    for num_layers_to_swap in $(seq 1 10); do
        
        start_layer=$((32 - num_layers_to_swap))
        
        # Build the list of layers to swap for this run
        SWAP_LAYERS=""
        for i in $(seq $start_layer 31); do
            SWAP_LAYERS+="$i "
        done

        echo "----------------------------------------------------"
        echo "Eval: ${MODEL_SHORTCODE} | Data: ${DATASET_SHORTCODE} | Temp: ${TEMPERATURE}"
        echo "Swapping Layers: [${SWAP_LAYERS}]"
        echo "----------------------------------------------------"

        # Define a unique output file for this specific evaluation run
        RESULTS_JSON_PATH="./results/temp_sampling/combo/id_calib_${RUN_NAME}_layers-${start_layer}-31_temp-${TEMPERATURE}.json"

        python evaluate.py \
            --method "temp_sampling" \
            --task "id_calibration" \
            --model_shortcode "$MODEL_SHORTCODE" \
            --dataset_shortcode "$DATASET_SHORTCODE" \
            --kvq_adapter_path "$BASE_ADAPTER_PATH" \
            --router_weights_path "$ROUTER_WEIGHTS_PATH" \
            --output_json_path "$RESULTS_JSON_PATH" \
            --swap_layers $SWAP_LAYERS \
            --temperature "$TEMPERATURE" \
            --batch_size "$BATCH_SIZE" \
            --seed "$SEED"

        echo "Evaluation completed."
    done
done

echo "All temperature sampling evaluation tasks completed successfully."
# --- End of Script ---
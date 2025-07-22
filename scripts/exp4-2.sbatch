#!/bin/bash
#SBATCH --job-name=exp4-2-swag-eval                 # Job name
#SBATCH --output=/vol/bitbucket/al1624/FIP/albus-bayesian-moe-router/logs/slurm_%j_exp4-2_swag_eval.log # Log file
#SBATCH --partition=gpgpuD                        # Partition
#SBATCH --gres=gpu:1                              # Request 1 GPU
#SBATCH --cpus-per-task=4                         # CPU cores
#SBATCH --mem=32G                                 # System RAM
#SBATCH --time=08:00:00                           # Max runtime
#SBATCH --mail-type=END,FAIL                      # Email notifications
#SBATCH --mail-user=al1624@ic.ac.uk               # Your email

# --- Setup ---
source ~/.bashrc
source /vol/bitbucket/al1624/.venv/moe_env/bin/activate
cd /vol/bitbucket/al1624/FIP/albus-bayesian-moe-router/

echo "Starting SWAG ID vs OOD evaluation..."

# --- Define Parameters ---
SWAG_WEIGHTS_PATH="./swag_weights/3b_router_arc_easy/swag_weights.pt"
NUM_SAMPLES=20

# --- 1. Run on In-Distribution (ID) Data ---
echo "----------------------------------------------------"
echo "Running evaluation on ID dataset (arc_easy)"
echo "----------------------------------------------------"
python exp4-2-swag-evaluation.py \
    --swag_weights_path "$SWAG_WEIGHTS_PATH" \
    --dataset_name arc_easy \
    --num_samples "$NUM_SAMPLES"

# --- 2. Run on Out-of-Distribution (OOD) Data ---
echo "----------------------------------------------------"
echo "Running evaluation on OOD dataset (medmcqa)"
echo "----------------------------------------------------"
python exp4-2-swag-evaluation.py \
    --swag_weights_path "$SWAG_WEIGHTS_PATH" \
    --dataset_name medmcqa \
    --num_samples "$NUM_SAMPLES"

echo "SWAG ID vs OOD evaluation finished."

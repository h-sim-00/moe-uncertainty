#!/bin/bash
#SBATCH --job-name=exp3-3-laplace-eval              # Job name
#SBATCH --output=/vol/bitbucket/al1624/FIP/albus-bayesian-moe-router/logs/slurm_%j_exp3-3_laplace_eval.log # Log file
#SBATCH --partition=gpgpuB                         # Partition
#SBATCH --gres=gpu:1                              # Request 1 GPU
#SBATCH --cpus-per-task=4                         # CPU cores
#SBATCH --mem=48G                                 # System RAM
#SBATCH --time=08:00:00                           # Max runtime
#SBATCH --mail-type=END,FAIL                      # Email notifications
#SBATCH --mail-user=al1624@ic.ac.uk               # Your email

# --- Setup ---
source ~/.bashrc
source /vol/bitbucket/al1624/.venv/moe_env/bin/activate
cd /vol/bitbucket/al1624/FIP/albus-bayesian-moe-router/

echo "Starting Laplace ID vs OOD evaluation..."

# --- Define Parameters ---
LAPLACE_PATH="./laplace_objects/3b_router_laplace_arc_easy.pt"
ADAPTER_PATH="./peft_adapters/3b_router_map_arc_easy"
NUM_SAMPLES=20

# --- 1. Run on In-Distribution (ID) Data ---
echo "----------------------------------------------------"
echo "Running evaluation on ID dataset (arc_easy)"
echo "----------------------------------------------------"
python exp3-3-laplace-evaluation.py \
    --param_num 3b \
    --laplace_path "$LAPLACE_PATH" \
    --adapter_path "$ADAPTER_PATH" \
    --dataset_name arc_easy \
    --num_samples "$NUM_SAMPLES"

# --- 2. Run on Out-of-Distribution (OOD) Data ---
# echo "----------------------------------------------------"
# echo "Running evaluation on OOD dataset (medmcqa)"
# echo "----------------------------------------------------"
# python exp3-3-laplace-evaluation.py \
    # --param_num 3b \
#     --adapter_path "$ADAPTER_PATH" \
#     --laplace_path "$LAPLACE_PATH" \
#     --dataset_name medmcqa \
#     --num_samples "$NUM_SAMPLES"

echo "Laplace ID vs OOD evaluation finished."

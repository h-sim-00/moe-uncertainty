#!/bin/bash
#SBATCH --job-name=exp7-token-collection
#SBATCH --output=/vol/bitbucket/al1624/FIP/albus-bayesian-moe-router/logs/slurm_%j_exp7_token_collection.log
#SBATCH --partition=gpgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=al1624@ic.ac.uk

# --- Setup ---
source ~/.bashrc
source /vol/bitbucket/al1624/.venv/moe_env/bin/activate
cd /vol/bitbucket/al1624/FIP/albus-bayesian-moe-router/

echo "Current working directory: $(pwd)"
echo "Starting Token-Level Expert Specialisation Data Collection..."

# --- Define Parameters ---
OUTPUT_DIR="./figs/phase-4.5"
NUM_PROMPTS=500 # A larger number of prompts will yield better statistics

# --- 1. Run on In-Distribution (ID) Data ---
echo "----------------------------------------------------"
echo "Collecting data for ID dataset (arc_easy)"
echo "----------------------------------------------------"
python exp7-token-specialisation-collection.py \
    --dataset_name arc_easy \
    --num_prompts "$NUM_PROMPTS" \
    --output_dir "$OUTPUT_DIR"

# --- 2. Run on Out-of-Distribution (OOD) Data ---
echo "----------------------------------------------------"
echo "Collecting data for OOD dataset (medmcqa)"
echo "----------------------------------------------------"
python exp7-token-specialisation-collection.py \
    --dataset_name medmcqa \
    --num_prompts "$NUM_PROMPTS" \
    --output_dir "$OUTPUT_DIR"

echo "Data collection for both datasets finished."

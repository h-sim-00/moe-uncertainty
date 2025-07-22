#!/bin/bash
#SBATCH --job-name=exp6-logit-collection
#SBATCH --output=/vol/bitbucket/al1624/FIP/albus-bayesian-moe-router/logs/slurm_%j_exp6_logit_collection.log
#SBATCH --partition=gpgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00 # This should be a relatively fast process
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=al1624@ic.ac.uk

# --- Setup ---
source ~/.bashrc
source /vol/bitbucket/al1624/.venv/moe_env/bin/activate
cd /vol/bitbucket/al1624/FIP/albus-bayesian-moe-router/

echo "Current working directory: $(pwd)"
echo "Starting Observational Study Data Collection..."

# --- Define Parameters ---
# We will use the base pre-trained model for this analysis, so no adapter is needed.
OUTPUT_DIR="./figs/phase-4"
NUM_PROMPTS=150 # Use a sufficient number of prompts for stable statistics

# --- 1. Run on In-Distribution (ID) Data ---
echo "----------------------------------------------------"
echo "Collecting data for ID dataset (arc_easy)"
echo "----------------------------------------------------"
python exp6-router-logits-dist-analysis.py \
    --dataset_name arc_easy \
    --num_prompts "$NUM_PROMPTS" \
    --output_dir "$OUTPUT_DIR"

# --- 2. Run on Out-of-Distribution (OOD) Data ---
echo "----------------------------------------------------"
echo "Collecting data for OOD dataset (medmcqa)"
echo "----------------------------------------------------"
python exp6-router-logits-dist-analysis.py \
    --dataset_name medmcqa \
    --num_prompts "$NUM_PROMPTS" \
    --output_dir "$OUTPUT_DIR"

echo "Data collection for both datasets finished."

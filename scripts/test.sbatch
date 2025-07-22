#!/bin/bash
#SBATCH --job-name=exp7-token-collection
#SBATCH --output=/vol/bitbucket/al1624/FIP/albus-bayesian-moe-router/logs/slurm_%j_exp7_token_collection.log
#SBATCH --partition=AMD7-A100-T
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

echo "Test for allocating A100."

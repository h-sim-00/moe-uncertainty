#!/bin/bash
#SBATCH --job-name=debug-paths
#SBATCH --output=debug_paths_output.log
#SBATCH --time=00:05:00
#SBATCH --mem=1G

# --- Setup ---
source ~/.bashrc
source /vol/bitbucket/al1624/.venv/moe_env/bin/activate
cd /vol/bitbucket/al1624/FIP/albus-bayesian-moe-router/

# --- Run the diagnostic script ---
python debug.py
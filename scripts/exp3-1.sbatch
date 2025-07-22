#!/bin/bash
#SBATCH --job-name=exp3-1-laplace-finetuing                # Job name
#SBATCH --output=/vol/bitbucket/al1624/FIP/albus-bayesian-moe-router/logs/slurm_%j_exp3-1_laplace_finetuning.log # Log file
#SBATCH --partition=gpgpuB                         # Partition
#SBATCH --gres=gpu:1                              # Request 1 GPU
#SBATCH --cpus-per-task=2                         # CPU cores
#SBATCH --mem=8G                                 # System RAM
#SBATCH --time=08:00:00                           # Max runtime
#SBATCH --mail-type=END,FAIL                      # Email notifications
#SBATCH --mail-user=al1624@ic.ac.uk               # Your email

# --- Setup ---
source ~/.bashrc
source /vol/bitbucket/al1624/.venv/moe_env/bin/activate
cd /vol/bitbucket/al1624/FIP/albus-bayesian-moe-router/

echo "Current working directory: $(pwd)"
echo "Starting Python fine-tuning script for Laplace MAP estimation..."

# --- Execute the Fine-tuning Script ---
python exp3-1-laplace-finetuning.py \
    --param_num 3b \
    --finetune_mode router \
    --dataset_name arc_easy \
    --output_dir ./peft_adapters/3b_router_map_arc_easy \
    --epochs 5 \
    --learning_rate 2e-4

echo "MAP fine-tuning job finished."

#!/bin/bash
#SBATCH --job-name=exp2-swag-train                # Job name
#SBATCH --output=/vol/bitbucket/al1624/FIP/albus-bayesian-moe-router/logs/slurm_%j_exp4-1_swag_finetuning.log # Log file
#SBATCH --partition=gpgpuB                         # Partition
#SBATCH --gres=gpu:1                              # Request 1 GPU
#SBATCH --cpus-per-task=4                         # CPU cores
#SBATCH --mem=48G                                 # System RAM
#SBATCH --time=12:00:00                           # Max runtime (increased for custom loop)
#SBATCH --mail-type=END,FAIL                      # Email notifications
#SBATCH --mail-user=al1624@ic.ac.uk               # Your email

# --- Setup ---
source ~/.bashrc
source /vol/bitbucket/al1624/.venv/moe_env/bin/activate
cd /vol/bitbucket/al1624/FIP/albus-bayesian-moe-router/

echo "Current working directory: $(pwd)"
echo "Starting Python SWAG fine-tuning script..."

# --- Execute the Fine-tuning Script ---
python exp4-1-swag-finetuning.py \
    --param_num 3b \
    --finetune_mode router \
    --dataset_name arc_easy \
    --output_dir ./swag_weights/3b_router_arc_easy \
    --warmup_epochs 3 \
    --swag_epochs 5 \
    --learning_rate 2e-5 \
    --swag_learning_rate 1e-5

echo "SWAG fine-tuning job finished."

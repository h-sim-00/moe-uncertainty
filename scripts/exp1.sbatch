#!/bin/bash
#SBATCH --job-name=exp1-per_layer_sequential              # Job name
#SBATCH --output=/vol/bitbucket/al1624/FIP/albus-bayesian-moe-router/logs/slurm_%j_exp1_sequential.log # Standard output and error log (%j = job ID)
#SBATCH --partition=gpgpu                                 # Partition/queue to use
#SBATCH --gres=gpu:1                                 # Request 1 GPU with specific memory
#SBATCH --cpus-per-task=4                                 # Number of CPU cores per task
#SBATCH --mem=16G                                         # System memory required (RAM)
#SBATCH --time=06:00:00                                   # Max runtime hh:mm:ss
#SBATCH --mail-type=END,FAIL                              # Send email on job end or failure
#SBATCH --mail-user=al1624@ic.ac.uk                       # Your email address

# --- Setup ---
source ~/.bashrc
source /vol/bitbucket/al1624/.venv/moe_env/bin/activate
cd /vol/bitbucket/al1624/FIP/albus-bayesian-moe-router/

echo "Current working directory: $(pwd)"
echo "Starting Python script sequence..."

# --- Execute Python scripts sequentially ---
echo "Running for temp=0.3"
python exp1-per_layer.py --param_num 3b --mode sample_k --temp 0.3 --dataset_name arc_easy

echo "Running for temp=0.7"
python exp1-per_layer.py --param_num 3b --mode sample_k --temp 0.7 --dataset_name arc_easy

echo "Running for temp=1.0"
python exp1-per_layer.py --param_num 3b --mode sample_k --temp 1.0 --dataset_name arc_easy

echo "Running for temp=1.5"
python exp1-per_layer.py --param_num 3b --mode sample_k --temp 1.5 --dataset_name arc_easy

echo "Running for temp=2.0"
python exp1-per_layer.py --param_num 3b --mode sample_k --temp 2.0 --dataset_name arc_easy


echo "All jobs finished."

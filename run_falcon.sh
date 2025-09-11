#!/bin/bash
#SBATCH --time=1:00:00               # Max runtime (hh:mm:ss)
#SBATCH --partition=gpu_a100         # GPU partition
#SBATCH --gpus=1                     # Number of GPUs
#SBATCH --ntasks=1                   # Number of tasks

# Usage: sbatch run_falcon.sh /abs/path/to/config [falcon_args...]

#source ~/.bashrc
#conda activate emri_few

# ---- Argument handling ----
CONFIG_PATH=$1
shift                     # remove first arg, leave the rest as ARGS

if [ -z "$CONFIG_PATH" ]; then
    echo "Error: CONFIG_PATH not provided"
    exit 1
fi

# ---- Setup environment ----
source ~/.bashrc
conda activate emri_few

# ---- Move to config dir ----
cd "$CONFIG_PATH" || { echo "Failed to cd into $CONFIG_PATH"; exit 1; }

# ---- Run Falcon ----
echo "Launching Falcon in $(pwd) with args: $*"
srun falcon launch "$@"
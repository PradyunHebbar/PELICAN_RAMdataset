#!/bin/bash
#SBATCH --job-name=pelican_nobeam_training_16M    # Job name
#SBATCH --account=m3246                     # Project to be billed
#SBATCH --qos=regular                       # Quality of Service
#SBATCH --time=48:00:00                     # Wall clock time limit (HH:MM:SS)
#SBATCH --nodes=1                           # Number of nodes
#SBATCH --ntasks-per-node=4                 # Number of tasks per node (one per GPU)
#SBATCH --gpus-per-node=4 
#SBATCH --cpus-per-task=32                  # Number of CPU cores per task
#SBATCH -C gpu                    # Use GPU nodes
#SBATCH --output=slurm_logs/pelican2_16M_%j.out   # Standard output and error log
#SBATCH -e slurm_logs/pelican2_16M_%j.out

# Load necessary modules
module load python
module load pytorch/2.3.1

# Activate conda environment
conda activate pelican

export MASTER_ADDR=$(hostname)
cmd="python3 train_pelican_classifier.py --datadir=../atlas_data/atlas_16M --ram_split=test,valid --target=is_signal --nobj=80 --nobj-avg=56 --num-epoch=14 --num-train=-1 --num-valid=-1 --num-test=-1  --batch-size=256 --prefix=classifier --optim=adamw --activation=leakyrelu --factorize --lr-decay-type=warm --lr-init=0.0025 --lr-final=1e-6 --drop-rate=0.05 --drop-rate-out=0.05 --weight-decay=0.005 --summarize"

# Launch the training script using srun
set -x
srun -l \
    bash -c "
    source export_DDP_vars.sh
    $cmd
    "

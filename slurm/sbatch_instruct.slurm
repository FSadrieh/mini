#!/bin/bash
#SBATCH --job-name=Fettuccine
#SBATCH --output=slurm/output_instruct.txt
#SBATCH --error=slurm/error_instruct.txt
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=10
#SBATCH --time=1-00:00:00                # Maximum runtime (D-HH:MM:SS)
#SBATCH --mail-type=END
#SBATCH --mail-user=frederic.sadrieh@student.hpi.uni-potsdam.de

source ~/.bash_profile

conda activate tune

tune run --nnodes 1 --nproc-per-node 4 custom_full_finetune_distributed.py --config cfgs/custom_config_instruct.yaml

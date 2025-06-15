#!/bin/bash
#SBATCH --job-name=mini
#SBATCH --output=slurm/output_2.txt
#SBATCH --error=slurm/error_2.txt
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=10
#SBATCH --time=1-00:00:00                # Maximum runtime (D-HH:MM:SS)
#SBATCH --mail-type=END
#SBATCH --mail-user=frederic.sadrieh@student.hpi.uni-potsdam.de

source ~/.bash_profile

conda activate tune

tune run custom_full_finetune.py --config cfgs/custom_config_single.yaml
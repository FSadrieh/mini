#!/bin/bash
source ~/.bash_profile
conda activate tune
tune run --nnodes 1 --nproc-per-node 2 custom_full_finetune_distributed.py  --config cfgs/custom_config.yaml
#!/bin/bash

# specify conda env 
conda activate cellmil

# specify the script 
eval_report --metrics f1 recall auroc precision c_index --team wandb-team --projects 'CELLMIL' --output-dir ./INT_performance
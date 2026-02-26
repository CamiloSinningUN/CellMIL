#!/bin/bash

# specify conda env 
conda activate cellmil

# specify the script 
eval_external --metrics f1 recall c_index precision auroc  --models-dir ./results --output-dir ./UOC_performance --final-model ensemble --aggregation-method everything --dataset-dir ./dataset_UOC --root-dir ./UOC_MIL_dataset --dp-metadata-file ./data/metadata_UOC.xlsx
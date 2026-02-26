#!/bin/bash

# specify conda env 
conda activate cellmil

# create dataset
dataset_creation --excel_path ./data/metadata_UOC.xlsx --output_path ./dataset_UOC --gpu 0 --segmentation_models cellvit --extractors resnet50 gigapath uni morphometrics pyradiomics_hed connectivity geometric --graph_methods delaunay_radius

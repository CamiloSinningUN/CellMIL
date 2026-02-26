#!/bin/bash

# specify conda env 
conda activate cellmil

# create dataset
dataset_creation --excel_path ./data/metadata_INT.xlsx --output_path ./dataset_INT --gpu 0 --segmentation_models cellvit --extractors resnet50 gigapath uni morphometrics pyradiomics_hed connectivity geometric --graph_methods delaunay_radius

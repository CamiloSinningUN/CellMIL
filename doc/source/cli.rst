=========
CLI Tools
=========

This package provides a comprehensive set of command-line interface (CLI) tools for digital pathology analysis. Each tool is designed to handle a specific part of the analysis pipeline.

.. note::
   For detailed documentation of each tool, see the :doc:`pipeline/index` section which provides comprehensive guides including Python API usage, troubleshooting, and advanced features.

Overview
========

All CLI commands follow this format:

.. code-block:: bash

   <command> [options]

Available commands:

- ``patch_extraction`` - Extract patches from whole slide images
- ``cell_segmentation`` - Segment cells in extracted patches
- ``graph_creation`` - Create spatial graphs from segmented cells
- ``feature_extraction`` - Extract features from segmented cells
- ``vis_features`` - Visualize extracted features
- ``create_dataset`` - Create datasets for training


Quick Reference
==============

For complete documentation with Python API usage, troubleshooting guides, and advanced features, see the dedicated pages:

- :doc:`pipeline/patch_extraction` - Detailed patch extraction guide
- :doc:`pipeline/cell_segmentation` - Complete cell segmentation documentation
- :doc:`pipeline/graph_creation` - Graph creation from segmented cells
- :doc:`pipeline/feature_extraction` - Comprehensive feature extraction guide
- :doc:`pipeline/dataset_creation` - Dataset creation and management

Basic Usage Examples
===================

Patch Extraction
================

Extract patches from whole slide images. For complete documentation, see :doc:`pipeline/patch_extraction`.

.. code-block:: bash

   patch_extraction \
       --output_path ./results \
       --wsi_path ./data/C3L-00001-21.svs \
       --patch_size 1024 \
       --patch_overlap 6.25 \
       --target_mag 20.0

Cell Segmentation
=================

Segment individual cells within extracted patches. For complete documentation, see :doc:`pipeline/cell_segmentation`.

.. code-block:: bash

   cell_segmentation \
       --model cellvit \
       --gpu 0 \
       --wsi_path ./data/C3L-00001-21.svs \
       --patched_slide_path ./results/C3L-00001-21

**Available models**: ``cellvit``, ``hovernet``, ``cellpose_sam``

Graph Creation
===============

Create spatial graphs from segmented cells. For complete documentation, see :doc:`pipeline/graph_creation`.

.. code-block:: bash

   graph_creation \
       --method knn \
       --gpu 0 \
       --patched_slide_path ./results/C3L-00001-21 \
       --segmentation_model cellvit

**Available methods**: ``knn``, ``delaunay_radius``, ``radius``, ``dilate``, ``similarity``

Feature Extraction  
==================

Extract morphological and radiomics features from segmented cells. For complete documentation, see :doc:`pipeline/feature_extraction`.

.. code-block:: bash

   feature_extraction \
       --extractor connectivity \
       --wsi_path ./data/C3L-00001-21.svs \
       --patched_slide_path ./results/C3L-00001-21 \
       --segmentation_model cellvit \
       --graph_method knn

**Available extractors**: ``pyradiomics_gray``, ``pyradiomics_hed``, ``pyradiomics_hue``, ``morphometrics``, ``connectivity``, ``geometric``, ``resnet50``, ``gigapath``, ``uni``

Dataset Creation
===============

Create datasets for MIL training from multiple slides. For complete documentation, see :doc:`pipeline/dataset_creation`.

.. code-block:: bash

   create_dataset \
       --excel_path ./data/metadata.xlsx \
       --output_path ./datasets/training_set \
       --gpu 0 \
       --segmentation_models cellvit hovernet cellpose_sam \
       --extractors morphometrics pyradiomics_hed \
       --graph_method knn radius


Feature Visualization
====================

Visualize extracted features for quality control and analysis.

.. code-block:: bash

   vis_features \
       --patched_slide_path ./results/C3L-00001-21


Complete Pipeline Example
========================

Here's a complete workflow from slide to prediction:

.. code-block:: bash

   # 1. Extract patches
   patch_extraction \
       --output_path ./results \
       --wsi_path ./data/slide.svs \
       --patch_size 1024 \
       --patch_overlap 6.25 \
       --target_mag 20.0

   # 2. Segment cells
   cell_segmentation \
       --model cellvit \
       --gpu 0 \
       --wsi_path ./data/slide.svs \
       --patched_slide_path ./results/slide

   # 3. Create graphs
   graph_creation \
       --method knn \
       --gpu 0 \
       --patched_slide_path ./results/slide \
       --segmentation_model cellvit

   # 4. Extract features
   feature_extraction \
       --extractor morphometrics \
       --wsi_path ./data/slide.svs \
       --patched_slide_path ./results/slide \
       --segmentation_model cellvit \
       --graph_method knn


Getting Help
============

Each command provides detailed help:

.. code-block:: bash

   patch_extraction --help
   cell_segmentation --help
   graph_creation --help
   feature_extraction --help
   create_dataset --help

For comprehensive guides with Python API usage, troubleshooting, and advanced features, see :doc:`pipeline/index`.

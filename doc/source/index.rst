=======================================
Cell Instance Segmentation for Multiple Instance Learning in Digital Pathology
=======================================

Python package for cell instance segmentation and multiple instance learning (MIL) in digital pathology. It provides a complete pipeline from patch extraction to MIL model training and prediction.

.. image:: https://img.shields.io/badge/python-3.10+-blue.svg
   :alt: Python Version
   :target: https://python.org

.. figure:: _static/overview.png
   :alt: Cell Instance Segmentation Pipeline Overview
   :width: 100%
   :align: center

   Overview of the cell instance segmentation and multiple instance learning pipeline.

Features
========

- **Patch Extraction**: Extract patches from whole slide images (WSI) with configurable parameters
- **Cell Segmentation**: Multiple segmentation models including CellViT, HoverNet, and Cellpose
- **Graph Creation**: Create spatial graphs from segmented cells for improved context understanding
- **Feature Extraction**: Extract morphological and radiomics features from segmented cells
- **Visualization**: Comprehensive feature visualization tools

Quick Start
===========

Installation (Linux)
------------

.. code-block:: bash
   
   # Create environment
   conda env create -f environments/environment_cellmil.yml

   # Activate environment
   conda activate cellmil

   # Install python dependencies
   poetry install

   # Install Pytorch Geomatric Dependencies (Issues could arise when installing)
   pip install --no-binary :all: torch-cluster torch-scatter pyg-lib


Installation (Windows)
------------

.. code-block:: bash
   
   # Create environment
   conda env create -f environments/environment_cellmil_win.yml

   # Activate environment
   conda activate cellmil_win

   # Install python dependencies
   poetry install

   # Install pyradiomics (Error Poetry)
   pip install pyradiomics

   # Install Pytorch Geomatric Dependencies (Issues could arise when installing)
   pip install --no-binary :all: torch-cluster torch-scatter pyg-lib

.. note::

   On Windows you may encounter errors or duplicated runs due to DataLoader num_workers. Comment out all num_workers arguments (or set them to 0) to mitigate this.

Basic Usage
-----------

1. **Extract patches from WSI**:

.. code-block:: bash

   poetry run patch_extraction --output_path ./results --wsi_path ./data/C3L-00001-21.svs --patch_size 1024 --patch_overlap 6.25 --target_mag 20.0

2. **Run cell segmentation**:

.. code-block:: bash

   poetry run cell_segmentation --model cellvit --gpu 0 --wsi_path ./data/C3L-00001-21.svs --patched_slide_path ./results/C3L-00001-21

3. **Graph creation**:
   
.. code-block:: bash

   poetry run graph_creation  --method knn --gpu 0 --patched_slide_path ./results/C3L-00001-21  --segmentation_model cellvit

4. **Extract features**:

   4.1. **Morphological features**:

   .. code-block:: bash

      poetry run feature_extraction  --extractor pyradiomics_gray  --patched_slide_path ./results/C3L-00001-21  --segmentation_model cellvit

   4.2. **Topological features**:

   .. code-block:: bash

      poetry run feature_extraction  --extractor connectivity --patched_slide_path ./results/C3L-00001-21  --segmentation_model cellvit --graph_method knn

   4.3. **Embedding features**:

   .. code-block:: bash

      poetry run feature_extraction  --extractor resnet50 --patched_slide_path ./results/C3L-00001-21 

Documentation Contents
======================

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   quickstart
   cli
   pipeline/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index

.. toctree::
   :maxdepth: 1
   :caption: Development

   contributing
   changelog

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

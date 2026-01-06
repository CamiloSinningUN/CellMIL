==============
Pipeline
==============

This section provides detailed documentation for all tools in the package. Each tool is designed to handle a specific part of the digital pathology analysis pipeline.

.. toctree::
   :maxdepth: 2
   :caption: Available Tools

   patch_extraction
   cell_segmentation
   graph_creation
   feature_extraction
   dataset_creation
   mil_training/index

Overview
========

The tools follow a modular design that enables flexible workflow composition. Each tool can be used independently or as part of a complete analysis pipeline.

Pipeline Flow
============

The typical workflow follows this sequence:

1. :doc:`patch_extraction` → Extract patches from whole slide images
2. :doc:`cell_segmentation` → Identify and segment individual cells
3. :doc:`graph_creation` → Create spatial graphs from segmented cells
4. :doc:`feature_extraction` → Compute quantitative features from cells
5. :doc:`dataset_creation` → Process multiple slides into training datasets
6. :doc:`mil_training/index` → Train Multiple Instance Learning models

Quick Reference
==============

Basic CLI Commands
--------------

.. code-block:: bash

   # Extract patches from a slide
   patch_extraction --output_path ./results --wsi_path ./data/slide.svs --patch_size 1024 --patch_overlap 6.25 --target_mag 20.0

   # Segment cells in patches
   cell_segmentation --model cellvit --gpu 0 --wsi_path ./data/slide.svs --patched_slide_path ./results/slide

   # Create spatial graphs from segmented cells
   graph_creation --methods knn --patched_slide_path ./results/slide --segmentation_model cellvit

   # Extract features from segmented cells
   feature_extraction --extractor pyradiomics --wsi_path ./data/slide.svs --patched_slide_path ./results/slide --segmentation_model cellvit

   # Create a dataset from multiple slides
   create_dataset --excel_path ./data/metadata.xlsx --output_path ./datasets --gpu 0 --segmentation_models cellvit hovernet --extractors morphometrics pyradiomics

Common Parameters
-----------------

**File Paths**
   - ``--wsi_path``: Path to whole slide image file
   - ``--patched_slide_path``: Path to directory with processed slide data
   - ``--output_path``: Directory for saving results
   - ``--excel_path``: Excel file with slide metadata and labels

**Model Selection**
   - ``--model``: Choice of segmentation or MIL model
   - ``--extractor``: Feature extraction method
   - ``--graph_method``: Graph creation methods
   - ``--segmentation_model``: Segmentation model used

**Computing Resources**
   - ``--gpu``: GPU device ID (or -1 for CPU)

Installation and Setup
=====================

Ensure you have the package installed with all dependencies. For detailed installation instructions, see :doc:`../installation`.

Tool Categories
==============

Data Processing Tools
--------------------

These tools handle the initial processing of whole slide images:

- :doc:`patch_extraction` - Divide large slides into manageable patches
- :doc:`cell_segmentation` - Identify individual cell instances
- :doc:`graph_creation` - Create spatial graphs from segmented cells
- :doc:`feature_extraction` - Compute quantitative cell features

Machine Learning Tools
---------------------

These tools handle model training and inference:

- :doc:`dataset_creation` - Prepare datasets for machine learning
- :doc:`mil_training/index` - Train Multiple Instance Learning models  

Performance Considerations
=========================

GPU Usage
---------

Most computationally intensive tools support GPU acceleration:

- Specify GPU device with ``--gpu N`` (where N is device ID)
- CPU is set automatically if it is not available

Memory Management
----------------

For large datasets:

- Tools automatically manage memory usage
- Process slides individually to prevent memory overflow
- Temporary files are cleaned up automatically
- Progress is saved incrementally

Parallel Processing
------------------

Where applicable, tools use parallel processing:

- Multi-core CPU utilization
- Batch processing for GPU operations
- Concurrent file I/O operations

Getting Help
===========

Command-Line Help
-----------------

Each tool provides built-in help:

.. code-block:: bash

   # Get help for any tool
   patch_extraction --help
   cell_segmentation --help
   graph_creation --help
   feature_extraction --help

Documentation Resources
----------------------

- **This documentation**: Comprehensive guides for each tool
- **API documentation**: Detailed technical reference
- **Quick start guide**: Step-by-step tutorial
- **GitHub repository**: Source code and issue tracking

See Also
========

- :doc:`../quickstart` - Step-by-step tutorial
- :doc:`../installation` - Installation instructions
- :doc:`../api/index` - API documentation

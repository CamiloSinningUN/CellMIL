================
Dataset Creation
================

The dataset creation tool processes multiple whole slide images according to metadata specifications, creating standardized datasets for Multiple Instance Learning (MIL) model training and evaluation.

Overview
========

Dataset creation automates the complete pipeline (patch extraction, cell segmentation, and feature extraction) across multiple slides, organizing results in a standardized format suitable for machine learning workflows. This tool is essential for creating training datasets from clinical or research slide collections.

CLI Usage
=========

Basic Command
-------------

.. code-block:: bash

   create_dataset [OPTIONS]

Required Arguments
------------------

.. option:: --excel_path PATH

   Path to Excel file containing slide metadata. Must include columns for slide paths, labels, and other relevant information.

.. option:: --output_path PATH

   Directory where the processed dataset will be saved. Creates organized subdirectories for each slide and processing step.

.. option:: --segmentation_models MODELS

   List of segmentation models to apply. Space-separated list of model names.

.. option:: --graph_methods METHODS

   List of graph creation methods to use. Space-separated list of method names.

.. option:: --extractors EXTRACTORS

   List of feature extractors to use. Space-separated list of extractor names.

Optional Arguments
------------------

.. option:: --gpu INTEGER

   GPU device ID to use for processing-intensive steps.
   
   *Default: 0*

Complete Example
----------------

.. code-block:: bash

   create_dataset \
       --excel_path ./data/metadata.xlsx \
       --output_path ./datasets/training_set \
       --gpu 0 \
       --segmentation_models cellvit hovernet cellpose_sam \
       --extractors morphometrics pyradiomics_hed \
       --graph_methods knn radius

This command will:

1. Read slide information from ``metadata.xlsx``
2. Process each slide through the complete pipeline
3. Apply all three segmentation models
4. Create spatial graphs from segmented cells
5. Extract both types of features
6. Save organized results to ``./datasets/training_set/``

Python API Usage
================

You can also create datasets programmatically:

.. code-block:: python

   from cellmil.dataset import DatasetCreator
   from cellmil.interfaces import DatasetCreatorConfig
   from pathlib import Path

   # Create configuration
   config = DatasetCreatorConfig(
       excel_path=Path("./data/metadata.xlsx"),
       output_path=Path("./datasets/training_set"),
       gpu=0,
       segmentation_models=["cellvit", "hovernet", "cellpose_sam"],
       extractors=["morphometrics", "pyradiomics_hed"],
       graph_methods=["knn", "radius"]
   )

   # Initialize dataset creator
   creator = DatasetCreator(config)
   
   # Create dataset
   creator.create()

Metadata Excel Format
====================

Required Columns
---------------

The Excel file must contain these essential columns:

.. csv-table:: metadata.xlsx
   :header: "FULL_PATH", "LABEL", "ID"
   :widths: 30, 10, 15

   "./data/patient001.svs", 1, "P001"
   "./data/patient002.svs", 0, "P002"
   "./data/patient003.svs", 1, "P003"
   "./data/patient004.svs", 0, "P004"

**FULL_PATH**
   Full or relative path to the WSI file

**LABEL**
   Target label for classification (integer or string)

**ID (Optional)**
   Unique identifier for the slide (for tracking and reference)

Optional Columns
---------------

Additional metadata can include:

- **MAGNIFICATION**: Target magnification level for WSI.

Processing Pipeline
==================

The dataset creation follows this automated pipeline:

1. **Metadata Validation**
   - Check Excel file format and required columns
   - Validate slide file paths and accessibility
   - Verify label consistency and format

2. **Patch Extraction**
   - Extract patches from each slide using consistent parameters
   - Apply quality filtering and tissue detection
   - Save patch coordinates and metadata

3. **Cell Segmentation**
   - Apply each specified segmentation model
   - Generate cell masks and detection metadata
   - Perform quality control on segmentation results

4. **Graph Creation**
   - Create spatial graphs from segmented cells

5. **Feature Extraction**
   - Extract features using each specified extractor
   - Validate feature completeness and quality

6. **Dataset Organization**
   - Organize results in standardized directory structure
   - Generate dataset-level metadata and statistics


Error Recovery
===============

The tool supports resumption of interrupted processing:

- Previously processed slides are automatically skipped
- Processing state is saved incrementally
- Failed slides are logged for manual review

.. code-block:: bash

   # Resume interrupted processing
   create_dataset \
       --excel_path ./data/metadata.xlsx \
       --output_path ./datasets/training_set \
       --gpu 0 \
       --segmentation_models cellvit hovernet \
       --extractors pyradiomics_hed \
       --graph_methods knn radius

See Also
========

- :doc:`patch_extraction` - Individual pipeline step documentation
- :doc:`cell_segmentation` - Individual pipeline step documentation
- :doc:`graph_creation` - Individual pipeline step documentation
- :doc:`feature_extraction` - Individual pipeline step documentation
- :doc:`mil_training` - Next step in the pipeline
- :doc:`../api/_autosummary/cellmil.dataset` - API documentation

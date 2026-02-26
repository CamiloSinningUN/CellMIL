===================================
Data Preparation
===================================

.. contents:: Table of Contents
   :depth: 2
   :local:

This section describes how to prepare data for MIL training, including feature extractors, transforms, filtering, and dataset creation.

Key Parameters
==============

Dataset Parameters
------------------

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Parameter
     - Description
   * - ``root``
     - Cache directory where processed datasets are stored 
   * - ``folder``
     - Path to the dataset directory created by the dataset creation tool
   * - ``label``
     - Column name in metadata containing target labels (str for classification, tuple for survival)
   * - ``data``
     - pandas DataFrame with slide metadata and labels
   * - ``extractor``
     - Feature extractor(s)
   * - ``segmentation_model``
     - Segmentation model used: ``ModelType.cellvit`` or ``ModelType.hovernet``
   * - ``graph_creator``
     - Graph construction method (required for graph models/features)
   * - ``cell_type``
     - Boolean - include cell type information (required for Head4Type model)

Training Parameters
-------------------

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Parameter
     - Description
   * - ``name``
     - Experiment name for tracking and logging
   * - ``output_dir``
     - Directory where results and checkpoints will be saved
   * - ``wandb_project``
     - Weights & Biases project name (required for experiment tracking)
   * - ``transforms``
     - TransformPipeline for feature preprocessing
   * - ``label_transforms``
     - Label transforms (e.g., TimeDiscretizerTransform for survival)
   * - ``balance_cell_counts``
     - Boolean - apply cell stratification to balance instance counts


Normalization and Filtering
============================

Transforms are applied using the ``TransformPipeline`` class.

Robust Scaling
--------------

Apply robust scaling normalization to features:

.. code-block:: python

   from cellmil.datamodels.transforms import (
       TransformPipeline,
       RobustScalerTransform,
   )
   
   transforms = TransformPipeline([
       RobustScalerTransform(apply_log_transform=True),
   ])
   
   # Pass to k-fold training
   model_storage = k_fold.evaluate(
       # ... other parameters
       transforms=transforms,
   )

This applies robust scaling which:

- Removes median and scales by IQR

- Robust to outliers

- Optional log transformation

Correlation Filtering
---------------------

Remove highly correlated features to reduce redundancy:

.. code-block:: python

   from cellmil.datamodels.transforms import (
       TransformPipeline,
       CorrelationFilterTransform,
   )
   
   transforms = TransformPipeline([
       CorrelationFilterTransform(
           correlation_threshold=0.95,
           plot_correlation_matrix=False,
       ),
   ])

**Recommended when:**

- Combining multiple feature extractors
  
- Using large feature sets (e.g., "ALL")
  
- Reducing dimensionality

Combined Transforms
-------------------

Chain multiple transforms together:

.. code-block:: python

   transforms = TransformPipeline([
       CorrelationFilterTransform(correlation_threshold=0.95),
       RobustScalerTransform(apply_log_transform=True),
   ])

**Note:** Order matters! Correlation filtering should typically come before normalization.

Configuration Examples
=======================

Basic Classification
--------------------

.. code-block:: python

   from pathlib import Path
   import pandas as pd
   from cellmil.interfaces.CellSegmenterConfig import ModelType
   from cellmil.datamodels.datasets import MILDataset
   from cellmil.utils.train import get_extractors_from_name
   from cellmil.utils.train.evals import KFoldCrossValidation
   
   # Constants
   ROOT = Path("./MIL_dataset")
   DATASET_FOLDER = Path("./dataset")
   
   # Load data
   df = pd.read_excel("metadata.xlsx")
   
   # Create dataset
   dataset = MILDataset(
       root=ROOT,
       label="subtype",
       folder=DATASET_FOLDER,
       data=df,
       extractor=ExtractorType.morphometrics,
       segmentation_model=ModelType.cellvit,
   )
   
   # Train
   k_fold = KFoldCrossValidation(k=5)
   model_storage = k_fold.evaluate(
       name="morpho_subtype",
       lit_model_creator=lit_model_creator,
       dataset=dataset,
       output_dir=Path("./results"),
       wandb_project="my_project",
   )

Graph-Based Model
-----------------

.. code-block:: python

   from cellmil.datamodels.datasets import GNNMILDataset
   from cellmil.datamodels.transforms import TransformPipeline, RobustScalerTransform
   
   # Use GNNMILDataset for graph models
   dataset = GNNMILDataset(
       root=ROOT,
       label="RESPONSE",
       folder=DATASET_FOLDER,
       data=df,
       extractor=ExtractorType.morphometrics,
       segmentation_model=ModelType.cellvit,
       graph_creator=GraphCreatorType.delaunay_radius,  # Required!
   )
   
   transforms = TransformPipeline([
       RobustScalerTransform(apply_log_transform=True),
   ])
   
   model_storage = k_fold.evaluate(
       name="gat_morpho_response",
       lit_model_creator=lit_model_creator,
       dataset=dataset,
       output_dir=Path("./results"),
       wandb_project="my_project",
       transforms=transforms,
   )

Survival Analysis
-----------------

.. code-block:: python

   from cellmil.datamodels.transforms import TimeDiscretizerTransform
   
   # Label is tuple of (duration, event)
   dataset = MILDataset(
       root=ROOT,
       label=("duration", "event"),  # Tuple for survival
       folder=DATASET_FOLDER,
       data=df,
       extractor=ExtractorType.pyradiomics_hed,
       segmentation_model=ModelType.cellvit,
   )
   
   # Create transforms
   transforms = TransformPipeline([
       RobustScalerTransform(apply_log_transform=True),
   ])
   
   # Label transform for time discretization
   label_transforms = TimeDiscretizerTransform(n_bins=4)
   
   # Train with survival model
   model_storage = k_fold.evaluate(
       name="survival_pyrad",
       lit_model_creator=lit_model_creator,
       dataset=dataset,
       output_dir=Path("./results"),
       wandb_project="my_project",
       transforms=transforms,
       label_transforms=label_transforms,  # Required for survival
   )

Foundation Model Embeddings
---------------------------

.. code-block:: python
   
   dataset = MILDataset(
       root=ROOT,
       label="subtype",
       folder=DATASET_FOLDER,
       data=df,
       extractor=ExtractorType.gigapath,
       segmentation_model=ModelType.cellvit,
   )
   
   model_storage = k_fold.evaluate(
       name="gigapath_subtype",
       lit_model_creator=lit_model_creator,
       dataset=dataset,
       output_dir=Path("./results"),
       wandb_project="my_project",
       transforms=transforms,
   )

Head4Type
--------------------------

.. code-block:: python
   
   # Enable cell_type parameter
   dataset = MILDataset(
       root=ROOT,
       label="subtype",
       folder=DATASET_FOLDER,
       data=df,
       extractor=ExtractorType.morphometrics,
       segmentation_model=ModelType.cellvit,
       cell_type=True,  # Required for Head4Type
   )
   
   transforms = TransformPipeline([
       RobustScalerTransform(apply_log_transform=True),
   ])
   
   # Use cell stratification for balanced sampling
   model_storage = k_fold.evaluate(
       name="head4type_morpho",
       lit_model_creator=lit_model_creator,
       dataset=dataset,
       output_dir=Path("./results"),
       wandb_project="my_project",
       transforms=transforms,
       balance_cell_counts=True,  # Cell stratification
   )

Data Filtering
==============

Cell Type Filtering
-------------------

Include only specific cell types in your analysis:

.. code-block:: python

   dataset = MILDataset(
       root=ROOT,
       label="DCR",
       folder=DATASET_FOLDER,
       data=df,
       extractor=extractors,
       segmentation_model=ModelType.cellvit,
       cell_types_to_keep=["Neoplastic", "Inflammatory"],  # Only these types
   )

**Available cell types** (CellViT/HoVerNet):

- ``Neoplastic``
- ``Inflammatory``
- ``Connective``
- ``Dead``
- ``Epithelial``


ROI Filtering
-------------

Filter cells to only those within specific regions of interest:

.. code-block:: python

   dataset = MILDataset(
       root=ROOT,
       label="DCR",
       folder=DATASET_FOLDER,
       data=df,  # Must contain 'ID', 'SLIDE_ID', 'CENTER' columns
       extractor=extractors,
       segmentation_model=ModelType.cellvit,
       roi_folder=Path("./data/rois"),  # Folder with ROI CSV files
   )

**ROI file format:**

Each ROI file should be a CSV with cell coordinates:

.. csv-table::
   :header: "x", "y"
   :widths: 50, 50

   1024, 2048
   1030, 2055
   ...

Next Steps
==========

- :doc:`training` - Train models with k-fold cross-validation
- :doc:`models` - Explore available model architectures

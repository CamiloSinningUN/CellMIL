===================================
MIL Training
===================================

This section guides you through building custom Multiple Instance Learning (MIL) training scripts using the CellMIL API. You'll learn how to configure datasets, select models, apply preprocessing transforms, and train deep learning models on preprocessed whole slide image data.

.. toctree::
   :maxdepth: 2
   :caption: MIL Training Guide

   overview
   models
   data_preparation
   training

Overview
========

Multiple Instance Learning (MIL) enables training models on whole slide images (WSI) using only slide-level labels. Each slide is treated as a "bag" of cell instances, and the model learns to aggregate cell-level features to predict slide-level outcomes such as:

- Binary classification (e.g., responder vs non-responder, histological subtype)
- Survival prediction (time-to-event outcomes)

Quick Start
===========

Here's a minimal example to get started:

.. code-block:: python

   from pathlib import Path
   import pandas as pd
   from cellmil.datamodels.datasets import MILDataset
   from cellmil.utils.train.evals import KFoldCrossValidation
   
   # Load metadata
   df = pd.read_excel("./data/metadata.xlsx")
   df = df[df["DCR"].isin([0, 1])]
   
   # Create dataset
   dataset = MILDataset(
       root=Path("./MIL_dataset"),
       label="DCR",
       folder=Path("./dataset"),
       data=df,
       extractor="morphometrics",
       segmentation_model="cellvit",
   )
   
   # Define model creator
   def lit_model_creator(input_dim: int):
       from cellmil.models.mil.attentiondeepmil import AttentionDeepMIL, LitAttentionDeepMIL
       from torch.optim import AdamW
       
       model = AttentionDeepMIL(embed_dim=input_dim, n_classes=2)
       lit_model = LitAttentionDeepMIL(
           model=model,
           optimizer=AdamW(model.parameters(), lr=1e-4),
       )
       return lit_model
   
   # Initialize k-fold cross-validation
   k_fold = KFoldCrossValidation(k=5, random_state=42)
   
   # Run evaluation
   model_storage = k_fold.evaluate(
       name="my_experiment",
       lit_model_creator=lit_model_creator,
       dataset=dataset,
       output_dir=Path("./results"),
       wandb_project="my_project",
   )

Next Steps
==========

- :doc:`overview` - Understand the MIL framework and aggregation strategies
- :doc:`models` - Explore available model architectures
- :doc:`configuration` - Learn about configuration parameters
- :doc:`data_preparation` - Prepare datasets and apply transforms
- :doc:`training` - Train models with k-fold cross-validation

See Also
========

- :doc:`../dataset_creation` - Creating the dataset structure
- :doc:`../feature_extraction` - Extracting features from cells

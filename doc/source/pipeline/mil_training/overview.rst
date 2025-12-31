===================================
Overview
===================================

.. contents:: Table of Contents
   :depth: 2
   :local:

What is Multiple Instance Learning?
====================================

Multiple Instance Learning (MIL) is a machine learning paradigm where:

- Each slide is a "bag" containing many instances (cells)
- You only have labels at the bag level (e.g., "cancer type" or "survival time")
- The model learns to aggregate item-level information to make bag-level predictions

This is perfect for digital pathology because pathologists diagnose slides.

Pipeline Overview
=================

The CellMIL training pipeline works with data that's already been processed through earlier steps:

1. **Cell Segmentation** 
2. **Feature Extraction**
3. **MIL Training** - This is what you're doing now!

Your job is to:

- Load the pre-extracted features
- Choose a model architecture
- Train the model to predict slide-level outcomes

Common Tasks
============

Binary Classification
---------------------

Predict one of two classes (e.g., cancer subtype A vs B).

**Label format:** Single column with 0/1 values

.. code-block:: python

   dataset = MILDataset(
       label="HISTOLOGY",  # Column with 0/1 labels
       # ...
   )

Survival Prediction
-------------------

Predict time-to-event outcomes (e.g., overall survival).

**Label format:** Tuple of (duration_column, event_column)

.. code-block:: python

   dataset = MILDataset(
       label=("OS_MONTHS", "OS_EVENT"),  # Duration and event status
       # ...
   )

Next Steps
==========

- :doc:`models` - Detailed model descriptions and when to use each
- :doc:`configuration` - Set up your experiment
- :doc:`data_preparation` - Load and prepare your dataset
- :doc:`training` - Train with k-fold cross-validation

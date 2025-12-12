=============
API Reference
=============

This section provides detailed documentation for all modules, classes, and functions.

The package is organized into several main modules:

Core Modules
============

.. autosummary::
   :toctree: _autosummary
   :recursive:

   cellmil.cli
   cellmil.config.python
   cellmil.data
   cellmil.datamodels
   cellmil.dataset
   cellmil.features
   cellmil.interfaces
   cellmil.models
   cellmil.segmentation
   cellmil.graph
   cellmil.utils
   cellmil.statistics
   cellmil.visualization

Package Structure
=================

The cellmil package is structured as follows:

.. code-block:: text

   cellmil/
   ├── cli/                    # Command-line interface tools
   │   ├── patch_extraction.py
   │   ├── cell_segmentation.py
   │   ├── feature_extraction.py
   │   └── ...
   ├── models/                # Deep learning models
   │   ├── enconders/         # Image encoders
   │   ├── segmentation/      # Cell segmentation models
   │   └── mil/               # Multiple instance learning models
   ├── data/                  # Data processing utilities
   ├── datamodels/            # Data models
   ├── dataset/               # Dataset utilities
   ├── features/              # Feature extraction modules
   ├── mil/                   # Multiple instance learning modules
   ├── interfaces/            # Configuration interfaces
   ├── segmentation/          # Cell segmentation utilities
   ├── graph/                 # Graph utilities
   ├── visualization/         # Visualization tools
   ├── tests/                 # Unit tests [Almost useless]
   └── utils/                 # Utility functions

Quick Access
============

Common Classes and Functions
----------------------------

**Data Processing:**

- :class:`cellmil.data.PatchExtractor` - Extract patches from WSI
- :class:`cellmil.interfaces.PatchExtractorConfig` - Patch extraction configuration

**Segmentation Models:**

- :class:`cellmil.models.segmentation.CellViTSAM` - CellViT segmentation
- :class:`cellmil.models.segmentation.HoverNet` - HoverNet segmentation
- :class:`cellmil.models.segmentation.CellposeSAM` - CellposeSAM segmentation

**MIL Models:**

- :class:`cellmil.models.mil.CLAM_SB` - CLAM single branch
- :class:`cellmil.models.mil.CLAM_MB` - CLAM multi branch
- :class:`cellmil.models.mil.MIL_fc` - Standard MIL with Max pooling (Binary Classification)
- :class:`cellmil.models.mil.MIL_fc_mc` - Standard MIL with Max pooling (Multi-class Classification)
- :class:`cellmil.models.mil.TransMIL` - Transformer MIL with PPEG
- :class:`cellmil.models.mil.HistoBistro` - HistoBistro MIL

**Feature Extraction:**

- :class:`cellmil.features.Extractor` - Base feature extractor interface

**Visualization:**

- :class:`cellmil.visualization.FeatureVisualizer` - Feature visualization tools

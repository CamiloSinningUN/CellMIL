=========
Changelog
=========

All notable changes to this project will be documented in this file.

[0.1.0] - 2025-01-29
====================

Added
-----
- Patch extraction from whole slide images
- Multiple cell segmentation models:
  - CellViT: Vision transformer-based nuclear instance segmentation
  - HoverNet: Nuclear instance segmentation
  - Cellpose: Deep learning cell instance segmentation
- Feature extraction capabilities:
  - PyRadiomics_Gray: Comprehensive radiomics features
  - PyRadiomics_HED: Comprehensive radiomics features
  - Morphometrics: Morphological features
- Multiple Instance Learning models:
  - CLAM: Clustering-constrained attention MIL
  - TransMIL: Transformer-based MIL with PPEG
  - Standard MIL: Max pooling MIL
  - HistoBistro: Transformer-based MIL
- Command-line interface for all operations
- Feature visualization tools
- Graph creation tools
- Dataset creation and management
- Model training and evaluation pipelines

Features
--------
- Support for multiple WSI formats (SVS, TIFF, NDPI, etc.)
- GPU acceleration for deep learning models
- Configurable patch extraction parameters
- Comprehensive logging and error handling
- Type-safe configuration with Pydantic

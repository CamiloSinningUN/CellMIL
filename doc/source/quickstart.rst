===========
Quick Start
===========

This guide will walk you through the basic workflow of using this package.

Overview
========

The pipeline consists of four main steps:

1. **Patch Extraction**: Extract patches from whole slide images (WSI)
2. **Cell Segmentation**: Segment individual cells within patches
3. **Graph Creation**: Create spatial graphs from segmented cells
4. **Feature Extraction**: Extract morphological and radiomics features from cells
   
Step-by-Step Workflow
====================

Prerequisites
-------------

Before starting, ensure you have:

- A whole slide image file compatible with `OpenSlide <https://openslide.org/>`_ (SVS, TIFF, NDPI, etc.)
- Package properly installed (see :doc:`installation`)
- Sufficient disk space for intermediate results

Step 1: Patch Extraction
------------------------

Extract patches from your WSI with specified parameters:

.. code-block:: bash

   patch_extraction \
       --output_path ./results \
       --wsi_path ./data/C3L-00001-21.svs \
       --patch_size 1024 \
       --patch_overlap 6.25 \
       --target_mag 20.0

**Parameters explained:**

- ``--output_path``: Directory where results will be saved
- ``--wsi_path``: Path to the whole slide image file
- ``--patch_size``: Size of extracted patches in pixels (default: 1024)
- ``--patch_overlap``: Overlap percentage between patches (default: 6.25)
- ``--target_mag``: Target magnification for extraction (default: 20.0)

Step 2: Cell Segmentation
-------------------------

Segment cells within the extracted patches using your preferred model:

.. code-block:: bash

   cell_segmentation \
       --model cellvit \
       --gpu 0 \
       --wsi_path ./data/C3L-00001-21.svs \
       --patched_slide_path ./results/C3L-00001-21

**Available segmentation models:**

- ``cellvit``: Vision transformer for nuclear instance segmentation and classification
- ``hovernet``: Convolutional model for nuclear instance segmentation and classification
- ``cellpose_sam``: Cellpose with SAM as backbone for cell segmentation

**Parameters explained:**

- ``--model``: Segmentation model to use
- ``--gpu``: GPU device ID
- ``--patched_slide_path``: Path to the directory containing extracted patches

Step 3: Graph Creation
----------------------

Create spatial graphs from the segmented cells:

.. code-block:: bash

   graph_creation \
       --method knn \
       --gpu 0 \
       --patched_slide_path ./results/C3L-00001-21 \
       --segmentation_model cellvit

**Available graph creation methods:**

- ``knn``: K-nearest neighbors graph
- ``delaunay_radius``: Delaunay triangulation graph with a radius limit
- ``radius``: Radius-based graph
- ``dilate``: Dilation of segmented nuclei based graph
- ``similarity``: Similarity-based graph using morphological features (Morphometrics and Pyradiomics)

**Parameters explained:**

- ``--method``: Graph creation method to use
- ``--gpu``: GPU device ID

Step 4: Feature Extraction
--------------------------

Extract morphological features from the segmented cells:

.. code-block:: bash
   feature_extraction \
      --extractor pyradiomics_gray \
      --wsi_path ./data/C3L-00001-21.svs \
      --patched_slide_path ./results/C3L-00001-21 \
      --segmentation_model cellvit

Extract Topological features from the segmented cells and created graph:

.. code-block:: bash
   feature_extraction \
       --extractor connectivity \
       --patched_slide_path ./results/C3L-00001-21 \
       --segmentation_model cellvit \
       --graph_method knn

.. **Available feature extractors:**

- ``pyradiomics_gray``: Comprehensive radiomics features with gray-scale preprocessing.
- ``pyradiomics_hed``: Comprehensive radiomics features with HED preprocessing taking the Haematoxylin channel.
- ``pyradiomics_hue``: Comprehensive radiomics features with Hue channel preprocessing.
- ``morphometrics``: Morphological features `Functional Morphometric Analysis in Cellular Behaviors: Shape and Size Matter <http://dx.doi.org/10.1002/adhm.201300053>`_
- ``connectivity``: Topological features based on the cell connectivity graph.
.. - ``structure``: Structural features based on the graph arrangement.
- ``geometric``: Geometric features based on the graph geometry.
- ``resnet50``: Embedding features extracted from the ResNet50 model.
- ``gigapath``: Embedding features extracted from the Prov-GigaPath model.
- ``uni``: Embedding features extracted from the UNI model.

Step 5: Visualization (Optional)
--------------------------------

Visualize the extracted features:

.. code-block:: bash

   vis_features \
       --patched_slide_path ./results/C3L-00001-21

This will generate visualizations of the feature distributions and correlation patterns.


Expected Output Structure
========================

After running the complete pipeline, your output directory will have this structure:

.. code-block:: text

   results/
   └── C3L-00001-21/
       ├── patches/                    # Extracted patches
       │   ├── patch_0_0.png
       │   ├── patch_0_1.png
       │   └── ...
       ├── cell_detection/             # Segmentation results
       │   └── cellvit/
       │       ├── cells.json
       │       ├── cells.geojson
       │       ├── cell_detection.json
       │       └── cell_detection.geojson
       ├── graphs/                    # Graphs
       │   └── cellvit/
       │       └── graph.pt
       ├── feature_extraction/                   # Extracted features
       │   └── pyradiomics_gray/
       │       └── cellvit/
       │           └── features.pt # [N, D], N: number of cells, D: number of features
       └── predictions/                # MIL predictions
           └── ... # TODO: Add MIL prediction output structure

.. Python API Usage
.. ================

.. You can also use this package programmatically in Python:

.. .. code-block:: python

..     from cellmil.data import PatchExtractor
..     from cellmil.interfaces import PatchExtractorConfig
..     from cellmil.segmentation import CellSegmenter
..     from cellmil.interfaces import CellSegmenterConfig
..     from cellmil.interfaces import FeatureExtractorConfig
..     from cellmil.features import FeatureExtractor

..     # Run patch extraction
..     config = PatchExtractorConfig(
..         output_path="./results",
..         wsi_path="./data/slide.svs",
..         patch_size=1024,
..         patch_overlap=6.25,
..         target_mag=20.0
..     )

..     extractor = PatchExtractor(config)
..     extractor.get_patches()

..     # Run segmentation
..     config = CellSegmenterConfig(
..         model="cellvit",
..         gpu=0,
..         wsi_path="./data/slide.svs",
..         patched_slide_path="./results/slide_name"
..     )

..     segmentation_model = CellSegmenter(config)
..     segmentation_model.process()

..     # Extract features
..     config = FeatureExtractorConfig(
..         extractor="pyradiomics_gray",
..         wsi_path="./data/slide.svs",
..         patched_slide_path="./results/slide_name",
..         segmentation_model="cellvit"
..     )

..     feature_extractor = FeatureExtractor(config)
..     features = feature_extractor.get_features()

..     # TODO: Run MIL prediction
..     # TODO: Run MIL Training
..     # TODO: Visualize features
..     # TODO: Dataset creation

Next Steps
==========

- Learn more about the :doc:`cli` for advanced options
- Explore the :doc:`api/index` for programmatic usage

Common Issues
=============

**Out of Memory Errors**
   - Reduce batch size or patch size

**Slow Processing**
   - Enable GPU acceleration
   - Use smaller patch sizes

**File Format Issues**
   - Ensure WSI format is supported by `OpenSlide <https://openslide.org/>`_ (SVS, TIFF, NDPI, etc.)
   - Check file permissions and paths
   - Verify file integrity

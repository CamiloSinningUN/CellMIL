===============
Explainability
===============

This package provides two complementary explainability methods for attention-based Multiple Instance Learning (MIL) models: **Attention Heatmaps** for local explanations and **SHAP Analysis** for global feature importance.

Overview
========

Attention Heatmaps
------------------

Attention heatmaps provide **local, instance-level explanations** by visualizing which cells (instances) the model focuses on when making a prediction for a specific slide. This shows *where* the model is looking.

**Supported Models:**

- CLAM (Clustering-constrained Attention MIL)
- AttentionDeepMIL (ABMIL)
- HEAD4TYPE (Cell type-aware attention)
- GraphMIL models with attention pooling

SHAP Analysis  
-------------

SHAP (SHapley Additive exPlanations) values provide **global feature importance explanations** across the entire dataset by identifying which cell features (morphological, textural, topological) drive the attention mechanism. This reveals *what* features the model considers important.

**Key Insight:** SHAP analyzes the attention module itself, explaining which features lead to high or low attention scores rather than the final prediction.

Attention Heatmaps
==================

Attention heatmaps visualize attention weights spatially on whole slide images, highlighting regions the model deems important.

Configuration
-------------

Create an ``AttentionExplainerConfig`` to customize the visualization:

.. code-block:: python

   from cellmil.interfaces.AttentionExplainerConfig import (
       AttentionExplainerConfig,
       VisualizationMode,
       Normalization
   )
   from pathlib import Path

   config = AttentionExplainerConfig(
       output_path=Path("./explanations/attention"),
       visualization_mode=VisualizationMode.graph,
       normalization=Normalization.min_max,
       normalize_attention=True,
       visualize_cell_types=False,
   )

**Configuration Parameters:**

- **output_path** (``Path``, required): Directory where visualizations will be saved
- **visualization_mode** (``VisualizationMode``, default: ``geojson``): Output format
  
  - ``geojson``: Compatible with pathology viewers (QuPath, etc.)
  - ``graph``: Interactive graph visualization with attention edges
  - ``all``: Generate both formats

- **normalization** (``Normalization``, default: ``min_max``): Attention weight normalization
  
  - ``min_max``: Scale to [0, 1]
  - ``z_score``: Standardize (mean=0, std=1)
  - ``robust``: Median and IQR scaling
  - ``softmax``: Normalize to sum to 1
  - ``sigmoid``: Sigmoid transformation
  - ``none``: No normalization

- **normalize_attention** (``bool``, default: ``True``): Whether to apply normalization
- **color_scheme** (``tuple[list[int], list[int]]``): RGB color gradient (start, end)
- **visualize_cell_types** (``bool``, default: ``False``): Create cell type visualizations
- **attention_aggregation** (``Aggregation``): For GraphMIL models only
  
  - ``pooling_only``: Only pooling layer attention
  - ``gnn_only``: Only GNN layer attention
  - ``gnn_layer``: Specific GNN layer (set ``gnn_layer_index``)
  - ``random_walk``: Aggregate via random walk

- **class_index** (``Optional[int]``): For multi-class models, specify class to explain
- **attention_head** (``Optional[int]``): For multi-head attention, specify head

Usage
-----

**Basic Example:**

.. code-block:: python

   from cellmil.explainability.attention import AttentionExplainer
   from cellmil.datamodels.model import ModelStorage
   from pathlib import Path

   # Load trained model
   model_storage = ModelStorage.from_directory(
       "./results/ADENOvsSQUA+PYRAD+HEAD4TYPE+REG+STRA"
   )

   # Initialize explainer
   explainer = AttentionExplainer(config)

   # Generate explanation for a specific slide
   results = explainer.generate_explanation(
       model_storage=model_storage,
       slide_path=Path("./dataset/DIG_PAT_1696325298"),
   )

**Using a Specific Fold:**

.. code-block:: python

   # Use fold 2 instead of the final model
   results = explainer.generate_explanation(
       model_storage=model_storage,
       slide_path=Path("./dataset/DIG_PAT_1696325298"),
       fold_idx=2,  # Use specific fold
   )

**GraphMIL Models:**

.. code-block:: python

   # For GraphMIL models, aggregate GNN attention
   config = AttentionExplainerConfig(
       output_path=Path("./explanations/graphmil"),
       visualization_mode=VisualizationMode.graph,
       attention_aggregation=Aggregation.gnn_only,
   )

Model Storage
-------------

The ``ModelStorage`` class manages trained models from k-fold cross-validation experiments. It automatically loads:

- Model checkpoint (weights)
- Feature transforms (normalization, scaling)
- Dataset configuration (extractors, segmentation model, graph creator)
- Model configuration (architecture, hyperparameters)

**Loading from Directory:**

.. code-block:: python

   from cellmil.datamodels.model import ModelStorage

   # Load from k-fold results directory
   model_storage = ModelStorage.from_directory(
       "./results/OS24+ALL+CLAM+REG+STRA"
   )

   # Check available folds
   available_folds = model_storage.list_folds()  # [0, 1, 2, 3, 4]

   # Check if final model exists
   has_final = model_storage.has_final_model()  # True/False

Model directories are typically created by training scripts and follow this structure:

.. code-block:: text

   results/
   └── TASK+FEATURES+MODEL+REG+STRA/
       ├── experiment_metadata.json
       ├── fold_0/
       │   ├── best_model.ckpt
       │   └── transforms/
       ├── fold_1/
       │   ├── best_model.ckpt
       │   └── transforms/
       └── final_model/
           ├── best_model.ckpt
           └── transforms/

Output
------

The explainer generates several output files:

**Visualizations:**

- **GeoJSON** (``*.geojson``): Cell polygons colored by attention weight
  
  - Import into QuPath or other pathology viewers
  - Each cell has ``attention_weight`` property

- **Graph** (``*_graph.html``): Interactive Plotly visualization
  
  - Nodes colored by attention weight
  - Edge opacity reflects spatial relationships
  - Hover to see cell details

**Metadata:**

- ``attention_weights.json``: Raw attention weights for each cell
- ``attention_metadata.json``: Statistics (mean, std, entropy, sparsity)

SHAP Analysis
=============

SHAP analysis identifies which cell features drive the attention mechanism globally across the entire dataset.

Methodology
-----------

1. **Cell-Level Dataset Creation**: Extract features from all cells across all slides in the dataset
2. **Attention Computation**: Compute attention weights for every cell using the trained model
3. **Stratified Sampling**: Sample cells from different attention quantile bins to ensure diverse representation
4. **SHAP Computation**: Use SHAP to explain which features contribute to high/low attention scores
5. **Visualization**: Generate feature importance plots and summary statistics

Configuration
-------------

.. code-block:: python

   from cellmil.interfaces.SHAPExplainerConfig import (
       SHAPExplainerConfig,
       SHAPExplainerType
   )
   from pathlib import Path

   config = SHAPExplainerConfig(
       output_path=Path("./explanations/shap"),
       
       # Sampling parameters
       num_bins=5,
       samples_per_bin=10000,
       
       # SHAP computation
       explainer_type=SHAPExplainerType.gradient,
       background_percentage=0.2,
       nsamples=500,  # Only for kernel explainer
       
       # Analysis options
       explain_per_head=True,
       explain_mean_head=True,
       top_features=20,
       
       # Output options
       save_raw_shap_values=True,
       create_summary_plots=True,
   )

**Configuration Parameters:**

**Sampling Configuration:**

- **num_bins** (``int``, default: 5): Number of attention quantile bins (e.g., 5 = quantiles)
- **samples_per_bin** (``int``, default: 10000): Cells to sample per bin
- **max_total_samples** (``Optional[int]``): Cap total samples (overrides ``samples_per_bin``)

**SHAP Computation:**

- **explainer_type** (``SHAPExplainerType``, default: ``gradient``):
  
  - ``gradient``: Fast, uses gradients (recommended for neural networks)
  - ``deep``: DeepExplainer for deep learning
  - ``kernel``: Model-agnostic but very slow

- **background_percentage** (``float``, default: 0.2): Fraction of sampled cells to use as background
- **nsamples** (``int``, default: 500): Coalitions per cell for kernel explainer only
- **explain_top_cells** (``Optional[int]``): Only explain top N cells (None = all sampled)
- **explain_per_head** (``bool``, default: ``True``): Separate SHAP for each attention head
- **explain_mean_head** (``bool``, default: ``True``): SHAP for mean attention across heads

**Analysis Options:**

- **top_features** (``int``, default: 20): Number of top features to highlight
- **random_seed** (``int``, default: 42): For reproducibility
- **batch_size** (``int``, default: 1024): Batch size for attention computation

**Output Options:**

- **save_raw_shap_values** (``bool``, default: ``True``): Save raw SHAP values (can be large)
- **create_summary_plots** (``bool``, default: ``True``): Generate visualizations

Usage
-----

**Basic Example:**

.. code-block:: python

   from cellmil.explainability.shap import SHAPExplainer
   from cellmil.datamodels.model import ModelStorage
   import pandas as pd
   from pathlib import Path

   # Load trained model
   model_storage = ModelStorage.from_directory(
       "./results/ADENOvsSQUA+PYRAD+HEAD4TYPE+REG+STRA"
   )

   # Load metadata (must have 'FULL_PATH' column with slide names)
   metadata = pd.read_excel("./data/dp_metadata_INT_40.xlsx")

   # Initialize explainer
   config = SHAPExplainerConfig(output_path=Path("./explanations/shap"))
   explainer = SHAPExplainer(config)

   # Generate explanation
   results = explainer.generate_explanation(
       model_storage=model_storage,
       dataset_folder=Path("./dataset"),
       data=metadata,
   )

**Using Specific Fold:**

.. code-block:: python

   # Explain fold 3
   results = explainer.generate_explanation(
       model_storage=model_storage,
       dataset_folder=Path("./dataset"),
       data=metadata,
       fold_idx=3,
   )

**Custom Sampling:**

.. code-block:: python

   # More aggressive sampling for larger dataset
   config = SHAPExplainerConfig(
       output_path=Path("./explanations/shap_large"),
       num_bins=10,  # Decile bins
       samples_per_bin=5000,
       max_total_samples=40000,  # Cap at 40k total
       top_features=30,
   )

**Kernel Explainer (Slower but Model-Agnostic):**

.. code-block:: python

   config = SHAPExplainerConfig(
       output_path=Path("./explanations/shap_kernel"),
       explainer_type=SHAPExplainerType.kernel,
       nsamples=1000,  # More samples for better approximation
       samples_per_bin=5000,  # Fewer cells due to computation cost
   )

Output
------

The SHAP explainer generates comprehensive analysis files:

**Visualizations (per head):**

- ``feature_importance_bar.png``: Top features ranked by importance
- ``beeswarm_plot.png``: SHAP values distribution (feature impact on attention)
- ``feature_distribution_top20.html``: Interactive Plotly distribution plots

Interpretation
--------------

**Feature Importance Bar Chart:**

Shows the top N features ranked by mean absolute SHAP value. Higher values indicate features that strongly influence attention weights (either positively or negatively).

**Beeswarm Plot:**

Each point represents a cell, with:

- **X-axis**: SHAP value (impact on attention)
  
  - Positive: Increases attention
  - Negative: Decreases attention

- **Color**: Feature value (red = high, blue = low)
- **Y-axis**: Features ranked by importance
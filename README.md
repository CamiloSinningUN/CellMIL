# CellMIL

![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)
![PyTorch 2.7.1](https://img.shields.io/badge/pytorch-2.7.1-orange.svg)
[![Documentation](https://img.shields.io/badge/docs-online-blue)](https://cellmil.netlify.app/)

A Controllable and Flexible Multiple Instance Learning Framework for Explainable Digital Pathology Using Cell-Level Features.

![CellMIL Overview](static/overview.png#gh-light-mode-only)
![CellMIL Overview](static/overview_inv.png#gh-dark-mode-only)

---

📚 **For comprehensive documentation, tutorials, and API reference, visit our [official documentation](https://cellmil.netlify.app/).**  
This README provides a quick overview and getting started guide. For detailed information on each component, advanced usage, and examples, please refer to the full documentation.

---

## Table of Contents

- [CellMIL](#cellmil)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
    - [Development Setup](#development-setup)
      - [1. Clone the Repository](#1-clone-the-repository)
      - [2. Create Conda Environment](#2-create-conda-environment)
      - [3. Install Python Dependencies](#3-install-python-dependencies)
      - [5. Install Additional Dependencies](#5-install-additional-dependencies)
      - [Windows DataLoader Workers](#windows-dataloader-workers)
      - [PyTorch Geometric Libraries](#pytorch-geometric-libraries)
      - [Pyradiomics Installation](#pyradiomics-installation)
  - [CLI Tools](#cli-tools)
    - [Data Preparation](#data-preparation)
    - [Cell segmentation](#cell-segmentation)
    - [Create graph](#create-graph)
    - [Feature extraction](#feature-extraction)
      - [Morphological](#morphological)
      - [Topological](#topological)
      - [Patch embeddings](#patch-embeddings)
    - [Feature visualization](#feature-visualization)
    - [Evaluation Report](#evaluation-report)
    - [External Validation Report](#external-validation-report)
    - [Create dataset](#create-dataset)
  - [Training MIL Models](#training-mil-models)
  - [Explainability](#explainability)
  - [Example Use Case](#example-use-case)
  - [Technical Details](#technical-details)
    - [Supported Input Formats](#supported-input-formats)
    - [Metadata Excel Format](#metadata-excel-format)
    - [Output Directory Structure](#output-directory-structure)
    - [Resource Requirements](#resource-requirements)
  - [Development](#development)
    - [Project Structure](#project-structure)
    - [Running Tests](#running-tests)
    - [Building Documentation](#building-documentation)
  - [References](#references)
    - [Multiple Instance Learning Models](#multiple-instance-learning-models)
    - [Cell Segmentation Models](#cell-segmentation-models)
    - [Tools and Frameworks](#tools-and-frameworks)
    - [Others](#others)
  - [Contributions](#contributions)

## Installation

### Development Setup

For users who want to modify the code or contribute to the project. This is the **recommended** approach for researchers and developers.

#### 1. Clone the Repository

```bash
git clone https://github.com/[ANONYMOUS]/CellMIL.git
cd CellMIL
```

#### 2. Create Conda Environment

> **Note:** This step can be skipped if you already have Python 3.10 and Poetry installed on your machine.

If you don't have conda installed, follow the instructions [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).

```bash
# Create environment
conda env create -f environment.yml

# Activate environment
conda activate cellmil
```

#### 3. Install Python Dependencies

```bash
poetry install
```

> **Note:** Poetry will install PyTorch with CUDA 11.8 support. If you have a GPU with an older CUDA version, install PyTorch manually. Visit [pytorch.org](https://pytorch.org/get-started/locally/) to get the correct command for your system.

#### 4. Install Additional Dependencies

```bash
# Pyradiomics (incompatible with Poetry resolver)
pip install pyradiomics
```

> **Optional:** Install cucim for GPU-accelerated image loading from their [official documentation](https://docs.rapids.ai/api/cucim/stable/)

<details>
<summary><h3>Common Issues</h3></summary>

#### Windows DataLoader Workers

If you encounter errors or duplicate runs, Windows may have issues with `num_workers` in DataLoader.

**Solution:** Comment out all `num_workers` arguments in the codebase.

#### PyTorch Geometric Libraries

Errors with `torch-sparse`, `torch-scatter`, or `pyg-lib` may occur due to pre-built binary incompatibilities.

> **Note:** This is only necessary if your GPU has a CUDA version older than 11.8 or you are experiencing specific errors. Otherwise, the default installation should work fine.

**Solution:** Compile the libraries from source:

```bash
pip install --no-binary :all: torch-sparse
pip install --no-binary :all: torch-scatter
pip install --no-binary :all: pyg-lib
```

#### Pyradiomics Installation

When installing `pyradiomics` with pip, you may encounter build errors like:

```
ModuleNotFoundError: No module named 'numpy'
ModuleNotFoundError: No module named 'versioneer'
```

This happens because pyradiomics has undeclared build dependencies, and pip's build isolation creates a clean environment without access to your installed packages.

**Solution:** Install the missing build dependencies first, then install pyradiomics with `--no-build-isolation`:

```bash
pip install versioneer cython
pip install pyradiomics --no-build-isolation
```

</details>

## CLI Tools

Every step of the pipeline can be executed using the provided CLI tools. Bellow there is a brief description of each step along with example commands. For a more detailed description of each command please refer to the documentation [here](https://cellmil.netlify.app/).

> ℹ️ **Note:** Options marked with ⭐ are recommended defaults based on best practices and empirical results.

### Data Preparation

The project includes a CLI tool for preparing WSI (Whole Slide Image) data for analysis a.k.a. Extracting the patches:

```bash
patch_extraction --output_path ./results   --wsi_path ./data/SLIDE_1.svs   --patch_size 256   --patch_overlap 6.25   --target_mag 40.0

```

### Cell segmentation

After preparing the data you can run cell segmentation on the slide using the follwing cli tool:

```bash
cell_segmentation --model cellvit --gpu 0  --wsi_path ./data/SLIDE_1.svs   --patched_slide_path ./results/SLIDE_1

```

Model options:
1. `cellvit` ⭐
2. `hovernet`
3. `cellpose_sam`

> **Note:** `cellpose_sam` does not provide cell type information, which is required for the Head4Type MIL model. Use `cellvit` or `hovernet` if you need cell types.

The results of the segmentation will be stored in the `patched_slide_path` folder under the subfolder `cell_detection / {model}`.

### Create graph

After extracting the cell instances from the slide:

```bash
graph_creation  --method delaunay_radius --gpu 0 --patched_slide_path ./results/SLIDE_1  --segmentation_model cellvit

```

Method options:
1. `knn`
2. `radius`
3. `delaunay_radius` ⭐
4. `dilate`
5. `similarity`

The results of the graph creation will be stored in the `patched_slide_path` folder under the subfolder `graphs / {graph_method} / {segmentation_model}`

### Feature extraction

#### Morphological

After extracting the cell instances from the slide:

```bash
feature_extraction  --extractor pyradiomics_hed --wsi_path ./data/SLIDE_1.svs  --patched_slide_path ./results/SLIDE_1  --segmentation_model cellvit
```

Extractor options:
1. `pyradiomics_gray`
2. `pyradiomics_hed` ⭐
3. `pyradiomics_hue`
4. `morphometrics`


#### Topological

After extracting the cell instances from the slide and creating a graph with one of the available methods:

```bash
feature_extraction  --extractor connectivity --patched_slide_path ./results/SLIDE_1  --segmentation_model cellvit --graph_method knn
```

Extractor options:
1. `connectivity`
2. `geometric`
<!-- 3. `structure` -->

#### Patch embeddings

After extracting the patches:

```bash
feature_extraction  --extractor resnet50 --patched_slide_path ./results/SLIDE_1
```

Extractor options:
1. `resnet50`
2. `gigapath`
3. `uni`

### Feature visualization

After extracting the features we can run the following command to have a visualization of the features extracted:

```bash
vis_features --dataset ./results
```

### Evaluation Report

After doing experiments this command will generate a report with series of plots with the evaluation metrics obtained:

```bash
eval_report --metrics f1 recall auroc c_index --team [ANONYMOUS] --projects 'CELLMIL' --output-dir ../evaluation_reports
```

Metric options:
1. f1
2. recall
3. precision
4. auroc
5. c_index

### External Validation Report

After doing experiments this command will run the external validation with the modality and dataset specified and generate a report with the evaluation metrics obtained:

```bash
eval_external --metrics f1 recall precision auroc c_index --models-dir ../experiments/checkpoints --output-dir ../external_evaluation --final-model ensemble --aggregation-method median --dataset-dir ../dataset_uoc --root-dir ../UOC_MIL_dataset --dp-metadata-file ../data/metadata_UOC.xlsx
```

Metric options:
1. f1
2. recall
3. precision
4. auroc
5. c_index

Final model options:
1. final
2. ensemble ⭐

Aggregation method options:
1. majority
2. median ⭐
3. mean
4. everything (Use all aggregation methods)

### Create dataset

This command takes the metadata excel and process all the slides present on it to then use them to train the MIL model.

```bash
create_dataset --excel_path ./data/metadata.xlsx --output_path ./results --gpu 0 --segmentation_models cellvit hovernet cellpose_sam --extractors handcrafted topology_measures --graph_methods knn radius
```

## Training MIL Models

After preparing your dataset, you can train Multiple Instance Learning models on your processed slides. MIL enables training models on whole slide images using only slide-level labels, where each slide is treated as a "bag" of cell instances.

**Supported Tasks:**
- **Binary Classification**: Predict outcomes like treatment response, histological subtype, or disease status
- **Survival Prediction**: Time-to-event analysis for prognosis prediction

**Available Models:**
- **ABMIL (Attention-based Deep MIL)**: Standard attention pooling over cell instances
- **CLAM**: Clustering-constrained attention with multi-instance learning
- **TransMIL**: Transformer-based MIL for capturing long-range dependencies
- **Head4Type**: Cell type-aware attention with separate heads for each cell type (requires cellvit/hovernet)
- **HistoBistro**: Transformer architecture optimized for histopathology
- **GraphMIL**: Graph neural network-based MIL for leveraging spatial relationships between cells

**Quick Example:**

```python
from pathlib import Path
import pandas as pd
from cellmil.datamodels.datasets import MILDataset
from cellmil.utils.train.evals import KFoldCrossValidation

# Load metadata
df = pd.read_excel("./data/metadata.xlsx")
df = df[df["label"].isin([0, 1])]

# Create dataset
dataset = MILDataset(
    root=Path("./MIL_dataset"),
    label="label",
    folder=Path("./dataset"),
    data=df,
    extractor="pyradiomics_hed",
    segmentation_model="cellvit",
)

# Define model creator function
def create_model(input_dim: int):
    from cellmil.models.mil.abmil import AttentionDeepMIL, LitAttentionDeepMIL
    from torch.optim import AdamW
    
    model = AttentionDeepMIL(embed_dim=input_dim, n_classes=2)
    lit_model = LitAttentionDeepMIL(
        model=model,
        optimizer=AdamW(model.parameters(), lr=1e-4)
    )
    return lit_model

# Train with cross-validation
k_fold = KFoldCrossValidation(
    dataset=dataset,
    lit_model_creator=create_model,
    n_splits=5
)
k_fold.run()
```

For detailed instructions on model configuration, hyperparameter tuning, and advanced training options, refer to the [MIL Training Documentation](https://cellmil.netlify.app/pipeline/mil_training/index.html).

## Explainability

CellMIL provides two complementary methods for explaining model predictions and understanding what the models learn from the data.

### Attention Heatmaps

Visualize which cells the model focuses on when making predictions for specific slides. This provides **local, instance-level explanations**.

**Supported Models:**
- CLAM, ABMIL, Head4Type

**Output Formats:**
- GeoJSON for pathology viewers (QuPath, etc.)
- Interactive graph visualizations

**Quick Example:**

```python
from cellmil.explainability.attention import AttentionExplainer
from cellmil.interfaces.AttentionExplainerConfig import AttentionExplainerConfig
from cellmil.datamodels.model import ModelStorage
from pathlib import Path

# Load trained model
model_storage = ModelStorage.from_directory("./results/trained_model")

# Configure explainer
config = AttentionExplainerConfig(
    output_path=Path("./explanations/attention"),
    visualization_mode="geojson"
)

# Initialize explainer
explainer = AttentionExplainer(config)

# Generate explanation for a specific slide
results = explainer.generate_explanation(
    model_storage=model_storage,
    slide_path=Path("./dataset/slide_name"),
)
```

### SHAP Analysis

Identify which cell features (morphological, textural, topological) drive the attention mechanism. This provides **global feature importance**.

**Key Insight:** SHAP analyzes which features lead to high or low attention scores.

**Quick Example:**

```python
from cellmil.explainability.shap import SHAPExplainer
from cellmil.interfaces.SHAPExplainerConfig import SHAPExplainerConfig
from cellmil.datamodels.model import ModelStorage
import pandas as pd
from pathlib import Path

# Load trained model
model_storage = ModelStorage.from_directory("./results/trained_model")

# Load metadata (must have 'FULL_PATH' column with slide names)
metadata = pd.read_excel("./data/metadata.xlsx")

# Configure SHAP explainer
config = SHAPExplainerConfig(
    output_path=Path("./explanations/shap")
)

# Initialize explainer
shap_explainer = SHAPExplainer(config)

# Generate explanation
results = shap_explainer.generate_explanation(
    model_storage=model_storage,
    dataset_folder=Path("./dataset"),
    data=metadata,
)
```

For more details on visualization options, normalization methods, and interpretation, see the [Explainability Documentation](https://cellmil.netlify.app/explainability.html).

## Example Use Case

For a complete end-to-end example demonstrating the full CellMIL pipeline, see our **NSCLC Immunotherapy Response Prediction** experiment:

📁 [experiments/NSCLC_IO_response/](experiments/NSCLC_IO_response/)

> **Note:** The dataset is private and cannot be provided, but the complete experimental code serves as a template for applying CellMIL to your own histopathology datasets.

## Technical Details

### Supported Input Formats

CellMIL supports Whole Slide Images (WSI) through [OpenSlide](https://openslide.org/).

### Metadata Excel Format

The `create_dataset` command requires an Excel file (`.xlsx`) with slide metadata. The file should contain the following columns:

| Column | Required | Description |
|--------|----------|-------------|
| `PATH` | ✅ | Absolute path to the WSI file |

> **Note:** The pipeline currently only supports **40x magnification**. For details on training with labels and other configurations, please refer to the [documentation](https://cellmil.netlify.app/).

Example:

| PATH                           |
|--------------------------------|
| /data/slides/SLIDE_1.svs       |
| /data/slides/SLIDE_2.svs       |


### Output Directory Structure

After running the pipeline, the output directory will have the following structure:

```
results/
├── {slide_name}/
│   ├── patches/                           # Extracted patches
│   │   ├── 0_0.png
│   │   └── ...
│   ├── cell_detection/
│   │   └── {segmentation_model}/          # Cell segmentation results
│   │       ├── cells.json
│   │       └── ...
│   ├── graphs/
│   │   └── {graph_method}/
│   │       └── {segmentation_model}/      # Graph representations
│   │           └── graph.pt
│   ├── features/
│   │   └── {extractor}/                   # Extracted features
│   │       └── features.pt
│   ├── thumbnails/                        # WSI thumbnails
│   └── metadata.json                      # Slide metadata
├── log.json                               # Processing progress log
└── processed.json                         # Dataset processing status
```

### Resource Requirements

For optimal performance, it is recommended to use a machine with at least a modern GPU (e.g., NVIDIA RTX 2080 or higher) and sufficient RAM (16GB or more). The exact requirements may vary based on the size of the WSI files and the complexity of the models used.

## Development

### Project Structure

```
cellmil/
├── cli/                # Command-line interface tools
├── config/             # Configuration files
├── data/               # Data loading and patch extraction
├── datamodels/         # Data models and schemas
├── dataset/            # Dataset creation utilities
├── explainability/     # Model explanation tools
├── features/           # Feature extraction modules
├── graph/              # Graph construction methods
├── interfaces/         # Pydantic configuration interfaces
├── models/             # MIL model implementations
├── segmentation/       # Cell segmentation models
├── statistics/         # Statistical analysis utilities
├── utils/              # Utility functions
└── visualization/      # Visualization tools
```

### Running Tests

The project uses `pytest` for testing. Tests are located in `cellmil/__tests__/`.

```bash
# Run all tests
pytest
```

Test reports are generated in `test_reports/report.html`.

### Building Documentation

The documentation is built using Sphinx and automatically deployed to GitHub Pages.

```bash
# Build documentation locally
cd doc
make html
```

Documentation is available at: [https://cellmil.netlify.app/](https://cellmil.netlify.app/)

## References

This project builds upon several key research papers and tools:

### Multiple Instance Learning Models

- **ABMIL** M. Ilse, J. Tomczak, and M. Welling. Attention-based  deep multiple instance learning. pages 2127–2136, 2018.

- **CLAM**: Data-efficient and weakly supervised computational pathology on whole-slide images  
  Lu, Ming Y et al., Nature Biomedical Engineering, 2021  
  [DOI: 10.1038/s41551-021-00707-9](https://doi.org/10.1038/s41551-021-00707-9)

- **TransMIL**: Transformer based correlated multiple instance learning for whole slide image classification  
  Shao, Zhuchen et al., Advances in Neural Information Processing Systems, 2021  
  [NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/10c272d06794d3e5785d5e7c5356e9ff-Abstract.html)

- **HistoBistro**: Transformer-based biomarker prediction from colorectal cancer histology: A large-scale multicentric study  
  Wagner, Sophia J et al., Cancer Cell, Elsevier  
  [DOI: 10.1016/j.ccell.2023.02.002](https://doi.org/10.1016/j.ccell.2023.02.002)

### Cell Segmentation Models

- **CellViT**: Vision Transformers for precise cell segmentation and classification  
  Fabian Hörst et al., Medical Image Analysis, 2024  
  [DOI: 10.1016/j.media.2024.103143](https://doi.org/10.1016/j.media.2024.103143)

- **HoVerNet**: Hover-Net: Simultaneous segmentation and classification of nuclei in multi-tissue histology images  
  Graham, Simon et al., Medical Image Analysis, 2019  
  [DOI: 10.1016/j.media.2019.101563](https://doi.org/10.1016/j.media.2019.101563)

- **CellposeSAM**: Cellpose-SAM: superhuman generalization for cellular segmentation  
  Pachitariu, Marius et al., bioRxiv preprint, 2025  
  [DOI: 10.1101/2025.04.28.651001](https://doi.org/10.1101/2025.04.28.651001)

### Tools and Frameworks

- **PathML**: Building tools for machine learning and artificial intelligence in cancer research: best practices and a case study   with the PathML toolkit for computational pathology  
  Rosenthal, J. et al., Molecular Cancer Research, 2022  
  [DOI: 10.1158/1541-7786.MCR-21-0665](https://doi.org/10.1158/1541-7786.MCR-21-0665)

### Foundational Models

- **ResNet**: Deep Residual Learning for Image Recognition  
  He, Kaiming et al., Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2016  
  [DOI: 10.1109/CVPR.2016.90](https://doi.org/10.1109/CVPR.2016.90)

- **GigaPath**: A whole-slide foundation model for digital pathology from real-world data  
  Xu, Hanwen et al., Nature, 2024

- **UNI**: Towards a General-Purpose Foundation Model for Computational Pathology  
  Chen, Richard J et al., Nature Medicine, 2024  

### Datasets

Images used in documentation are from this dataset:

- **CPTAC**: National Cancer Institute Clinical Proteomic Tumor Analysis Consortium (CPTAC). The clinical proteomic tumor analysis consortium lung adenocarcinoma collection (cptac-luad), 2018. 
[URL: https://www.cancerimagingarchive.net/collection/cptac-luad/](https://www.cancerimagingarchive.net/collection/cptac-luad/)  
  

### Others

- **Ceograph**: Deep learning of cell spatial organizations identifies clinically relevant insights in tissue images  
  Wang, Shidan et al., Nature Communications, 2023  
  [DOI: 10.1038/s41467-023-43172-6](https://doi.org/10.1038/s41467-023-43172-6)

- **Pyradiomics** van Griethuysen, J. J. M., Fedorov, A., Parmar, C., Hosny, A., Aucoin, N., Narayan, V., Beets-Tan, R. G. H., Fillion-Robin, J. C., Pieper, S., Aerts, H. J. W. L. (2017). Computational Radiomics System to Decode the Radiographic Phenotype. Cancer Research, 77(21), e104–e107. https://doi.org/10.1158/0008-5472.CAN-17-0339

## Contributions

* [Anonymous]

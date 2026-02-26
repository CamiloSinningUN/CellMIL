# NSCLC Immunotherapy Response Prediction

This experiment demonstrates the complete CellMIL pipeline applied to Non-Small Cell Lung Cancer (NSCLC) immunotherapy response prediction. It includes dataset creation, model training, performance evaluation, and explainability analysis across multiple cohorts and prediction tasks.

> **Dataset Availability**: The datasets used in this experiment are private and cannot be provided publicly due to institutional data protection policies.

> **Experiment Tracking**: All training runs, metrics, and hyperparameters are publicly available on Weights & Biases:  
> [https://wandb.ai/camilosinning-cs-politecnico-di-milano/CELLMIL](https://wandb.ai/camilosinning-cs-politecnico-di-milano/CELLMIL)

## Experiment Overview

This study evaluates cell-level histopathology features for predicting immunotherapy response and patient outcomes in NSCLC. The experiment includes:

- **Two cohorts**: 
  - **INT**: Internal training cohort
  - **UOC**: External validation cohort

- **Prediction tasks**:
  - **Classification**: DCR (Disease Control Rate), OS6/OS24 (Overall Survival at 6/24 months), PDL1 status, ADENOvsSQUA (Adenocarcinoma vs Squamous)
  - **Survival**: OS (Overall Survival), PFS (Progression-Free Survival)

- **Feature extractors**:
  - Morphological: `morphometrics`, 
  - Texture: `pyradiomics_hed`
  - Topological: `connectivity`, `geometric`
  - Deep learning: `resnet50`, `gigapath`, `uni`
  - Combined: `ALL` (morphological + texture + topological)

- **MIL models**: ABMIL, CLAM, Head4Type

- **Training configurations**: With/without regularization and cell stratification

## Directory Structure

```
NSCLC_IO_response/
├── README.md                          # This file
├── dataset_creation_INT.sh            # Create INT training dataset
├── dataset_creation_UOC.sh            # Create UOC validation dataset
├── INT_performance.sh                 # Evaluate models on INT cohort
├── UOC_performance.sh                 # Evaluate models on UOC cohort
├── training_scripts/                  # Model training scripts
│   ├── [LABEL]+[FEATURES]+[MODEL]+[REG]+[STRA].py
│   └── ...
├── explainability/                    # Explainability analysis
│   ├── [LABEL]+[FEATURES]+[MODEL]+[REG]+[STRA]/
│   │   ├── attention_heatmap.py
│   │   ├── shap.py
│   │   ├── INT/                       # SHAP analysis results
│   │   └── SLIDE_*/                   # Attention heatmaps per slide
├── INT_performance/                   # INT cohort results
│   ├── classification_*.tex           # LaTeX tables
│   ├── survival_*.tex
│   ├── [metric]/[label].html          # Interactive result visualizations
├── UOC_performance/                   # UOC cohort results
│   └── median/                        # Median performance across runs
├── results/                           # Trained models and checkpoints (Included in release)
└── data/                              # (Not included - private dataset)
    ├── metadata_INT.xlsx
    └── metadata_UOC.xlsx
```

## Script Naming Convention

Training scripts and result folders follow a systematic naming convention:

```
[LABEL]+[FEATURES]+[MODEL]+[REG]+[STRA]
```

**Components:**
- `LABEL`: Prediction task (DCR, OS6, OS24, OS, PFS, PDL1, ADENOvsSQUA)
- `FEATURES`: Feature type (MORPHO, PYRAD, TOPO, ALL, GIGAPATH, RESNET, UNI)
- `MODEL`: MIL architecture (ABMIL, CLAM, HEAD4TYPE)
- `REG`: Regularization (`REG` = enabled, `*` = disabled)
- `STRA`: Cell stratification (`STRA` = enabled, `*` = disabled)

**Examples:**
- `DCR+ALL+CLAM+REG+STRA`: DCR prediction using all handcrafted features with CLAM, regularization, and cell stratification

---

For general CellMIL usage and documentation, see the [main README](../../README.md) and [full documentation](https://camilosinningun.github.io/CellMIL/).

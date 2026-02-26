import logging
from pathlib import Path
import pandas as pd
import re
from cellmil.interfaces.CellSegmenterConfig import ModelType
from cellmil.interfaces.GraphCreatorConfig import GraphCreatorType
from cellmil.datamodels.datasets import MILDataset
from cellmil.utils.train.evals.k_fold_cross_validation import KFoldCrossValidation
from cellmil.datamodels.transforms import (
    TransformPipeline,
    TimeDiscretizerTransform,
    CorrelationFilterTransform,
    RobustScalerTransform,
    Transform,
)
from cellmil.utils.train import (
    get_extractors_from_name,
    preprocess_df,
    get_lit_model_creator,
)

root_path = Path("../../")

# --- CONSTANTS ---
ROOT = root_path / "MIL_dataset"
DP_METADATA_PATH = root_path / "data" / "metadata_INT.xlsx"
DATASET_FOLDER = root_path / "dataset"
GPU = 0
SEGMENTATION_MODEL = ModelType.cellvit
GRAPH_CREATOR = GraphCreatorType.delaunay_radius
N_BINS = 4
WANDB_PROJECT = "CELLMIL"
# -----------------


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Expected format: [LABEL1, LABEL2]+[FEATURES1,FEATURES2]+[MODEL1, MODEL2]+[REG,_]+[STRA,_].py
filename = Path(__file__).stem

# Parse filename to extract experiment parameters
# Pattern: [items,items]+[items,items]+...
pattern = r"\[([^\]]+)\]"
matches = re.findall(pattern, filename)

if len(matches) != 5:
    raise ValueError(
        f"Filename {filename} does not match expected format. Expected 5 groups, found {len(matches)}"
    )

# Extract and process each group
TASKS: list[str] = [item.strip() for item in matches[0].split(",")]
FEATURES: list[str] = [item.strip() for item in matches[1].split(",")]
MODELS: list[str] = [item.strip() for item in matches[2].split(",")]
REG: list[str] = [item.strip() for item in matches[3].split(",")]
STRA: list[str] = [item.strip() for item in matches[4].split(",")]

logger.info(f"Parsed filename: {filename}")
logger.info(f"Tasks: {TASKS}")
logger.info(f"Features: {FEATURES}")
logger.info(f"Models: {MODELS}")
logger.info(f"Regularization: {REG}")
logger.info(f"Stratification: {STRA}")


for task in TASKS:
    for feature in FEATURES:
        for model in MODELS:
            for reg in REG:
                for stra in STRA:
                    print(
                        f"Running experiment with Task: {task}, Feature: {feature}, Model: {model}, Reg: {reg}, Stra: {stra}"
                    )
                    NAME = f"{task}+{feature}+{model}+{reg}+{stra}"

                    try:
                        if (feature in ["RESNET", "GIGAPATH"] and stra == "STRA"):
                            raise ValueError(f"Stratification not applicable for feature {feature}")
                        
                        if (feature in ["RESNET", "GIGAPATH"] and model == "HEAD4TYPE"):
                            raise ValueError(f"Cell type model not applicable for feature {feature}")
                        
                        # --- Set up variables from constants ---
                        normalization = (
                            True
                            if feature != "RESNET" and feature != "GIGAPATH"
                            else False
                        )
                        correlation_filter = 0.95 if feature == "ALL" else 0.0
                        regularization = True if reg == "REG" else False
                        cell_stratification = True if stra == "STRA" else False

                        # --- Set up Dataset configuration ---
                        extractors = get_extractors_from_name(feature)

                        df = pd.read_excel( # type: ignore
                            root_path / "data" / "metadata_INT.xlsx"
                        )  

                        df = preprocess_df(df, task)

                        dataset = MILDataset(
                            root=ROOT,
                            label=task
                            if task not in ["OS", "PFS"]
                            else ("duration", "event"),
                            folder=DATASET_FOLDER,
                            data=df,
                            extractor=extractors,
                            segmentation_model=SEGMENTATION_MODEL,
                            graph_creator=GRAPH_CREATOR,
                            cell_type=True if model == "HEAD4TYPE" else False,
                        )

                        print(
                            f"Shape of a single feature vector: {dataset[0][0].shape}"
                        )

                        # --- Create transforms ---
                        transform_list: list[Transform] = []

                        if correlation_filter > 0.0:
                            transform_list.append(
                                CorrelationFilterTransform(
                                    correlation_threshold=correlation_filter,
                                    plot_correlation_matrix=False,
                                )
                            )
                        if normalization:
                            transform_list.append(
                                RobustScalerTransform(apply_log_transform=True)
                            )

                        transforms = TransformPipeline(transform_list)

                        label_transforms = TimeDiscretizerTransform(n_bins=N_BINS)

                        # --- Model definition ---
                        lit_model_creator = get_lit_model_creator(
                            model=model,
                            task=task,
                            n_bins=N_BINS,
                            feature=feature,
                            df=df,
                            regularization=regularization,
                        )

                        k_fold = KFoldCrossValidation(k=5)
                        model_storage = k_fold.evaluate(
                            name=NAME,
                            lit_model_creator=lit_model_creator,
                            dataset=dataset,
                            output_dir=Path("./results"),
                            wandb_project=WANDB_PROJECT,
                            transforms=transforms,
                            label_transforms=label_transforms
                            if task in ["OS", "PFS"]
                            else None,
                            balance_cell_counts=cell_stratification,
                        )

                    except Exception as e:
                        logger.error(f"Error in experiment {NAME}: {e}")
                        # Ensure wandb is finished to prevent name leakage
                        try:
                            import wandb
                            wandb.finish()
                        except:
                            pass
                        continue

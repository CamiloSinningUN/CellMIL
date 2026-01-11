"""
SHAP Explainer for MIL Models.

This module provides SHAP-based explainability for attention-based MIL models.
It creates a cell-level dataset from all slides, computes attention weights,
performs stratified sampling based on attention quantiles, and computes SHAP values.
"""

from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, cast
import torch
import numpy as np
import pandas as pd
import lightning as Pl
import json
import re
from tqdm import tqdm
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

from cellmil.interfaces.SHAPExplainerConfig import SHAPExplainerConfig
from cellmil.interfaces.FeatureExtractorConfig import ExtractorType
from cellmil.interfaces.CellSegmenterConfig import ModelType
from cellmil.interfaces.GraphCreatorConfig import GraphCreatorType
from cellmil.datamodels.transforms import TransformPipeline
from cellmil.datamodels.model import ModelStorage
from cellmil.datamodels.datasets.utils import (
    get_cell_features,
    get_cell_types,
    cell_types_to_tensor,
)
from cellmil.explainability.shap.core import AttentionStratifiedSampler, SHAPComputer
from cellmil.explainability.shap.visualizers import SHAPVisualizer
from cellmil.utils import logger
from cellmil.models.mil import LitAttentionDeepMIL
from cellmil.models.mil.head4type import LitHead4Type
from cellmil.models.mil.clam import LitCLAM

# Model class registry for loading from checkpoint
MODEL_CLASS_REGISTRY = {
    "LitAttentionDeepMIL": LitAttentionDeepMIL,
    "LitCLAM": LitCLAM,
    "LitHead4Type": LitHead4Type,
}


def _parse_enum_config(value: Any) -> Any:
    """
    Parse configuration values that may be stored as enum string representations.
    
    Handles cases where values are stored as:
    - String representation of enums: "<ExtractorType.morphometrics: 'morphometrics'>"
    - List of enum representations: "[<ExtractorType.morphometrics: 'morphometrics'>, ...]"
    - Regular strings or lists: "morphometrics" or ["morphometrics", ...]
    
    Args:
        value: The configuration value to parse
        
    Returns:
        Parsed value with enum representations converted to their string values
    """
    if value is None:
        return None
    
    # Convert to string for processing
    value_str = str(value)
    
    # Check if it's a list representation (starts with [ and ends with ])
    if value_str.strip().startswith("[") and value_str.strip().endswith("]"):
        # Extract all quoted strings from the list
        matches = re.findall(r"'([^']*)'", value_str)
        if matches:
            return matches if len(matches) > 1 else matches[0]
    
    # Check if it's a single enum representation
    if "<" in value_str and ":" in value_str and ">" in value_str:
        # Extract the quoted value from enum: <Type.name: 'value'>
        match = re.search(r"'([^']*)'", value_str)
        if match:
            return match.group(1)
    
    # Return as-is if no enum representation found
    return value


class SHAPExplainer:
    """
    SHAP-based explainability for attention-based MIL models.

    This class creates a cell-level dataset from multiple slides, computes attention
    weights for all cells, performs stratified sampling based on attention quantiles,
    and uses SHAP to explain which features are most important for the attention mechanism.
    """

    def __init__(self, config: SHAPExplainerConfig):
        """
        Initialize the SHAP explainer.

        Args:
            config: Configuration for SHAP explanation process
        """
        self.config = config
        np.random.seed(config.random_seed)
        torch.manual_seed(config.random_seed)  # type: ignore

        # Initialize components
        self.sampler = AttentionStratifiedSampler(
            num_bins=config.num_bins,
            samples_per_bin=config.samples_per_bin,
            random_seed=config.random_seed,
        )

        self.shap_computer = SHAPComputer(
            explainer_type=config.explainer_type,
            background_percentage=config.background_percentage,
            nsamples=config.nsamples,
            explain_top_cells=config.explain_top_cells,
            explain_per_head=config.explain_per_head,
            explain_mean_head=config.explain_mean_head,
        )

        self.visualizer = SHAPVisualizer(config)

        logger.info("Initialized SHAP explainer")
        logger.info(f"  Explainer type: {config.explainer_type.value}")
        logger.info(
            f"  Stratified sampling: {config.num_bins} bins, {config.samples_per_bin} samples/bin"
        )
        logger.info(
            f"  Background: {config.background_percentage * 100:.1f}% of sampled cells"
        )
        if config.explainer_type == "kernel":
            logger.info(f"  SHAP nsamples: {config.nsamples}")

    def generate_explanation(
        self,
        model_storage: ModelStorage,
        dataset_folder: Path | str,
        data: pd.DataFrame,
        fold_idx: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Generate SHAP explanations for the given model and dataset.

        Args:
            model_storage: ModelStorage instance containing model checkpoint, transforms, and configuration
            dataset_folder: Path to the folder containing all slide data
            data: DataFrame with slide metadata (must have 'FULL_PATH' column)
            fold_idx: Optional fold index to use. If None, uses the final model. Must be in range [0, k_folds-1]
            **kwargs: Additional arguments

        Returns:
            Dictionary containing SHAP values, explanations, and metadata
        """
        logger.info("Starting SHAP explanation generation")

        # Load experiment metadata
        if model_storage.experiment_metadata is None:
            raise ValueError("No experiment metadata found in model_storage")

        experiment_metadata = model_storage.experiment_metadata
        dataset_config = experiment_metadata.dataset_config
        model_config = experiment_metadata.model_config

        # Extract configuration from model_storage
        extractor = _parse_enum_config(dataset_config.get("extractor"))
        segmentation_model = _parse_enum_config(dataset_config.get("segmentation_model"))
        graph_creator = _parse_enum_config(dataset_config.get("graph_creator"))

        if extractor is None or segmentation_model is None:
            raise ValueError(
                "extractor and segmentation_model must be in dataset_config"
            )

        logger.info(f"Loaded config - Extractor: {extractor}, Segmentation: {segmentation_model}")

        # Determine which model and transforms to load
        if fold_idx is not None:
            # Validate fold index
            available_folds = model_storage.list_folds()
            if fold_idx not in available_folds:
                raise ValueError(
                    f"fold_idx {fold_idx} not available. Available folds: {available_folds}"
                )
            logger.info(f"Loading model and transforms from fold {fold_idx}")
            checkpoint_path = model_storage.load_fold_checkpoint(fold_idx)
            transforms, label_transforms = model_storage.load_fold_transforms(fold_idx)
        else:
            # Use final model
            if not model_storage.has_final_model():
                raise ValueError("No final model found in model_storage")
            logger.info("Loading final model and transforms")
            checkpoint_path = model_storage.load_final_checkpoint()
            transforms, label_transforms = model_storage.load_final_transforms()

        # Load model from checkpoint using model class from config
        model_class_name = model_config.get("model_class")
        if model_class_name is None:
            raise ValueError("model_class not found in model_config")

        if model_class_name not in MODEL_CLASS_REGISTRY:
            raise ValueError(
                f"Unknown model class: {model_class_name}. "
                f"Available classes: {list(MODEL_CLASS_REGISTRY.keys())}"
            )

        model_class = MODEL_CLASS_REGISTRY[model_class_name]
        logger.info(f"Loading model from checkpoint: {checkpoint_path}")
        model = model_class.load_from_checkpoint(str(checkpoint_path))
        model.eval()

        # Use the fold transforms for feature transformation
        transforms_to_use = transforms if transforms is not None else None

        dataset_folder = Path(dataset_folder)

        # Check GPU availability and move model to GPU if available
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"GPU available: {torch.cuda.get_device_name(0)}")
            logger.info("Moving model to GPU for faster computation...")
            model = model.to(device)
        else:
            device = torch.device("cpu")
            logger.info("No GPU available, using CPU")

        # Validate model type
        self._validate_model(model)

        # Validate transforms are available
        if transforms_to_use is None:
            raise ValueError("No transforms found in model checkpoint")

        # Create output directory
        self.config.output_path.mkdir(parents=True, exist_ok=True)

        # Step 1: Load cell-level dataset from all slides
        logger.info("Loading cell-level dataset from all slides...")
        cell_data = self._load_cell_dataset(
            dataset_folder=dataset_folder,
            data=data,
            extractor=extractor,
            segmentation_model=segmentation_model,
            graph_creator=graph_creator,
            transforms=transforms_to_use,
        )

        logger.info(f"Total cells loaded: {cell_data['features'].shape[0]:,}")

        # Step 2: Compute SHAP values (includes attention computation and sampling internally)
        logger.info("Computing SHAP explanations...")
        shap_results = self._compute_shap_values(
            model=model,
            features=cell_data["features"],
            cell_types=cell_data.get("cell_types"),
        )

        # Step 3: Save results
        logger.info("Saving SHAP results...")
        saved_files = self._save_results(
            shap_results=shap_results,
            cell_data=cell_data,
        )

        # Compile results
        results: Dict[str, Any] = {
            "model": model.__class__.__name__,
            "total_cells": cell_data["features"].shape[0],
            "sampled_cells": len(shap_results["sampled_indices"]),
            "num_features": cell_data["features"].shape[1],
            "shap_results": shap_results,
            "saved_files": saved_files,
            "config": self.config.dict(),  # type: ignore
        }

        logger.info("SHAP explanation completed successfully")
        logger.info(f"Results saved to: {self.config.output_path}")

        return results

    def _validate_model(self, model: Pl.LightningModule) -> None:
        """Validate that the model is supported."""
        supported_models = (LitAttentionDeepMIL, LitHead4Type, LitCLAM)
        if not isinstance(model, supported_models):
            raise ValueError(
                f"Unsupported model type: {model.__class__.__name__}. "
                f"Supported types: {[cls.__name__ for cls in supported_models]}"
            )

        if not hasattr(model, "get_attention_weights"):
            raise ValueError(
                f"Model {model.__class__.__name__} must have get_attention_weights method"
            )

    @staticmethod
    def _process_single_slide(
        slide_name: str,
        dataset_folder: Path,
        extractor: ExtractorType | list[ExtractorType],
        segmentation_model: ModelType,
        graph_creator: Optional[GraphCreatorType],
        transforms: TransformPipeline,
    ) -> Optional[Dict[str, Any]]:
        """
        Process a single slide to extract features and cell types.

        Args:
            slide_name: Name of the slide to process
            dataset_folder: Path to dataset folder
            extractor: Feature extractor type
            segmentation_model: Segmentation model type
            graph_creator: Graph creation method
            transforms: Transform pipeline to apply

        Returns:
            Dictionary with features, cell_types, feature_names, and slide_name, or None if failed
        """
        try:
            # Load features for this slide
            features, cell_id_mapping, slide_feature_names = get_cell_features(
                folder=dataset_folder,
                slide_name=slide_name,
                extractor=extractor,  # type: ignore
                graph_creator=graph_creator,  # type: ignore
                segmentation_model=segmentation_model,  # type: ignore
            )

            if features is None or len(features) == 0:
                logger.warning(
                    f"No features found for slide {slide_name}, skipping"
                )
                return None

            # Apply transforms
            features = transforms.transform(features)

            # Load cell types if needed
            cell_types = None
            cell_types_dict = get_cell_types(
                folder=dataset_folder,
                slide_name=slide_name,
                segmentation_model=segmentation_model,
            )
            if cell_types_dict is not None and cell_id_mapping is not None:
                cell_types = cell_types_to_tensor(cell_types_dict, cell_id_mapping)

            return {
                "slide_name": slide_name,
                "features": features,
                "cell_types": cell_types,
                "feature_names": slide_feature_names,
                "num_cells": features.shape[0],
            }

        except Exception as e:
            logger.error(f"Error processing slide {slide_name}: {e}")
            return None

    def _load_cell_dataset(
        self,
        dataset_folder: Path,
        data: pd.DataFrame,
        extractor: ExtractorType | list[ExtractorType],
        segmentation_model: ModelType,
        graph_creator: Optional[GraphCreatorType],
        transforms: TransformPipeline,
    ) -> Dict[str, Any]:
        """
        Load cell-level features from all slides in the dataset.

        Returns:
            Dictionary with:
                - features: torch.Tensor of shape [total_cells, num_features]
                - cell_types: Optional torch.Tensor of shape [total_cells, num_types] (for Head4Type)
                - slide_indices: List of (slide_name, cell_idx) for each cell
        """
        all_features: List[torch.Tensor] = []
        all_cell_types: List[torch.Tensor] = []
        slide_indices: List[Tuple[str, int]] = []

        # Get list of slides
        slide_names = [Path(slide).stem for slide in data["FULL_PATH"].tolist()]
        logger.info(f"Processing {len(slide_names)} slides...")

        feature_names: Optional[List[str]] = None

        # Determine number of workers (use all available CPUs, but cap at reasonable limit)
        num_workers = min(multiprocessing.cpu_count(), len(slide_names), 16)
        logger.info(f"Using {num_workers} parallel workers for slide processing")

        # Process slides in parallel
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            future_to_slide = {
                executor.submit(
                    self._process_single_slide,
                    slide_name,
                    dataset_folder,
                    extractor,
                    segmentation_model,
                    graph_creator,
                    transforms,
                ): slide_name
                for slide_name in slide_names
            }

            # Collect results as they complete
            with tqdm(total=len(slide_names), desc="Loading slides") as pbar:
                for future in as_completed(future_to_slide):
                    result = future.result()  # type: ignore
                    pbar.update(1)

                    if result is None: # type: ignore
                        continue

                    # Store feature names from first slide (should be same for all)
                    if (
                        feature_names is None
                        and result["feature_names"] is not None
                    ):
                        feature_names = result["feature_names"]
                        logger.info(f"Captured {len(feature_names)} feature names") # type: ignore

                    # Add to collections
                    all_features.append(result["features"])
                    if result["cell_types"] is not None:
                        all_cell_types.append(result["cell_types"])

                    # Track which slide each cell belongs to
                    slide_indices.extend(
                        [
                            (result["slide_name"], i)
                            for i in range(result["num_cells"])
                        ]
                    )

        if len(all_features) == 0:
            raise ValueError("No valid features found in any slide")

        # Concatenate all features
        features_tensor = torch.cat(all_features, dim=0)
        logger.info(f"Concatenated features shape: {features_tensor.shape}")

        result: Dict[str, Any] = {
            "features": features_tensor,
            "slide_indices": slide_indices,
        }

        # Add feature names if available
        if feature_names is not None:
            result["feature_names"] = feature_names
            logger.info(f"Stored {len(feature_names)} feature names")

        if len(all_cell_types) > 0:
            cell_types_tensor = torch.cat(all_cell_types, dim=0)
            result["cell_types"] = cell_types_tensor
            logger.info(f"Cell types tensor shape: {cell_types_tensor.shape}")

        return result

    def _compute_shap_values(
        self,
        model: Pl.LightningModule,
        features: torch.Tensor,
        cell_types: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Compute SHAP values for cells.
        Delegates to SHAPComputer which handles attention computation, sampling, and SHAP.

        Args:
            model: The MIL model
            features: Full feature tensor [total_cells, num_features]
            cell_types: Optional cell types tensor

        Returns:
            Dictionary containing SHAP values, attention scores, sampling info, etc.
        """
        device = next(model.parameters()).device
        model.eval()

        logger.info(f"Total cells for analysis: {features.shape[0]:,}")

        # Computer handles: attention computation → sampling → SHAP computation
        shap_results = self.shap_computer.compute_shap_values(
            model=cast(LitAttentionDeepMIL | LitHead4Type, model),
            all_features=cast(np.ndarray[Any, Any], features.numpy()),  # type: ignore
            device=device,
            sampler=self.sampler,
            max_total_samples=self.config.max_total_samples,
        )

        return shap_results

    def _save_results(
        self,
        shap_results: Dict[str, Any],
        cell_data: Dict[str, Any],
    ) -> Dict[str, List[Path]]:
        """Save SHAP results and visualizations."""
        # Extract from shap_results
        sampled_indices = shap_results["sampled_indices"]
        sampling_info = shap_results["sampling_info"]
        attention_scores = shap_results["attention_scores"]

        saved_files: Dict[str, List[Path]] = {
            "data": [],
            "plots": [],
            "metadata": [],
        }

        # Save raw SHAP values (per head)
        if self.config.save_raw_shap_values:
            shap_file = self.config.output_path / "shap_values.npz"

            # Prepare data to save
            save_dict = {
                "explained_cells_features": shap_results["explained_cells_features"],
                "explained_cells_indices": shap_results["explained_cells_indices"],
                "sampled_indices": sampled_indices,
                "background_data": shap_results["background_data"],
                "background_indices": shap_results["background_indices"],
                "num_heads": shap_results["num_heads"],
            }

            # Add per-head SHAP values
            for head_name, shap_vals in shap_results["shap_values_per_head"].items():
                save_dict[f"shap_values_{head_name}"] = shap_vals
                save_dict[f"feature_importance_{head_name}"] = shap_results[
                    "feature_importance_per_head"
                ][head_name]
                save_dict[f"top_features_{head_name}"] = shap_results[
                    "top_features_per_head"
                ][head_name]

            np.savez(shap_file, **save_dict)
            saved_files["data"].append(shap_file)
            logger.info(f"Saved SHAP values to {shap_file}")

        # Save feature importance for each head
        for head_name, feature_importance in shap_results[
            "feature_importance_per_head"
        ].items():
            importance_file = (
                self.config.output_path / f"feature_importance_{head_name}.csv"
            )
            importance_df = pd.DataFrame(
                {
                    "feature_idx": np.arange(len(feature_importance)),
                    "importance": feature_importance,
                }
            )
            importance_df = importance_df.sort_values("importance", ascending=False)  # type: ignore
            importance_df.to_csv(importance_file, index=False)
            saved_files["data"].append(importance_file)
            logger.info(
                f"Saved feature importance for {head_name} to {importance_file}"
            )

        # Save attention score distribution
        attention_file = self.config.output_path / "attention_distribution.npz"
        np.savez(
            attention_file,
            attention_scores=attention_scores,
            sampled_indices=sampled_indices,
        )
        saved_files["data"].append(attention_file)

        # Save metadata
        metadata: dict[str, Any] = {
            "total_cells": cell_data["features"].shape[0],
            "sampled_cells": len(sampled_indices),
            "num_features": cell_data["features"].shape[1],
            "num_heads": shap_results["num_heads"],
            "heads_explained": list(shap_results["shap_values_per_head"].keys()),
            "sampling_info": sampling_info,
            "attention_stats": {
                "min": float(attention_scores.min()),
                "max": float(attention_scores.max()),
                "mean": float(attention_scores.mean()),
                "std": float(attention_scores.std()),
            },
            "config": self.config.dict(),  # type: ignore
        }

        metadata_file = self.config.output_path / "shap_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=4, default=str)
        saved_files["metadata"].append(metadata_file)
        logger.info(f"Saved metadata to {metadata_file}")

        # Create summary plots using visualizer
        if self.config.create_summary_plots:
            try:
                plots_dir = self.config.output_path / "plots"

                # Create plots for each head
                for head_name in shap_results["shap_values_per_head"].keys():
                    head_plots_dir = plots_dir / head_name
                    head_plots_dir.mkdir(parents=True, exist_ok=True)

                    logger.info(f"Creating visualizations for {head_name}...")

                    plot_files = self.visualizer.create_visualizations(
                        shap_values=shap_results["shap_values_per_head"][head_name],
                        sampled_features=shap_results["explained_cells_features"],
                        feature_importance=shap_results["feature_importance_per_head"][
                            head_name
                        ],
                        top_features_idx=shap_results["top_features_per_head"][
                            head_name
                        ][: self.config.top_features],
                        output_dir=head_plots_dir,
                        feature_names=cell_data.get("feature_names"),
                    )
                    saved_files["plots"].extend(plot_files)

            except Exception as e:
                logger.error(f"Error creating visualizations: {e}")

        return saved_files

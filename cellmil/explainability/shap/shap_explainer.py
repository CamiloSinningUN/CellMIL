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
from tqdm import tqdm

from cellmil.interfaces.SHAPExplainerConfig import SHAPExplainerConfig
from cellmil.interfaces.FeatureExtractorConfig import ExtractorType
from cellmil.interfaces.CellSegmenterConfig import ModelType
from cellmil.interfaces.GraphCreatorConfig import GraphCreatorType
from cellmil.datamodels.transforms import TransformPipeline
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
        model: LitAttentionDeepMIL | LitHead4Type,
        dataset_folder: Path | str,
        data: pd.DataFrame,
        extractor: ExtractorType | list[ExtractorType],
        segmentation_model: ModelType,
        graph_creator: Optional[GraphCreatorType] = None,
        transforms_path: Path | str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Generate SHAP explanations for the given model and dataset.

        Args:
            model: The MIL model to explain (AttentionDeepMIL or Head4Type)
            dataset_folder: Path to the folder containing all slide data
            data: DataFrame with slide metadata (must have 'FULL_PATH' column)
            extractor: Feature extractor type used for the slides
            segmentation_model: Segmentation model used for cell detection
            transforms_path: Path to the transform pipeline file
            graph_creator: Graph creation method (if needed for features)
            **kwargs: Additional arguments

        Returns:
            Dictionary containing SHAP values, explanations, and metadata
        """
        logger.info("Starting SHAP explanation generation")

        dataset_folder = Path(dataset_folder)
        transforms_path = Path(transforms_path)

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

        # Load transforms
        if transforms_path.exists():
            transforms = TransformPipeline.load(transforms_path)
        else:
            raise FileNotFoundError(f"Transforms file not found: {transforms_path}")

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
            transforms=transforms,
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
        supported_models = (LitAttentionDeepMIL, LitHead4Type)
        if not isinstance(model, supported_models):
            raise ValueError(
                f"Unsupported model type: {model.__class__.__name__}. "
                f"Supported types: {[cls.__name__ for cls in supported_models]}"
            )

        if not hasattr(model, "get_attention_weights"):
            raise ValueError(
                f"Model {model.__class__.__name__} must have get_attention_weights method"
            )

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
        slide_names = data["FULL_PATH"].tolist()
        logger.info(f"Processing {len(slide_names)} slides...")

        feature_names: Optional[List[str]] = None

        for slide_name in tqdm(slide_names, desc="Loading slides"):
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
                    continue

                # Store feature names from first slide (should be same for all)
                if feature_names is None and slide_feature_names is not None:
                    feature_names = slide_feature_names
                    logger.info(f"Captured {len(feature_names)} feature names")

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

                # Add to collections
                all_features.append(features)
                if cell_types is not None:
                    all_cell_types.append(cell_types)

                # Track which slide each cell belongs to
                num_cells = features.shape[0]
                slide_indices.extend([(slide_name, i) for i in range(num_cells)])

            except Exception as e:
                logger.error(f"Error processing slide {slide_name}: {e}")
                continue

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

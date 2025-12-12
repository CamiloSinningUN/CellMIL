"""
Main explainability interface for MIL models.

This module provides a unified entry point for all explainability methods
in the CellMIL framework. Currently supports attention-based explanations
with plans for future expansion to other methods.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple
from torch_geometric.data import Data  # type: ignore
import torch
import lightning as Pl

from cellmil.interfaces.ExplainerCreatorConfig import (
    ExplainerCreatorConfig,
    ExplainMethod,
)
from cellmil.datamodels.transforms import TransformPipeline
from cellmil.datamodels.datasets.utils import (
    get_cell_features,
    get_centroids,
    load_precomputed_graph,
    merge_graph_with_features,
    centroids_to_tensor,
)
from cellmil.interfaces.FeatureExtractorConfig import (
    ExtractorType,
    FeatureExtractionType,
)
from cellmil.interfaces.CellSegmenterConfig import ModelType
from cellmil.interfaces.GraphCreatorConfig import GraphCreatorType
from .explainer import Attention
from cellmil.utils import logger
from cellmil.models.mil import LitAttentionDeepMIL, LitCLAM, LitGraphMIL


class Explain:
    """
    Main explainability interface for MIL models.

    This class provides a unified interface for generating explanations
    across different MIL model types and explanation methods.
    """

    def __init__(self, config: ExplainerCreatorConfig):
        """
        Initialize the explainer with the given configuration.

        Args:
            config: Configuration specifying explanation method and parameters
        """
        self.config = config

        # Initialize the appropriate explainer based on method
        if self.config.method == ExplainMethod.attention:
            self.explainer = Attention(config)
        else:
            raise ValueError(f"Unsupported explanation method: {self.config.method}")

        logger.info(f"Initialized {self.config.method} explainer")

    def generate_explanation(
        self,
        model: Pl.LightningModule,
        slide_path: Path | str,
        extractor: ExtractorType | list[ExtractorType],
        segmentation_model: ModelType,
        transforms_path: Path | str,
        graph_creator: Optional[GraphCreatorType] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Generate explanations for the given model and slide data.

        Args:
            model: The MIL model to explain (CLAM, AttentionDeepMIL, GraphMIL, etc.)
            slide_path: Path to the slide directory containing patches, cell_detection, features, etc.
            extractor: Feature extractor type used for the slide
            segmentation_model: Segmentation model used for cell detection
            graph_creator: Graph creation method for GraphMIL models and Graph Features
            transforms: Transform pipeline to apply to features
            **kwargs: Additional method-specific arguments

        Returns:
            Dictionary containing explanation results, file paths, and metadata

        Raises:
            ValueError: If model_type is not supported or required data is missing
            FileNotFoundError: If required data files are missing
        """
        logger.info("Starting explanation generation process")

        slide_path = Path(slide_path)
        transforms_path = Path(transforms_path)

        if transforms_path.exists():
            transforms = TransformPipeline.load(transforms_path)
        else:
            raise FileNotFoundError(f"Transforms file not found: {transforms_path}")

        logger.info(
            f"Generating explanation for {model.__class__.__name__} model using {self.config.method}"
        )

        # Load data from slide path
        data, cell_data_path, cell_indices, cell_coordinates = self._load_slide_data(
            slide_path=slide_path,
            model=model,
            extractor=extractor,
            segmentation_model=segmentation_model,
            graph_creator=graph_creator,
            transforms=transforms,
        )

        # Validate inputs
        self._validate_inputs(model, cell_data_path, cell_indices)

        result = self.explainer.explain(
            model=model,
            data=data,
            cell_data_path=cell_data_path,
            cell_indices=cell_indices,
            cell_coordinates=cell_coordinates,
            **kwargs,
        )

        logger.info("Explanation generation completed successfully")

        return result

    def _load_slide_data(
        self,
        slide_path: Path,
        model: Pl.LightningModule,
        extractor: ExtractorType | list[ExtractorType],
        segmentation_model: ModelType,
        transforms: TransformPipeline,
        graph_creator: Optional[GraphCreatorType] = None,
    ) -> Tuple[
        Union[torch.Tensor, Data],
        Path,
        Dict[int, int],
        Dict[int, tuple[float, float]],
    ]:
        """
        Load data from the slide path for explanation.

        Args:
            slide_path: Path to the slide directory
            model: MIL model to explain
            extractor: Feature extractor to use
            segmentation_model: Segmentation model to use
            graph_creator: Graph creator for GraphMIL models
            transforms: Transform pipeline to apply

        Returns:
            Tuple of (data, cell_data_path, cell_indices, cell_coordinates)
        """
        slide_name = slide_path.name
        logger.info(f"Loading data for slide: {slide_name}")

        graph_creator_needed = False
        # Check if any extractor is in the topological feature extraction types
        if isinstance(extractor, list):
            if any(ext in FeatureExtractionType.Topological for ext in extractor):
                graph_creator_needed = True
        else:
            graph_creator_needed = extractor in FeatureExtractionType.Topological

        if (
            isinstance(model, LitGraphMIL) or graph_creator_needed
        ) and not graph_creator:
            logger.error("Graph creator required but not provided")
            raise ValueError(
                "Graph creator must be specified for GraphMIL models or topological features"
            )

        # Load features and related data
        features, cell_indices, _ = get_cell_features(
            folder=slide_path.parent,
            slide_name=slide_name,
            extractor=extractor,  # type: ignore
            graph_creator=graph_creator,  # type: ignore
            segmentation_model=segmentation_model,  # type: ignore
        )

        if features is None:
            logger.error(f"Failed to load features for slide {slide_name}")
            raise ValueError(f"Could not load features for slide {slide_name}")

        if cell_indices is None:
            raise ValueError("cell_indices could not be loaded")

        logger.info(f"Features loaded with shape: {features.shape}")

        # Apply transforms if provided
        if transforms:
            original_shape = features.shape
            features = transforms.transform(features)
            logger.info(f"Transforms applied: {original_shape} -> {features.shape}")

        logger.info("Loading cell data path...")
        # Get cell data path

        cell_data_path = (
            slide_path / "cell_detection" / str(segmentation_model) / "cells.json"
        )
        if not cell_data_path.exists():
            raise FileNotFoundError(f"Cell data file not found: {cell_data_path}")
        else:
            logger.info(f"Cell data path found: {cell_data_path}")

        # Prepare data based on model type
        if isinstance(model, LitGraphMIL):
            logger.info("Processing GraphMIL model data...")

            if graph_creator is None:
                logger.error("graph_creator is required for GraphMIL but not provided")
                raise ValueError("graph_creator is required for GraphMIL models")

            logger.info("Loading precomputed graph...")
            precomputed_graph = load_precomputed_graph(
                folder=slide_path.parent,
                slide_name=slide_name,
                graph_creator=graph_creator,
                segmentation_model=segmentation_model,
            )

            logger.info(
                f"Graph loaded with {precomputed_graph.num_nodes} nodes and {precomputed_graph.num_edges} edges"
            )

            logger.info("Loading centroids...")
            # Get centroids and convert to tensor like in the dataset
            cell_coordinates = get_centroids(
                folder=slide_path.parent,
                slide_name=slide_name,
                segmentation_model=segmentation_model,
            )

            logger.info("Converting centroids to tensor...")
            # Convert centroids to tensor using the same logic as the dataset
            if cell_coordinates is None or len(cell_coordinates) == 0:
                raise ValueError("Cell coordinates could not be loaded or are empty")
            else:
                logger.info(f"Found {len(cell_coordinates)} centroids")
                centroids_tensor = centroids_to_tensor(cell_coordinates, cell_indices)
                # Normalize centroids to [0, 1] per slide like in the dataset
                if (
                    centroids_tensor.shape[0] > 1
                ):  # Only normalize if we have multiple points
                    min_vals = centroids_tensor.min(dim=0).values
                    max_vals = centroids_tensor.max(dim=0).values
                    denom = max_vals - min_vals
                    denom[denom == 0] = 1.0  # Prevent division by zero
                    centroids_tensor = (centroids_tensor - min_vals) / denom
                    logger.info("Centroids normalized to [0,1]")

            # Properly align features with graph nodes using cell IDs
            logger.info("Merging graph with features...")
            data = merge_graph_with_features(
                graph_data=precomputed_graph,
                features=features,
                cell_indices=cell_indices,
                cell_coordinates=centroids_tensor,
            )

            if hasattr(data, "x") and data.x is not None:
                logger.info(
                    f"Merged data has {data.num_nodes} nodes and {data.x.shape[1]} features"
                )

        else:
            logger.info("Processing standard MIL model data...")
            # Standard MIL models use tensor input
            data = features
            logger.info(f"Using features tensor directly with shape: {data.shape}")

            # Get cell coordinates for non-GraphMIL models
            cell_coordinates = get_centroids(
                folder=slide_path.parent,
                slide_name=slide_name,
                segmentation_model=segmentation_model,
            )

            if cell_coordinates is None:
                raise ValueError("Cell coordinates could not be loaded")
            else:
                logger.info(f"Loaded {len(cell_coordinates)} cell coordinates")

        cell_coordinates = {
            cell_indices[cell_id]: (x, y)
            for cell_id, (x, y) in cell_coordinates.items()
            if cell_id in cell_indices
        }

        logger.info("Data loading completed successfully")
        
        return data, cell_data_path, cell_indices, cell_coordinates

    def _validate_inputs(
        self,
        model: Pl.LightningModule,
        cell_data_path: Optional[Path],
        cell_indices: Optional[Dict[int, int]],
    ) -> None:
        """Validate input parameters."""

        # Check model type
        supported_model_classes = (LitCLAM, LitAttentionDeepMIL, LitGraphMIL)
        if not isinstance(model, supported_model_classes):
            raise ValueError(
                f"Unsupported model type: {model.__class__.__name__}. "
                f"Supported types: {[cls.__name__ for cls in supported_model_classes]}"
            )

        # Check if model has required attention methods
        if not hasattr(model, "get_attention_weights"):
            logger.warning(
                f"Model of type {model.__class__.__name__} does not have get_attention_weights method. "
                "Some explanation features may not work."
            )

        # Validate spatial visualization requirements
        if cell_data_path and not cell_data_path.exists():
            raise FileNotFoundError(f"Cell data file not found: {cell_data_path}")

        if cell_data_path and not cell_indices:
            logger.warning(
                "Cell data path provided but cell_indices missing. "
                "Visualization may not work correctly."
            )

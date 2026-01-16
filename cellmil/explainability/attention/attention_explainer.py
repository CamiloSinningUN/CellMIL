from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple, List, Literal
from torch_geometric.data import Data  # type: ignore
import torch
import lightning as Pl
import json
import re

from cellmil.interfaces.AttentionExplainerConfig import (
    AttentionExplainerConfig,
    VisualizationMode,
)
from cellmil.datamodels.transforms import TransformPipeline
from cellmil.datamodels.model import ModelStorage
from cellmil.datamodels.datasets.utils import (
    get_cell_features,
    get_centroids,
    load_precomputed_graph,
    merge_graph_with_features,
    centroids_to_tensor,
    get_cell_types,
    cell_types_to_tensor,
)
from cellmil.interfaces.FeatureExtractorConfig import (
    ExtractorType,
    FeatureExtractionType,
)
from cellmil.interfaces.CellSegmenterConfig import ModelType
from cellmil.interfaces.GraphCreatorConfig import GraphCreatorType
from cellmil.explainability.attention.core.extractor import (
    AttentionExtractorFactory,
    AttentionResult,
)
from cellmil.explainability.attention.visualizers.geojson import (
    AttentionGeoJSONVisualizer,
)
from cellmil.explainability.attention.visualizers.graph import (
    AttentionGraphVisualizer,
)
from cellmil.utils import logger
from cellmil.models.mil.graphmil import LitGraphMIL, LitSurvGraphMIL
from cellmil.models.mil.clam import LitCLAM, LitSurvCLAM
from cellmil.models.mil.attentiondeepmil import LitAttentionDeepMIL, LitSurvAttentionDeepMIL
from cellmil.models.mil.head4type import LitHead4Type, LitSurvHead4Type

# Model class registry for loading from checkpoint
MODEL_CLASS_REGISTRY = {
    "LitAttentionDeepMIL": LitAttentionDeepMIL,
    "LitCLAM": LitCLAM,
    "LitGraphMIL": LitGraphMIL,
    "LitHead4Type": LitHead4Type,
    "LitSurvAttentionDeepMIL": LitSurvAttentionDeepMIL,
    "LitSurvCLAM": LitSurvCLAM,
    "LitSurvGraphMIL": LitSurvGraphMIL,
    "LitSurvHead4Type": LitSurvHead4Type,
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


class AttentionExplainer:
    """
    Attention-based explainability for MIL models.

    This class provides comprehensive attention explanation capabilities for
    multiple MIL model types including CLAM, AttentionDeepMIL, GraphMIL, and Head4Type.
    It extracts attention weights, creates visualizations, and saves metadata.
    """

    def __init__(self, config: AttentionExplainerConfig):
        """
        Initialize the attention explainer.

        Args:
            config: Configuration for attention explanation process
        """
        self.config = config
        logger.info(
            f"Initialized Attention explainer with mode: {config.visualization_mode}"
        )

    def generate_explanation(
        self,
        model_storage: ModelStorage,
        slide_path: Path | str,
        fold_idx: Optional[Union[int, Literal["all"]]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Generate attention explanations for the given model and slide data.

        Args:
            model_storage: ModelStorage instance containing model checkpoint, transforms, and configuration
            slide_path: Path to the slide directory containing patches, cell_detection, features, etc.
            fold_idx: Optional fold index to use. If None, uses the final model. 
                     If -1 or "all", processes all folds and the final model.
                     Otherwise, must be in range [0, k_folds-1]
            **kwargs: Additional method-specific arguments

        Returns:
            Dictionary containing explanation results, file paths, and metadata.
            When fold_idx is -1 or "all", returns a dict with keys for each fold and 'final_model'

        Raises:
            ValueError: If model_type is not supported, required data is missing, or fold_idx is invalid
            FileNotFoundError: If required data files are missing
        """
        # Handle "all" case - process all folds and final model
        if fold_idx == -1 or fold_idx == "all":
            logger.info("Processing all folds and final model")
            all_results = {}
            
            # Get available folds
            available_folds = model_storage.list_folds()
            logger.info(f"Found {len(available_folds)} folds: {available_folds}")
            
            # Store original output path
            original_output_path = self.config.output_path
            
            # Process each fold
            for fold in available_folds:
                logger.info(f"Processing fold {fold}...")
                # Create fold-specific output directory
                self.config.output_path = original_output_path / f"fold_{fold}"
                try:
                    fold_result = self.generate_explanation(
                        model_storage=model_storage,
                        slide_path=slide_path,
                        fold_idx=fold,
                        **kwargs
                    )
                    all_results[f"fold_{fold}"] = fold_result
                    logger.info(f"Fold {fold} completed successfully")
                except Exception as e:
                    logger.error(f"Error processing fold {fold}: {e}")
                    all_results[f"fold_{fold}"] = {"error": str(e)}
            
            # Process final model if it exists
            if model_storage.has_final_model():
                logger.info("Processing final model...")
                self.config.output_path = original_output_path / "final_model"
                try:
                    final_result = self.generate_explanation(
                        model_storage=model_storage,
                        slide_path=slide_path,
                        fold_idx=None,
                        **kwargs
                    )
                    all_results["final_model"] = final_result
                    logger.info("Final model completed successfully")
                except Exception as e:
                    logger.error(f"Error processing final model: {e}")
                    all_results["final_model"] = {"error": str(e)}
            
            # Restore original output path
            self.config.output_path = original_output_path
            
            logger.info(f"Completed processing all models. Total results: {len(all_results)}")
            return all_results
        
        logger.info("Starting attention explanation generation process")

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
        model = model_class.load_from_checkpoint(str(checkpoint_path), map_location=torch.device("cpu") if not torch.cuda.is_available() else None)
        model.eval()

        # Use the fold transforms for feature transformation
        transforms_to_use = transforms if transforms is not None else None

        slide_path = Path(slide_path)

        # Prepare transforms path - store transforms for later use
        if transforms_to_use is not None:
            transforms_path = (
                model_storage.output_dir
                / ("fold_" + str(fold_idx) if fold_idx is not None else "final_model")
                / "transforms"
            )
        else:
            raise ValueError("No transforms found in model checkpoint")

        logger.info(f"Generating explanation for {model.__class__.__name__} model")
        logger.info(f"Visualization mode: {self.config.visualization_mode}")
        logger.info(f"Output directory: {self.config.output_path}")

        # Create output directory
        self.config.output_path.mkdir(parents=True, exist_ok=True)

        # Load data from slide path
        data, cell_data_path, cell_indices, cell_coordinates, cell_types = (
            self._load_slide_data(
                slide_path=slide_path,
                model=model,
                extractor=extractor,
                segmentation_model=segmentation_model,
                graph_creator=graph_creator,
                transforms=transforms_to_use,
            )
        )

        # Validate inputs
        self._validate_inputs(model, cell_data_path, cell_indices)

        # Log data information
        if isinstance(data, torch.Tensor):
            logger.info(f"Input data: Tensor with shape {data.shape}")
        else:
            logger.info(
                f"Input data: Graph with {data.num_nodes} nodes, {data.num_edges} edges"
            )
            if hasattr(data, "x") and data.x is not None:
                logger.info(f"Node features shape: {data.x.shape}")

        if cell_indices:
            logger.info(f"Number of cell indices: {len(cell_indices)}")
        if cell_coordinates:
            logger.info(f"Number of cell coordinates: {len(cell_coordinates)}")

        # Convert cell types to tensor if available
        cell_types_tensor = None
        if cell_types is not None:
            logger.info("Converting cell types to tensor...")
            cell_types_tensor = cell_types_to_tensor(cell_types, cell_indices)

            # Log cell type statistics
            cell_type_counts = cell_types_tensor.sum(dim=0)
            for idx, count in enumerate(cell_type_counts):
                logger.info(f"Cell type {idx}: {int(count)} cells")
            logger.info(f"Cell types tensor shape: {cell_types_tensor.shape}")

        # Extract attention weights
        attention_result = self._extract_attention(model, data, cell_types_tensor)
        logger.info(
            f"Extracted attention types: {list(attention_result.attention_weights.keys())}"
        )

        # Create visualizations
        visualization_files = self._create_visualizations(
            attention_result,
            data,
            cell_data_path,
            cell_indices,
            cell_coordinates,
            cell_types_tensor=cell_types_tensor,
            **kwargs,
        )

        # Save attention data and metadata
        metadata_files = self._save_attention_metadata(attention_result)
        logger.info(f"Metadata saved to {len(metadata_files)} files")

        # Compile results
        results: dict[str, Any] = {
            "model": model.__class__.__name__,
            "attention_result": attention_result,
            "visualization_files": visualization_files,
            "metadata_files": metadata_files,
            "config": self.config.dict(),  # type: ignore
            "summary": self._generate_summary(attention_result),
        }

        logger.info("Attention explanation completed successfully")
        logger.info(f"Results saved to: {self.config.output_path}")
        return results

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
        Optional[Dict[int, int]],
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
            Tuple of (data, cell_data_path, cell_indices, cell_coordinates, cell_types)
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

        # Load cell types for Head4Type models
        cell_types = None
        if isinstance(model, LitHead4Type):
            logger.info("Loading cell types for Head4Type model...")

            cell_types = get_cell_types(
                folder=slide_path.parent,
                slide_name=slide_name,
                segmentation_model=segmentation_model,
            )

            if cell_types is None:
                raise ValueError(f"Could not load cell types for slide {slide_name}")

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

        return data, cell_data_path, cell_indices, cell_coordinates, cell_types

    def _extract_attention(
        self,
        model: Pl.LightningModule,
        data: Union[torch.Tensor, Data],
        cell_types_tensor: Optional[torch.Tensor] = None,
    ) -> AttentionResult:
        """Extract attention weights using appropriate extractor."""

        logger.info(
            f"Creating attention extractor for model type: {model.__class__.__name__}"
        )
        try:
            extractor = AttentionExtractorFactory.create_extractor(model, self.config)
            logger.info(f"Extractor created: {extractor.__class__.__name__}")

            logger.info("Extracting attention weights from model...")
            # Pass cell_types_tensor for models that need it (e.g., Head4Type)
            if cell_types_tensor is not None:
                result = extractor.extract(
                    model, data, cell_types_tensor=cell_types_tensor
                )
            else:
                result = extractor.extract(model, data)
            logger.info(
                f"Attention extraction successful - found {len(result.attention_weights)} attention types"
            )

            # Log attention weight statistics
            for key, weights in result.attention_weights.items():
                logger.info(
                    f"  - {key}: shape {weights.shape}, mean={weights.mean():.4f}, std={weights.std():.4f}"
                )

            return result
        except Exception as e:
            logger.error(
                f"Error extracting attention from {model.__class__.__name__}: {e}"
            )
            raise

    def _create_visualizations(
        self,
        attention_result: AttentionResult,
        data: torch.Tensor | Data,
        cell_data_path: Path,
        cell_indices: Dict[int, int],
        cell_coordinates: Dict[int, tuple[float, float]],
        **kwargs: Any,
    ) -> Dict[str, List[Path]]:
        """Create visualizations based on configuration."""

        logger.info("Creating visualizations based on configuration...")
        logger.info(f"Visualization mode: {self.config.visualization_mode}")

        # Extract cell_types_tensor from kwargs if provided
        cell_types_tensor = kwargs.get("cell_types_tensor", None)

        visualization_files: dict[str, List[Path]] = {}

        # ---- GeoJSON Visualizations ----

        if self.config.visualization_mode in [
            VisualizationMode.geojson,
            VisualizationMode.all,
        ]:
            try:
                logger.info("Creating GeoJSON visualizations...")
                geojson_visualizer = AttentionGeoJSONVisualizer(self.config)
                geojson_files = geojson_visualizer.create_visualization(
                    attention_result,
                    cell_data_path,
                    cell_indices,
                    self.config.output_path / "geojson",
                )
                visualization_files["geojson"] = geojson_files
                logger.info(f"Created {len(geojson_files)} GeoJSON files")

            except Exception as e:
                logger.error(f"Error creating GeoJSON visualizations: {e}")

        # ---- Graph Visualizations ----

        if self.config.visualization_mode in [
            VisualizationMode.graph,
            VisualizationMode.all,
        ]:
            try:
                # Convert tensor to Data object if necessary
                graph_data = data
                if isinstance(data, torch.Tensor):
                    logger.info(
                        "Converting tensor input to graph format (nodes only, no edges)"
                    )
                    # Create a Data object with nodes but no edges
                    node_features = data.squeeze(0) if data.dim() > 2 else data
                    graph_data = Data(x=node_features)
                    # Create empty edge_index for no edges
                    graph_data.edge_index = torch.empty((2, 0), dtype=torch.long)
                    graph_data.num_nodes = graph_data.x.shape[0]  # type: ignore
                    logger.info(
                        f"Created graph with {graph_data.num_nodes} nodes and 0 edges"
                    )
                else:
                    graph_data = data

                logger.info("Creating graph visualizations...")
                graph_visualizer = AttentionGraphVisualizer(self.config)
                graph_files = graph_visualizer.create_visualization(
                    attention_result,
                    graph_data,
                    self.config.output_path / "graphs",
                    cell_coordinates,
                    cell_types_tensor,
                )
                visualization_files["graphs"] = graph_files
                logger.info(f"Created {len(graph_files)} graph visualization files")

            except Exception as e:
                logger.error(f"Error creating graph visualizations: {e}")

        total_files = sum(len(files) for files in visualization_files.values())
        logger.info(
            f"Visualization creation completed - Total files created: {total_files}"
        )

        return visualization_files

    def _save_attention_metadata(self, attention_result: AttentionResult) -> List[Path]:
        """Save attention weights and metadata to files."""

        saved_files: list[Path] = []

        logger.info("Processing attention weights for saving...")
        # Save raw attention weights
        attention_data: dict[str, Any] = {}
        for key, weights in attention_result.attention_weights.items():
            logger.info(f"Processing attention weights for: {key}")
            logger.info(f"  Shape: {weights.shape}")
            logger.info(f"  Mean: {weights.mean():.4f}, Std: {weights.std():.4f}")

            attention_data[key] = {
                "shape": list(weights.shape),
                "data": weights.cpu().detach().numpy().tolist(),  # type: ignore
                "statistics": {
                    "mean": float(weights.mean()),
                    "std": float(weights.std()),
                    "min": float(weights.min()),
                    "max": float(weights.max()),
                    "sum": float(weights.sum()),
                },
            }

        logger.info("Saving attention weights to JSON...")
        attention_file = self.config.output_path / "attention_weights.json"
        with open(attention_file, "w") as f:
            json.dump(attention_data, f, indent=4)
        saved_files.append(attention_file)
        logger.info(f"Saved attention weights: {attention_file}")

        logger.info("Preparing explanation metadata...")
        # Save metadata
        metadata: dict[str, Any] = {
            "model": attention_result.model_type,
            "metadata": attention_result.metadata,
            "attention_keys": attention_result.get_all_keys(),
            "shape_info": attention_result.get_shape_info(),
            "config": self.config.dict(),  # type: ignore
        }

        logger.info("Saving explanation metadata to JSON...")
        metadata_file = self.config.output_path / "explanation_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=4, default=str)
        saved_files.append(metadata_file)
        logger.info(f"Saved explanation metadata: {metadata_file}")

        logger.info(f"Metadata saving completed - {len(saved_files)} files created")
        return saved_files

    def _generate_summary(self, attention_result: AttentionResult) -> Dict[str, Any]:
        """Generate a summary of the attention explanation."""

        logger.info("Generating explanation summary...")

        total_parameters = sum(
            weights.numel() for weights in attention_result.attention_weights.values()
        )

        logger.info(
            f"Summary stats - Model: {attention_result.model_type}, Attention types: {len(attention_result.attention_weights)}, Total parameters: {total_parameters}"
        )

        summary: dict[str, Any] = {
            "model": attention_result.model_type,
            "num_attention_types": len(attention_result.attention_weights),
            "attention_types": list(attention_result.attention_weights.keys()),
            "total_parameters": total_parameters,
            "shapes": attention_result.get_shape_info(),
            "statistics": {},
        }

        logger.info("Computing detailed statistics for each attention type...")
        # Compute statistics for each attention type
        for key, weights in attention_result.attention_weights.items():
            logger.info(f"Computing statistics for: {key}")
            weights_flat = weights.flatten()

            entropy = float(-torch.sum(weights_flat * torch.log(weights_flat + 1e-8)))
            sparsity = float((weights_flat == 0).float().mean())
            concentration = float(weights_flat.max() / (weights_flat.sum() + 1e-8))

            summary["statistics"][key] = {
                "mean": float(weights_flat.mean()),
                "std": float(weights_flat.std()),
                "entropy": entropy,
                "sparsity": sparsity,
                "concentration": concentration,
            }

            logger.info(
                f"  Statistics - Mean: {weights_flat.mean():.4f}, Std: {weights_flat.std():.4f}, Entropy: {entropy:.4f}, Sparsity: {sparsity:.4f}"
            )

        logger.info("Summary generation completed")
        return summary

    def _validate_inputs(
        self,
        model: Pl.LightningModule,
        cell_data_path: Optional[Path],
        cell_indices: Optional[Dict[int, int]],
    ) -> None:
        """Validate input parameters."""

        # Check model type
        supported_model_classes = (
            LitCLAM,
            LitAttentionDeepMIL,
            LitGraphMIL,
            LitHead4Type,
        )
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

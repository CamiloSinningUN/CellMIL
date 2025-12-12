"""
Advanced Attention Explainer for MIL Models.

This module provides comprehensive attention explanation capabilities for
multiple MIL model types including CLAM, AttentionDeepMIL, and GraphMIL.
"""

import torch
from pathlib import Path
from typing import Dict, Any, List, Union
import json
import lightning as Pl

from cellmil.interfaces.ExplainerCreatorConfig import (
    ExplainerCreatorConfig,
    VisualizationMode,
)
from cellmil.explainability.core.attention_extractor import (
    AttentionExtractorFactory,
    AttentionResult,
)
from cellmil.explainability.visualizers.geojson import AttentionGeoJSONVisualizer
from cellmil.explainability.visualizers.heatmap import AttentionHeatmapVisualizer
from cellmil.explainability.visualizers.graph import AttentionGraphVisualizer
from cellmil.utils import logger
from torch_geometric.data import Data  # type: ignore


class Attention:
    """
    Advanced attention explainer for MIL models.

    This class provides a unified interface for explaining attention mechanisms
    across different MIL model architectures. It supports multiple visualization
    modes and can handle complex attention patterns.
    """

    def __init__(self, config: ExplainerCreatorConfig):
        """
        Initialize the attention explainer.

        Args:
            config: Configuration for explanation process
        """
        self.config = config
        logger.info(
            f"Initialized Attention explainer with mode: {config.visualization_mode}"
        )

    def explain(
        self,
        model: Pl.LightningModule,
        data: torch.Tensor | Data,
        cell_data_path: Path,
        cell_indices: Dict[int, int],
        cell_coordinates: Dict[int, tuple[float, float]],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Generate attention explanations for the given model and data.

        Args:
            model: The MIL model to explain
            data: Input data (tensor for standard models, Data object for GraphMIL)
            cell_data_path: Path to cell detection JSON file (for spatial visualization)
            cell_indices: Mapping from cell_id to tensor index
            cell_coordinates: Node coordinates for graph visualization
            **kwargs: Additional arguments specific to model types

        Returns:
            Dictionary containing explanation results and file paths
        """
        logger.info("Starting attention explanation process")
        logger.info(f"Model type: {model.__class__.__name__}")
        logger.info(f"Visualization mode: {self.config.visualization_mode}")
        logger.info(f"Output directory: {self.config.output_path}")

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

        # Create output directory
        self.config.output_path.mkdir(parents=True, exist_ok=True)

        # Extract attention weights
        attention_result = self._extract_attention(model, data)
        logger.info(
            f"Extracted attention types: {list(attention_result.attention_weights.keys())}"
        )

        visualization_files = self._create_visualizations(
            attention_result,
            data,
            cell_data_path,
            cell_indices,
            cell_coordinates,
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

    def _extract_attention(
        self, model: Any, data: Union[torch.Tensor, Any]
    ) -> AttentionResult:
        """Extract attention weights using appropriate extractor."""

        logger.info(
            f"Creating attention extractor for model type: {model.__class__.__name__}"
        )
        try:
            extractor = AttentionExtractorFactory.create_extractor(model, self.config)
            logger.info(f"Extractor created: {extractor.__class__.__name__}")

            logger.info("Extracting attention weights from model...")
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
                    logger.info("Converting tensor input to graph format (nodes only, no edges)")
                    # Create a Data object with nodes but no edges
                    node_features = data.squeeze(0) if data.dim() > 2 else data
                    graph_data = Data(x=node_features)
                    # Create empty edge_index for no edges
                    graph_data.edge_index = torch.empty((2, 0), dtype=torch.long)
                    graph_data.num_nodes = graph_data.x.shape[0] # type: ignore
                    logger.info(f"Created graph with {graph_data.num_nodes} nodes and 0 edges")
                else:
                    graph_data = data

                logger.info("Creating graph visualizations...")
                graph_visualizer = AttentionGraphVisualizer(self.config)
                graph_files = graph_visualizer.create_visualization(
                    attention_result,
                    graph_data,
                    self.config.output_path / "graphs",
                    cell_coordinates,
                )
                visualization_files["graphs"] = graph_files
                logger.info(f"Created {len(graph_files)} graph visualization files")

            except Exception as e:
                logger.error(f"Error creating graph visualizations: {e}")

        # ---- Heatmap Visualizations ----

        if self.config.visualization_mode in [
            VisualizationMode.heatmap,
            VisualizationMode.all,
        ]:
            try:
                logger.info("Creating heatmap visualizations...")
                heatmap_visualizer = AttentionHeatmapVisualizer(self.config)
                heatmap_files = heatmap_visualizer.create_visualization(
                    attention_result,
                    self.config.output_path / "heatmaps"
                )
                visualization_files["heatmaps"] = heatmap_files
                logger.info(f"Created {len(heatmap_files)} heatmap files")

            except Exception as e:
                logger.error(f"Error creating heatmap visualizations: {e}")

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

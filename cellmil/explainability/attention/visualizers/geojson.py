"""
GeoJSON visualization for attention weights.

This module creates GeoJSON outputs compatible with pathology viewers like QuPath.
It handles attention binning, color mapping, and spatial visualization.
"""

import json
import torch
import numpy as np
import ujson
from pathlib import Path
from typing import Dict, Any, List, Optional, cast

from cellmil.interfaces.AttentionExplainerConfig import AttentionExplainerConfig
from cellmil.explainability.attention.core.extractor import AttentionResult
from cellmil.utils import logger
from cellmil.utils.templates import get_template_segmentation


class AttentionGeoJSONVisualizer:
    """Creates GeoJSON visualizations for attention weights."""

    def __init__(self, config: AttentionExplainerConfig):
        self.config = config
        self.alpha = 128  # Transparency value (0-255), 128 = 50% transparent

    def create_visualization(
        self,
        attention_result: AttentionResult,
        cell_data_path: Path,
        cell_indices: Dict[int, int],
        output_path: Path,
    ) -> List[Path]:
        """
        Create GeoJSON visualizations for all attention types in the result.

        Args:
            attention_result: AttentionResult containing attention weights
            cell_data_path: Path to cell detection JSON file
            cell_indices: Mapping from cell_id to tensor index
            output_path: Directory to save GeoJSON files

        Returns:
            List of paths to created GeoJSON files
        """
        logger.info(
            f"Creating GeoJSON visualizations for {attention_result.model_type}"
        )

        output_path.mkdir(parents=True, exist_ok=True)
        created_files: list[Path] = []

        # Load cell data once
        cell_data = self._load_cell_data(cell_data_path)
        if not cell_data:
            logger.error("No cell data found, cannot create GeoJSON")
            return created_files

        # Create visualization for each attention type
        for (
            attention_key,
            attention_weights,
        ) in attention_result.attention_weights.items():
            try:
                geojson_path = output_path / f"{attention_key}_attention.geojson"

                self._create_single_geojson(
                    attention_weights,
                    attention_key,
                    cell_data,
                    cell_indices,
                    geojson_path,
                )

                created_files.append(geojson_path)
                logger.info(f"Created GeoJSON: {geojson_path}")

            except Exception as e:
                logger.error(f"Error creating GeoJSON for {attention_key}: {e}")

        return created_files

    def _load_cell_data(self, cell_data_path: Path) -> List[Dict[str, Any]]:
        """Load cell data from JSON file."""
        if not cell_data_path.exists():
            logger.error(f"Cell data file not found: {cell_data_path}")
            return []

        try:
            with open(cell_data_path, "r") as f:
                data = json.load(f)
                return data.get("cells", [])
        except Exception as e:
            logger.error(f"Error loading cell data: {e}")
            return []

    def _create_single_geojson(
        self,
        attention_weights: torch.Tensor,
        attention_name: str,
        cell_data: List[Dict[str, Any]],
        cell_indices: Dict[int, int],
        output_path: Path,
    ) -> None:
        """Create GeoJSON for a single attention type."""

        # Handle different attention tensor shapes
        if len(attention_weights.shape) == 2 and attention_weights.shape[0] == 1:
            # Shape [1, n_instances] - squeeze to [n_instances]
            attention_flat = attention_weights.squeeze(0)
        elif len(attention_weights.shape) == 1:
            # Shape [n_instances]
            attention_flat = attention_weights
        else:
            logger.warning(
                f"Unexpected attention shape: {attention_weights.shape}, flattening by mean"
            )
            attention_flat = attention_weights.mean(dim=0)

        # Convert to numpy for processing
        attention_np = cast(np.ndarray[Any, Any], attention_flat.cpu().detach().numpy())  # type: ignore

        if len(attention_np) == 0:
            logger.warning(f"Empty attention weights for {attention_name}")
            return

        # Create seamless color mapping based on attention values
        geojson_features = self._create_features(
            attention_np, cell_data, cell_indices, attention_name
        )

        # Calculate percentiles for metadata
        p1, p99 = np.percentile(attention_np, [1, 99])

        # Create FeatureCollection
        geojson_output: dict[str, Any] = {
            "type": "FeatureCollection",
            "properties": {
                "attention_type": attention_name,
                "visualization_type": "continuous_viridis_robust",
                "colormap": "viridis",
                "total_cells": len(attention_np),
                "attention_range": {
                    "min": float(attention_np.min()),
                    "max": float(attention_np.max()),
                    "mean": float(attention_np.mean()),
                    "std": float(attention_np.std()),
                    "p1": float(p1),
                    "p99": float(p99),
                },
                "color_scaling": {
                    "method": "percentile_clipping",
                    "lower_percentile": 1,
                    "upper_percentile": 99,
                    "description": "Values below p1 and above p99 are clipped to handle outliers",
                },
                "model_type": "attention_visualization",
            },
            "features": geojson_features,
        }

        # Save to file
        with open(output_path, "w") as f:
            ujson.dump(geojson_output, f, indent=4)

    def _create_features(
        self,
        attention_np: np.ndarray[Any, Any],
        cell_data: List[Dict[str, Any]],
        cell_indices: Dict[int, int],
        attention_name: str,
    ) -> List[Dict[str, Any]]:
        """
        Create GeoJSON features with continuous color mapping based on attention values.

        Each cell gets its own feature with a color that represents its attention value
        on a continuous scale using Viridis colormap with percentile-based robust scaling.
        """
        geojson_features: list[Dict[str, Any]] = []

        # Normalize attention values to [0, 1] for color mapping
        if len(attention_np) == 0:
            return geojson_features

        # Apply percentile-based robust scaling to handle outliers better
        p1, p99 = np.percentile(attention_np, [1, 99])

        logger.info(
            f"Attention range for {attention_name}: "
            f"min={attention_np.min():.4f}, max={attention_np.max():.4f}, "
            f"p1={p1:.4f}, p99={p99:.4f}"
        )

        # Create robust normalized values that clip outliers
        if p99 > p1:
            normalized_attention = np.where(
                attention_np <= p1,
                0.0,  # Bottom 1% get minimum color
                np.where(
                    attention_np >= p99,
                    1.0,  # Top 1% get maximum color
                    (attention_np - p1)
                    / (p99 - p1),  # Middle 98% get proportional scaling
                ),
            )
        else:
            # Handle case where all attention values are similar
            normalized_attention = attention_np * 0 + 0.5  # Mid-range color for all

        # Create inverted mapping from tensor index to cell_id
        inverted_indices = {idx: cell_id for cell_id, idx in cell_indices.items()}

        # Create a feature for each cell
        for tensor_idx, attention_value in enumerate(normalized_attention):
            if tensor_idx not in inverted_indices:
                continue

            cell_id = inverted_indices[tensor_idx]

            # Find the cell in cell_data
            cell = None
            for c in cell_data:
                if c.get("cell_id") == cell_id:
                    cell = c
                    break

            if cell is None:
                continue

            # Extract cell polygon
            polygon = self._extract_cell_polygon(cell)
            if polygon is None:
                continue

            # Create color based on attention value (continuous mapping)
            color = self._get_continuous_color(attention_value)

            # Create GeoJSON feature for this cell
            feature = get_template_segmentation()
            feature["id"] = f"{attention_name}_cell_{cell_id}"
            feature["geometry"]["type"] = "Polygon"
            feature["geometry"]["coordinates"] = polygon

            # Set feature properties
            feature["properties"]["classification"]["name"] = (
                f"{attention_name.replace('_', ' ').title()} - Cell {cell_id}"
            )
            feature["properties"]["classification"]["color"] = color

            # Add custom properties
            feature["properties"]["cell_id"] = cell_id
            feature["properties"]["attention_value"] = float(attention_np[tensor_idx])
            feature["properties"]["normalized_attention"] = float(attention_value)
            feature["properties"]["attention_type"] = attention_name

            geojson_features.append(feature)

        logger.info(
            f"Created {len(geojson_features)} continuous features for {attention_name}"
        )
        return geojson_features

    def _get_continuous_color(self, normalized_value: float) -> List[int]:
        """
        Get RGBA color for a normalized attention value (0-1).

        Uses Viridis colormap for perceptually uniform color mapping.
        Viridis ranges from purple (low) through blue, green, yellow to yellow (high).
        Returns [R, G, B, A] where A is the alpha/transparency value (0-255).
        """
        # Ensure value is in [0, 1] range
        normalized_value = max(0.0, min(1.0, normalized_value))

        # Viridis colormap approximation using piecewise linear interpolation
        # These are key colors from the Viridis colormap at specific points
        viridis_colors = [
            (68, 1, 84),  # 0.0 - Deep purple
            (59, 82, 139),  # 0.25 - Blue
            (33, 145, 140),  # 0.5 - Teal/Green
            (94, 201, 98),  # 0.75 - Yellow-green
            (253, 231, 37),  # 1.0 - Yellow
        ]

        # Find the two colors to interpolate between
        if normalized_value <= 0.0:
            return list(viridis_colors[0]) + [self.alpha]
        elif normalized_value >= 1.0:
            return list(viridis_colors[-1]) + [self.alpha]

        # Determine which segment we're in
        segment_size = 1.0 / (len(viridis_colors) - 1)
        segment_idx = int(normalized_value / segment_size)
        segment_idx = min(segment_idx, len(viridis_colors) - 2)

        # Local position within the segment [0, 1]
        local_t = (normalized_value - segment_idx * segment_size) / segment_size

        # Interpolate between the two colors
        color1 = viridis_colors[segment_idx]
        color2 = viridis_colors[segment_idx + 1]

        r = int(color1[0] + (color2[0] - color1[0]) * local_t)
        g = int(color1[1] + (color2[1] - color1[1]) * local_t)
        b = int(color1[2] + (color2[2] - color1[2]) * local_t)

        return [r, g, b, self.alpha]

    def _extract_cell_polygon(
        self, cell: Dict[str, Any]
    ) -> Optional[List[List[List[int]]]]:
        """Extract valid polygon from cell contour data."""

        if "contour" not in cell or not cell["contour"]:
            return None

        contour = cast(list[list[float] | tuple[float, float] | None], cell["contour"])
        if len(contour) < 3:
            return None

        # Convert to valid polygon coordinates
        valid_contour: list[list[int]] = []
        for point in contour:
            if isinstance(point, (list, tuple)) and len(point) >= 2:
                valid_contour.append([int(point[0]), int(point[1])])
            else:
                return None  # Invalid contour

        # Ensure polygon is closed
        if valid_contour and valid_contour[0] != valid_contour[-1]:
            valid_contour.append(valid_contour[0])

        if len(valid_contour) >= 4:  # At least 3 unique points + closing point
            return [valid_contour]  # MultiPolygon format

        return None

from pydantic import BaseModel, Field, field_validator
from enum import Enum
from pathlib import Path
from typing import Optional


class ExplainMethod(str, Enum):
    """Type for explainability methods."""

    attention = "attention"

    @classmethod
    def values(cls):
        return [member.value for member in cls]

    def __str__(self):
        return self.value


class AttentionAggregation(str, Enum):
    """Attention aggregation methods for GraphMIL."""

    pooling_only = "pooling_only"  # Only pooling attention
    gnn_only = "gnn_only"  # Only GNN attention
    gnn_layer = "gnn_layer"  # Specific GNN layer
    random_walk = "random_walk"  # a = p.T @ P_(l) @ p_(l-1) @ ... @ p_(1)


class Normalization(str, Enum):
    """Attention normalization methods."""

    min_max = "min_max"  # Min-Max normalization to [0, 1]
    z_score = "z_score"  # Z-score standardization (mean=0, std=1)
    robust = "robust"  # Robust scaling using median and IQR
    softmax = "softmax"  # Softmax normalization (sum to 1)
    sigmoid = "sigmoid"  # Sigmoid normalization to [0, 1]
    none = "none"  # No normalization


class VisualizationMode(str, Enum):
    """Visualization output modes."""

    geojson = "geojson"  # GeoJSON for pathology viewers (QuPath, etc.)
    graph = "graph"  # Interactive graph with attention edges
    heatmap = "heatmap"  # Attention heatmap overlay
    all = "all"  # Generate all visualization types


class ExplainerCreatorConfig(BaseModel):
    """Configuration for explainability methods."""

    method: ExplainMethod = Field(..., description="Explainability method to use")
    output_path: Path = Field(
        ..., description="Path where the explanations will be saved"
    )

    # Attention-specific configurations
    attention_aggregation: AttentionAggregation = Field(
        default=AttentionAggregation.pooling_only,
        description="How to aggregate attention for GraphMIL models",
    )
    gnn_layer_index: Optional[int] = Field(
        default=None,
        description="Specific GNN layer index when using gnn_layer aggregation",
    )
    class_index: Optional[int] = Field(
        default=None,
        description="Specific class index for multi-class attention (None for all classes)",
    )
    attention_head: Optional[int] = Field(
        default=None,
        description="Specific attention head for multi-head attention (None for mean)",
    )

    # Visualization configurations
    visualization_mode: VisualizationMode = Field(
        default=VisualizationMode.geojson,
        description="Type of visualization to generate",
    )
    color_scheme: tuple[list[int], list[int]] = Field(
        default=([35, 92, 236], [255, 0, 0]),  # Blue to red
        description="Color gradient for attention visualization (start_rgb, end_rgb)",
    )

    # Advanced options
    normalize_attention: bool = Field(
        default=True, description="Whether to normalize attention weights"
    )
    normalization: Normalization = Field(
        default=Normalization.min_max,
        description="Type of normalization to apply to attention weights",
    )
    visualize_cell_types: bool = Field(
        default=False,
        description="Whether to create cell type visualization (graph colored by cell types)",
    )

    @field_validator("method")
    def validate_method(cls, v: str) -> str:
        if v not in ExplainMethod.values():
            raise ValueError(
                f"Unsupported explainability method: {v}. Supported methods are: {ExplainMethod.values()}"
            )
        return v

    @field_validator("gnn_layer_index")
    def validate_gnn_layer_index(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v < 0:
            raise ValueError("GNN layer index must be non-negative")
        return v

from pydantic import BaseModel, Field, field_validator
from pathlib import Path
from enum import Enum
from .CellSegmenterConfig import ModelType

class GraphCreatorType(str, Enum):
    knn = "knn"
    radius = "radius"
    delaunay_radius = "delaunay_radius"
    dilate = "dilate"
    similarity = "similarity"
    
    @classmethod
    def values(cls):
        return [member.value for member in cls]

    def __str__(self):
        return self.value
    
class GraphCreatorCategory(str, Enum):
    """Type for feature extraction methods."""
    Positional = ["knn", "radius", "delaunay_radius", "dilate"]
    FeatureDependent = ["similarity"]
    
    def __contains__(self, item: str) -> bool:
        """Check if the item is in the feature extraction type."""
        return item in self.value

class GraphCreatorConfig(BaseModel):
    """Configuration for graph creation from segmented cells"""
    method: GraphCreatorType = Field(..., description="Name of the graph creator to use")
    gpu: int = Field(0, description="GPU device ID to use for processing")
    patched_slide_path: Path = Field(..., description="Path to the patched slide image")
    segmentation_model: ModelType = Field(..., description="Name of the segmentation model used to extract cells")
    plot: bool = Field(True, description="Whether to plot the graph")
    debug: bool = Field(default=False, description="Enable debug mode with interactive parameter tuning")

    @field_validator('method')
    def validate_model(cls, v: str) -> str:
        if v not in GraphCreatorType.values():
            raise ValueError(f"Unsupported graph creator type: {v}. Supported graph creators are: {GraphCreatorType.values()}")
        return v
    
    @field_validator('segmentation_model')
    def validate_segmentation_model(cls, v: str) -> str:
        if v not in ModelType.values():
            raise ValueError(f"Unsupported segmentation model: {v}. Supported segmentation models are: {ModelType.values()}")
        return v
    
    class Config:
        arbitrary_types_allowed = True
from pydantic import BaseModel, Field, field_validator
from pathlib import Path
from typing import Optional
from enum import Enum
from .CellSegmenterConfig import ModelType

class ExtractorType(str, Enum):
    """Type for feature extractors."""
    pyradiomics_gray = "pyradiomics_gray"
    pyradiomics_hed = "pyradiomics_hed"
    pyradiomics_hue = "pyradiomics_hue"
    morphometrics = "morphometrics"
    connectivity = "connectivity"
    structure = "structure"
    geometric = "geometric"
    resnet50 = "resnet50"
    gigapath = "gigapath"
    uni = "uni"
    titan = "titan"
    
    @classmethod
    def values(cls):
        return [member.value for member in cls]

    def __str__(self):
        return self.value
    
class FeatureExtractionType(str, Enum):
    """Type for feature extraction methods."""
    Morphological = ["pyradiomics_hed", "morphometrics", "pyradiomics_gray", "pyradiomics_hue"]
    Topological = ["connectivity", "structure", "geometric"]
    Embedding = ["resnet50", "gigapath", "uni", "titan"]
    
    def __contains__(self, item: str) -> bool:
        """Check if the item is in the feature extraction type."""
        return item in self.value

class FeatureExtractorConfig(BaseModel):
    """Configuration for feature extraction from segmented cells"""
    extractor: ExtractorType = Field(..., description="Name of the feature extractor to use")
    patched_slide_path: Path = Field(..., description="Path to the patched slide image")
    wsi_path: Optional[Path] = Field(None, description="Path to the whole slide image")
    segmentation_model: Optional[ModelType] = Field(None, description="Name of the segmentation model used to extract cells")
    graph_method: Optional[str] = Field(None, description="Graph method to use for feature extraction")

    @field_validator('extractor')
    def validate_model(cls, v: str) -> str:
        if v not in ExtractorType.values():
            raise ValueError(f"Unsupported feature extractor type: {v}. Supported feature extractors are: {ExtractorType.values()}")
        return v
    
    @field_validator('segmentation_model')
    def validate_segmentation_model(cls, v: str | None) -> str | None:
        if v is not None and v not in ModelType.values():
            raise ValueError(f"Unsupported segmentation model: {v}. Supported segmentation models are: {ModelType.values()}")
        return v
    
    class Config:
        arbitrary_types_allowed = True
from pydantic import BaseModel, Field, field_validator
from pathlib import Path

class PatchExtractorConfig(BaseModel):
    """Configuration for data preparation"""
    output_path: Path = Field(..., description="Path where output files will be saved")
    wsi_path: Path = Field(..., description="Path to whole slide image file")
    patch_size: int = Field(..., gt=0, description="Patch size in pixels")
    patch_overlap: float | int = Field(..., ge=0, le=100, description="Overlap between adjacent patches as percentage (0-100)")
    target_mag: float = Field(..., gt=0, description="Target magnification")
    
    @field_validator('patch_overlap')
    def validate_overlap(cls, v: int | float) -> float | int:
        if v < 0 or v > 100:
            raise ValueError(f"Overlap must be between 0 and 100 (got {v})")
        return v
    
    class Config:
        arbitrary_types_allowed = True

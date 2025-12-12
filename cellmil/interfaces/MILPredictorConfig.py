from pydantic import BaseModel, Field, field_validator
from pathlib import Path
from .CellSegmenterConfig import ModelType
from .FeatureExtractorConfig import ExtractorType
from .MIL import MILType

class MILPredictorConfig(BaseModel):
    """Configuration for MIL prediction using CLAM or standard MIL models"""
    model: MILType = Field(..., description="Type of MIL model to use")
    gpu: int = Field(0, description="GPU index to use for prediction")
    patched_slide_path: Path = Field(..., description="Path to the patched slide image")
    extractor: ExtractorType = Field(..., description="Feature extractor to use for patch extraction")
    segmentation_model: ModelType = Field(..., description="Segmentation model used to extract cells")
    chkpt_path: Path | None = Field(None, description="Path to the checkpoint file for the MIL model")

    @field_validator('model')
    def validate_model(cls, v: str) -> str:
        if v not in MILType.values():
            raise ValueError(f"Unsupported MIL model type: {v}. Supported models are: {MILType.values()}")
        return v

    class Config:
        arbitrary_types_allowed = True

from pydantic import BaseModel, Field, field_validator
from pathlib import Path
from .CellSegmenterConfig import ModelType
from .FeatureExtractorConfig import ExtractorType
from .GraphCreatorConfig import GraphCreatorType
from .MIL import MILType
import torch

torch.set_float32_matmul_precision('medium')

class MILTrainerConfig(BaseModel):
    """Configuration for MIL prediction using CLAM or standard MIL models"""
    root: Path = Field(..., description="Root directory for the dataset")
    folder: Path = Field(..., description="Path to the dataset directory")
    excel_path: Path = Field(..., description="Path to the Excel file with target labels")
    label: str = Field(..., description="Label for the dataset (e.g., 'dcr')")
    model: MILType = Field(..., description="Type of MIL model to use")
    gpu: int = Field(0, description="GPU index to use for prediction")
    extractor: ExtractorType | list[ExtractorType] = Field(..., description="Feature extractor to use for patch extraction")
    segmentation_model: ModelType = Field(..., description="Segmentation model used to extract cells")
    graph_creator: GraphCreatorType = Field(..., description="Graph creation method to use for GNN+MIL models")
    ckpt_path: Path = Field(..., description="Path to the checkpoint directory where the model will be saved")
    normalization: bool = Field(default=False, description="Whether to apply normalization to the input data")
    correlation_filter: float = Field(default=0.0, description="Correlation filter threshold for feature selection")
    cell_type: bool = Field(default=False, description="Whether to include cell type information in the features")
    n_bins: int = Field(default=4, description="Number of time bins for discrete-time survival models")

    @field_validator('model')
    def validate_model(cls, v: str) -> str:
        if v not in MILType.values():
            raise ValueError(f"Unsupported MIL model type: {v}. Supported models are: {MILType.values()}")
        return v

    class Config:
        arbitrary_types_allowed = True
    
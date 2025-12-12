from pydantic import BaseModel, Field, field_validator
from pathlib import Path
from .CellSegmenterConfig import ModelType
from .FeatureExtractorConfig import ExtractorType
from .GraphCreatorConfig import GraphCreatorType

class DatasetCreatorConfig(BaseModel):
    """Configuration for creating a dataset for MIL training."""
    excel_path: Path = Field(..., description="Path to the Excel file containing metadata information")
    output_path: Path = Field(..., description="Path where the dataset will be saved")
    gpu: int = Field(0, ge=0, description="GPU index to use for processing (0 for CPU)")
    segmentation_models: list[ModelType] | None = Field(..., description="List of segmentation models to use for cell segmentation")
    extractors: list[ExtractorType] = Field(..., description="List of feature extractors to use for feature extraction")
    graph_methods: list[GraphCreatorType] | None = Field(..., description="Graph creation methods to use")

    @field_validator('segmentation_models')
    def validate_segmentation_models(cls, v: list[ModelType]) -> list[ModelType]:
        if not v:
            raise ValueError("At least one segmentation model must be specified")
        else:
            for model in v:
                if model not in ModelType.values():
                    raise ValueError(f"Invalid segmentation model: {model}")
        return v

    @field_validator('extractors')
    def validate_extractors(cls, v: list[ExtractorType]) -> list[ExtractorType]:
        if not v:
            raise ValueError("At least one feature extractor must be specified")
        else:
            for extractor in v:
                if extractor not in ExtractorType.values():
                    raise ValueError(f"Invalid feature extractor: {extractor}")
        return v

    @field_validator('graph_methods')
    def validate_graph_methods(cls, v: list[GraphCreatorType]) -> list[GraphCreatorType]:
        if not v:
            raise ValueError("At least one graph creation method must be specified")
        else:
            for method in v:
                if method not in GraphCreatorType.values():
                    raise ValueError(f"Invalid graph creation method: {method}")
        return v

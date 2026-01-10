from pydantic import BaseModel, Field
from pathlib import Path
from .EvaluationReporterConfig import Metrics
from enum import Enum

class AggregationMethod(str, Enum):
    """Enumeration of available aggregation methods for external validation."""
    mean = "mean"
    median = "median"
    majority = "majority"
    everything = "everything"
    
    @classmethod
    def values(cls):
        return [member.value for member in cls]

    def __str__(self):
        return self.value
    
class FinalModel(str, Enum):
    final = "final"
    ensemble = "ensemble"
    
    @classmethod
    def values(cls):
        return [member.value for member in cls]
    
    def __str__(self):
        return self.value

class EvaluationExternalValidatorConfig(BaseModel):
    metrics: list[Metrics] = Field(..., description="List of metrics to generate evaluation reports for")
    output_dir: Path = Field(default=Path("./evaluation_reports"), description="Directory to save evaluation reports")
    models_dir: Path = Field(..., description="Directory containing pre-trained models for external validation")
    final_model: FinalModel = Field(default=FinalModel.final, description="Whether to use the final model or an ensemble for validation")
    aggregation_method: AggregationMethod = Field(default=AggregationMethod.mean, description="Method to aggregate predictions from multiple models in ensemble")
    dataset_dir: Path = Field(..., description="Directory containing datasets for external validation")
    root_dir: Path = Field(..., description="Root directory for caching datasets")
    dp_metadata_file: Path = Field(..., description="Path to the metadata Excel file for datasets")
    
    class Config:
        arbitrary_types_allowed = True
from pydantic import BaseModel, Field
from enum import Enum
from pathlib import Path

class Metrics(str, Enum):
    """Enumeration of available metrics for metrics report."""
    accuracy = "accuracy"
    precision = "precision"
    recall = "recall"
    f1 = "f1"
    auroc = "auroc"
    c_index = "c_index"
    
    @classmethod
    def values(cls):
        return [member.value for member in cls]

    def __str__(self):
        return self.value

class EvaluationReporterConfig(BaseModel):
    """Configuration for generating evaluation reports"""
    metrics: list[Metrics] = Field(..., description="List of metrics to generate evaluation reports for")
    team : str = Field(..., description="Team name where the project belongs in wandb")
    projects : list[str] = Field(..., description="Project/s name where data is allocated in wandb")
    output_dir: Path = Field(default=Path("./evaluation_reports"), description="Directory to save evaluation reports")

    class Config:
        arbitrary_types_allowed = True
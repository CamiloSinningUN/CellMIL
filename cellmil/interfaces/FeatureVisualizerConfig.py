from pydantic import BaseModel, Field
from pathlib import Path


class FeatureVisualizerConfig(BaseModel):
    """Configuration for feature visualization from a dataset containing multiple slides"""
    dataset: Path = Field(..., description="Path to the folder containing multiple slide folders.")

    class Config:
        arbitrary_types_allowed = True
from pydantic import BaseModel, Field
from pathlib import Path

class EvaluationExternalValidatorConfig(BaseModel):
    model_path: Path = Field(..., description="Path to the pre-trained folder of model for external validation")

    class Config:
        arbitrary_types_allowed = True
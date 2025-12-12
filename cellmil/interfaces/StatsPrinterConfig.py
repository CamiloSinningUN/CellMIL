from pydantic import BaseModel, Field

class StatsPrinterConfig(BaseModel):
    """Configuration for generating statistics reports"""
    team : str = Field(..., description="Team name where the project belongs in wandb")
    projects : list[str] = Field(..., description="Project/s name where data is allocated in wandb")

    class Config:
        arbitrary_types_allowed = True
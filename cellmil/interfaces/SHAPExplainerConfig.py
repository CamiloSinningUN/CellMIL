from pydantic import BaseModel, Field, field_validator
from pathlib import Path
from typing import Optional
from enum import Enum


class SHAPExplainerType(str, Enum):
    """SHAP explainer types."""

    gradient = "gradient"  # GradientExplainer (fast, requires gradients)
    deep = "deep"  # DeepExplainer (for deep learning models)
    kernel = "kernel"  # KernelExplainer (model-agnostic, slow)
    
    @classmethod
    def values(cls):
        return [member.value for member in cls]

    def __str__(self):
        return self.value


class SHAPExplainerConfig(BaseModel):
    """Configuration for SHAP explainability method."""

    output_path: Path = Field(
        ..., description="Path where the SHAP explanations will be saved"
    )

    # Sampling configuration
    num_bins: int = Field(
        default=5,
        description="Number of quantile bins for stratified sampling based on attention scores",
    )
    samples_per_bin: int = Field(
        default=1000,
        description="Number of cells to sample from each attention quantile bin",
    )
    max_total_samples: Optional[int] = Field(
        default=None,
        description="Maximum total number of samples to use (overrides samples_per_bin if specified)",
    )

    # SHAP computation configuration
    explainer_type: SHAPExplainerType = Field(
        default=SHAPExplainerType.gradient,
        description="Type of SHAP explainer to use (gradient is fastest for neural networks)",
    )
    background_percentage: float = Field(
        default=0.2,
        description="Percentage of sampled cells to use as background data (0.0 to 1.0)",
    )
    nsamples: int = Field(
        default=500,
        description="Number of coalitions for SHAP kernel explainer per cell (only for kernel explainer)",
    )
    explain_top_cells: Optional[int] = Field(
        default=None,
        description="Number of top cells to explain (None = explain all sampled cells)",
    )
    explain_per_head: bool = Field(
        default=True,
        description="Whether to compute SHAP values for each attention head separately",
    )
    explain_mean_head: bool = Field(
        default=True,
        description="Whether to also compute SHAP values for mean attention across heads",
    )

    # Analysis configuration
    top_features: int = Field(
        default=20,
        description="Number of top features to highlight in summary visualizations",
    )

    # Advanced options
    random_seed: int = Field(
        default=42, description="Random seed for reproducible sampling"
    )
    batch_size: int = Field(
        default=1024,
        description="Batch size for computing attention weights on the full dataset",
    )
    save_raw_shap_values: bool = Field(
        default=True,
        description="Whether to save raw SHAP values (can be large for many features)",
    )
    create_summary_plots: bool = Field(
        default=True,
        description="Whether to create summary plots (bar plot, beeswarm plot, etc.)",
    )

    @field_validator("num_bins")
    def validate_num_bins(cls, v: int) -> int:
        if v < 2:
            raise ValueError("num_bins must be at least 2")
        if v > 20:
            raise ValueError(
                "num_bins should not exceed 20 for meaningful stratification"
            )
        return v

    @field_validator("samples_per_bin")
    def validate_samples_per_bin(cls, v: int) -> int:
        if v < 10:
            raise ValueError("samples_per_bin must be at least 10")
        return v

    @field_validator("background_percentage")
    def validate_background_percentage(cls, v: float) -> float:
        if not 0.0 < v <= 1.0:
            raise ValueError("background_percentage must be between 0.0 and 1.0")
        return v

    @field_validator("nsamples")
    def validate_nsamples(cls, v: int) -> int:
        if v < 10:
            raise ValueError("nsamples must be at least 10 for reliable SHAP values")
        return v

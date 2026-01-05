from dataclasses import dataclass, field
from typing import Literal


@dataclass
class TableConfig:
    """Configuration for LaTeX table generation."""

    # Output configuration
    output_file: str

    # Table type
    table_type: Literal["classification", "survival"]

    # Filtering
    tasks: list[str] | None = None  # None means all tasks
    features: list[str] | None = None  # None means all features
    models: list[str] | None = None  # None means all models
    baseline_features: list[str] | None = None  # Features to show in baseline section (e.g., ResNet50, GigaPath)
    include_regularized: bool = True
    include_non_regularized: bool = True
    include_stratified: bool = True

    # Metrics for each table type
    classification_metrics: list[str] = field(default_factory=lambda: ["f1", "recall"])
    survival_metrics: list[str] = field(default_factory=lambda: ["c_index"])

    # Mapping for display names (optional)
    task_mapping: dict[str, str] = field(default_factory=dict) #type: ignore
    feature_mapping: dict[str, str] = field(default_factory=dict) #type: ignore
    model_mapping: dict[str, str] = field(default_factory=dict) #type: ignore
    metric_mapping: dict[str, str] = field(default_factory=dict) #type: ignore

    # Table formatting
    caption: str = ""
    label: str = ""
    support_mapping: dict[str, int] = field( #type: ignore
        default_factory=dict 
    )  # task -> support count

    # Precision for numbers
    precision: int = 3

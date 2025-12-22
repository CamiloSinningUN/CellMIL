import json
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any, Union, Optional
from dataclasses import dataclass, asdict
from cellmil.utils import logger
from cellmil.datamodels.transforms import TransformPipeline, LabelTransformPipeline
        


@dataclass
class FoldMetadata:
    """Metadata for a single fold."""

    fold_idx: int
    train_size: int
    val_size: int
    best_epoch: int
    best_metric_value: float
    metric_name: str
    is_survival: bool
    metrics: dict[str, Any]


@dataclass
class ExperimentMetadata:
    """Metadata for the entire k-fold experiment."""

    name: str
    k_folds: int
    random_state: int
    balance_cell_counts: bool
    cell_balance_bins: int
    is_survival: bool
    aggregated_metrics: dict[str, Any]
    best_fold_idx: int
    avg_best_epoch: float
    dataset_config: dict[str, Any]
    model_config: dict[str, Any]


class ModelStorage:
    """
    Manages storage and retrieval of k-fold cross-validation results.

    Directory structure:
    {output_dir}/
        ├── experiment_metadata.json
        ├── fold_0/
        │   ├── best_model.ckpt
        │   ├── train_indices.json
        │   ├── val_indices.json
        │   ├── predictions.csv
        │   ├── transforms/                   
        │   │   ├── pipeline_config.json
        │   │   ├── transform_0_*.json
        │   │   └── ...
        │   ├── label_transforms/             
        │   │   ├── pipeline.json
        │   │   ├── transform_0.json
        │   │   └── ...
        │   └── metadata.json
        ├── fold_1/
        │   └── ...
        ├── ...
        └── final_model/
            ├── final_model.ckpt
            └── metadata.json
    """

    def __init__(self, output_dir: Union[str, Path], experiment_name: str, load_existing: bool = False):
        """
        Initialize ModelStorage.

        Args:
            output_dir: Base directory for storing results
            experiment_name: Name of the experiment
            load_existing: If True, load from existing directory without versioning
        """
        base_dir = Path(output_dir)
        proposed_dir = base_dir / experiment_name
        
        # If loading existing, use the directory as-is
        if load_existing:
            if not proposed_dir.exists():
                raise FileNotFoundError(
                    f"Cannot load existing experiment: '{proposed_dir}' does not exist"
                )
            self.output_dir = proposed_dir
            self.experiment_name = experiment_name
            logger.info(f"Loading existing experiment: '{self.experiment_name}'")
        else:
            # If directory exists, create versioned name
            if proposed_dir.exists():
                version = 2
                while (base_dir / f"{experiment_name}_v{version}").exists():
                    version += 1
                self.output_dir = base_dir / f"{experiment_name}_v{version}"
                self.experiment_name = f"{experiment_name}_v{version}"
                logger.warning(
                    f"Experiment '{experiment_name}' already exists. "
                    f"Creating new version: '{self.experiment_name}'"
                )
            else:
                self.output_dir = proposed_dir
                self.experiment_name = experiment_name
            
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.fold_metadata: dict[int, FoldMetadata] = {}
        self.experiment_metadata: Optional[ExperimentMetadata] = None
        
        # If loading existing, load all metadata
        if load_existing:
            self._load_all_metadata()

    def save_fold_results(
        self,
        fold_idx: int,
        checkpoint_path: Union[str, Path],
        train_indices: list[int],
        val_indices: list[int],
        predictions: dict[str, Any],
        metadata: FoldMetadata,
        transforms: Any = None,
        label_transforms: Any = None,
    ) -> None:
        """
        Save all results for a single fold.

        Args:
            fold_idx: Fold index
            checkpoint_path: Path to the best checkpoint for this fold
            train_indices: Training indices
            val_indices: Validation indices
            predictions: Dictionary with 'y_true' and 'y_pred' arrays
            metadata: Fold metadata
            transforms: Optional feature transforms
            label_transforms: Optional label transforms
        """
        fold_dir = self.output_dir / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        # Save checkpoint
        if Path(checkpoint_path).exists():
            shutil.copy2(checkpoint_path, fold_dir / "best_model.ckpt")

        # Save indices
        with open(fold_dir / "train_indices.json", "w") as f:
            json.dump(train_indices, f, indent=2)

        with open(fold_dir / "val_indices.json", "w") as f:
            json.dump(val_indices, f, indent=2)

        # Save predictions as CSV
        df = pd.DataFrame(predictions)
        df.to_csv(fold_dir / "predictions.csv", index=False)

        # Save transforms using their native save methods (JSON format)
        if transforms is not None:
            # TransformPipeline or Transform - saves to directory/file
            transforms_dir = fold_dir / "transforms"
            transforms.save(transforms_dir)

        if label_transforms is not None:
            # LabelTransformPipeline or LabelTransform - saves to directory/file
            label_transforms_dir = fold_dir / "label_transforms"
            label_transforms.save(label_transforms_dir)

        # Save metadata
        with open(fold_dir / "metadata.json", "w") as f:
            json.dump(asdict(metadata), f, indent=2)

        self.fold_metadata[fold_idx] = metadata

    def save_experiment_metadata(self, metadata: ExperimentMetadata) -> None:
        """Save overall experiment metadata."""
        self.experiment_metadata = metadata
        with open(self.output_dir / "experiment_metadata.json", "w") as f:
            json.dump(asdict(metadata), f, indent=2)

    def save_final_model(
        self,
        checkpoint_path: Union[str, Path],
        avg_epochs: float,
        final_metrics: dict[str, Any],
    ) -> None:
        """
        Save the final model trained on average epochs.

        Args:
            checkpoint_path: Path to final model checkpoint
            avg_epochs: Average number of epochs used
            final_metrics: Metrics from final model
        """
        final_dir = self.output_dir / "final_model"
        final_dir.mkdir(parents=True, exist_ok=True)

        if Path(checkpoint_path).exists():
            shutil.copy2(checkpoint_path, final_dir / "final_model.ckpt")

        metadata: dict[str, Any] = {
            "avg_epochs": avg_epochs,
            "metrics": final_metrics,
        }

        with open(final_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    def load_fold_checkpoint(self, fold_idx: int) -> Path:
        """Load checkpoint path for a specific fold."""
        checkpoint_path = self.output_dir / f"fold_{fold_idx}" / "best_model.ckpt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found for fold {fold_idx}")
        return checkpoint_path

    def load_final_checkpoint(self) -> Path:
        """Load the final model checkpoint."""
        checkpoint_path = self.output_dir / "final_model" / "final_model.ckpt"
        if not checkpoint_path.exists():
            raise FileNotFoundError("Final model checkpoint not found")
        return checkpoint_path

    def load_fold_predictions(self, fold_idx: int) -> pd.DataFrame:
        """Load predictions for a specific fold."""
        pred_path = self.output_dir / f"fold_{fold_idx}" / "predictions.csv"
        if not pred_path.exists():
            raise FileNotFoundError(f"Predictions not found for fold {fold_idx}")
        return pd.read_csv(pred_path) # type: ignore

    def load_all_predictions(self) -> pd.DataFrame:
        """Load and concatenate predictions from all folds."""
        all_preds: list[pd.DataFrame] = []
        for fold_idx in sorted(self.fold_metadata.keys()):
            df = self.load_fold_predictions(fold_idx)
            df["fold"] = fold_idx
            all_preds.append(df)
        return pd.concat(all_preds, ignore_index=True)

    def load_fold_transforms(self, fold_idx: int) -> tuple[Any, Any]:
        """Load transforms for a specific fold."""
        
        fold_dir = self.output_dir / f"fold_{fold_idx}"

        transforms = None
        label_transforms = None

        # Try loading from JSON (native format)
        transforms_dir = fold_dir / "transforms"
        if transforms_dir.exists():
            try:
                transforms = TransformPipeline.load(transforms_dir)
            except Exception as e:
                logger.warning(f"Failed to load transforms from JSON: {e}")

        # Try loading label transforms from JSON (native format)
        label_transforms_dir = fold_dir / "label_transforms"
        if label_transforms_dir.exists():
            try:
                label_transforms = LabelTransformPipeline.load(label_transforms_dir)
            except Exception as e:
                logger.warning(f"Failed to load label transforms from JSON: {e}")

        return transforms, label_transforms

    def get_average_best_epoch(self) -> float:
        """Calculate average of best epochs across all folds."""
        if not self.fold_metadata:
            raise ValueError("No fold metadata available")
        epochs = [meta.best_epoch for meta in self.fold_metadata.values()]
        return float(np.mean(epochs))

    def get_experiment_summary(self) -> dict[str, Any]:
        """Get a summary of the entire experiment."""
        if self.experiment_metadata is None:
            raise ValueError("Experiment metadata not saved yet")

        summary = asdict(self.experiment_metadata)
        summary["folds"] = {
            f"fold_{idx}": asdict(meta) for idx, meta in self.fold_metadata.items()
        }

        return summary

    def get_fold_indices(self, fold_idx: int) -> tuple[list[int], list[int]]:
        """Get train and validation indices for a specific fold."""
        fold_dir = self.output_dir / f"fold_{fold_idx}"

        with open(fold_dir / "train_indices.json", "r") as f:
            train_indices = json.load(f)

        with open(fold_dir / "val_indices.json", "r") as f:
            val_indices = json.load(f)

        return train_indices, val_indices

    @classmethod
    def from_directory(cls, experiment_dir: Union[str, Path]) -> "ModelStorage":
        """
        Load an existing experiment from a directory.

        Args:
            experiment_dir: Path to the experiment directory

        Returns:
            ModelStorage instance with loaded metadata

        Example:
            >>> storage = ModelStorage.from_directory("/path/to/experiments/my_experiment")
            >>> print(storage.experiment_metadata)
            >>> predictions = storage.load_all_predictions()
        """
        experiment_dir = Path(experiment_dir)
        if not experiment_dir.exists():
            raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")

        # Extract experiment name and parent directory
        experiment_name = experiment_dir.name
        output_dir = experiment_dir.parent

        # Create instance with load_existing=True
        return cls(output_dir, experiment_name, load_existing=True)

    def _load_experiment_metadata(self) -> None:
        """Load experiment metadata from disk."""
        metadata_path = self.output_dir / "experiment_metadata.json"
        if not metadata_path.exists():
            logger.warning("Experiment metadata file not found")
            return

        with open(metadata_path, "r") as f:
            metadata_dict = json.load(f)

        self.experiment_metadata = ExperimentMetadata(**metadata_dict)

    def _load_fold_metadata(self, fold_idx: int) -> None:
        """Load metadata for a specific fold."""
        metadata_path = self.output_dir / f"fold_{fold_idx}" / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found for fold {fold_idx}")

        with open(metadata_path, "r") as f:
            metadata_dict = json.load(f)

        self.fold_metadata[fold_idx] = FoldMetadata(**metadata_dict)

    def _load_all_metadata(self) -> None:
        """Load all experiment and fold metadata from disk."""
        # Load experiment metadata
        self._load_experiment_metadata()

        # Find all fold directories and load their metadata
        fold_dirs = sorted(self.output_dir.glob("fold_*"))
        for fold_dir in fold_dirs:
            try:
                fold_idx = int(fold_dir.name.split("_")[1])
                self._load_fold_metadata(fold_idx)
            except (ValueError, IndexError) as _:
                logger.warning(f"Skipping invalid fold directory: {fold_dir.name}")

        logger.info(
            f"Loaded experiment '{self.experiment_name}' with {len(self.fold_metadata)} folds"
        )

    def list_folds(self) -> list[int]:
        """Get list of available fold indices."""
        return sorted(self.fold_metadata.keys())

    def has_final_model(self) -> bool:
        """Check if a final model exists."""
        return (self.output_dir / "final_model" / "final_model.ckpt").exists()

    def __repr__(self) -> str:
        n_folds = len(self.fold_metadata)
        return f"ModelStorage(experiment='{self.experiment_name}', folds={n_folds}, dir='{self.output_dir}')'"

"""
Transform pipeline for chaining multiple feature transforms.
"""

from typing import Any, Dict, List, Optional, Union
import torch
from pathlib import Path
import json

from cellmil.utils import logger
from .base_transform import Transform, FittableTransform
from .correlation_filter import CorrelationFilterTransform
from .normalization import RobustScalerTransform


class TransformPipeline:
    """
    Pipeline for chaining multiple feature transforms.

    Supports both fittable and non-fittable transforms, automatic fitting
    on training data, and serialization for reuse on new samples.
    """

    def __init__(self, transforms: List[Transform]):
        """
        Initialize the transform pipeline.

        Args:
            transforms: List of transforms to apply in order
        """
        self.transforms = transforms
        self.is_fitted = False

    def fit(
        self, features: torch.Tensor, feature_names: Optional[List[str]] = None
    ) -> "TransformPipeline":
        """
        Fit all fittable transforms in the pipeline on training data.

        Args:
            features: Training features tensor of shape (n_instances, n_features)
            feature_names: Optional list of feature names for logging

        Returns:
            Self for method chaining
        """
        logger.info(
            f"Fitting transform pipeline with {len(self.transforms)} transforms..."
        )

        current_features = features
        current_feature_names: list[str] | None = feature_names

        for i, transform in enumerate(self.transforms):
            logger.info(
                f"Fitting transform {i + 1}/{len(self.transforms)}: {transform.name}"
            )

            if isinstance(transform, FittableTransform):
                # Fit the transform
                transform.fit(current_features, current_feature_names)

                # Apply the transform to get features for next transform
                current_features = transform.transform(current_features)

                # Update feature names if we have them and the transform removed features
                if current_feature_names is not None and isinstance(
                    transform, CorrelationFilterTransform
                ):
                    feature_mask = transform.get_feature_importance_mask()
                    if feature_mask is not None:
                        current_feature_names = [
                            name
                            for j, name in enumerate(current_feature_names)
                            if j < len(feature_mask) and feature_mask[j]
                        ]
            else:
                # For non-fittable transforms, just apply them
                current_features = transform.transform(current_features)

            logger.info(
                f"Transform {transform.name} output shape: {current_features.shape}"
            )

        self.is_fitted = True
        logger.info("Transform pipeline fitting completed")
        return self

    def transform(self, features: torch.Tensor) -> torch.Tensor:
        """
        Apply all transforms in the pipeline to features.

        Args:
            features: Input features tensor

        Returns:
            Transformed features tensor

        Raises:
            RuntimeError: If pipeline contains unfitted transforms
        """
        # Check if all fittable transforms are fitted
        for transform in self.transforms:
            if isinstance(transform, FittableTransform) and not transform.is_fitted:
                raise RuntimeError(
                    f"Transform '{transform.name}' must be fitted before transforming data. "
                    f"Call fit() or fit_transform() first."
                )

        current_features = features

        for transform in self.transforms:
            current_features = transform.transform(current_features)

        return current_features

    def fit_transform(
        self, features: torch.Tensor, feature_names: Optional[List[str]] = None
    ) -> torch.Tensor:
        """
        Fit the pipeline and apply it to the features.

        Args:
            features: Features to fit and transform
            feature_names: Optional list of feature names for logging

        Returns:
            Transformed features
        """
        self.fit(features, feature_names)
        return self.transform(features)

    def save(self, path: Union[str, Path]) -> None:
        """
        Save the entire pipeline to disk.

        Args:
            path: Path to save the pipeline (should be a directory)
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save pipeline metadata
        pipeline_config: dict[str, Any] = {
            "num_transforms": len(self.transforms),
            "transform_names": [t.name for t in self.transforms],
            "is_fitted": self.is_fitted,
        }

        with open(path / "pipeline_config.json", "w") as f:
            json.dump(pipeline_config, f, indent=2)

        # Save each transform individually
        for i, transform in enumerate(self.transforms):
            transform_path = path / f"transform_{i}_{transform.name}.json"
            transform.save(transform_path)

        logger.info(f"Saved transform pipeline to {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "TransformPipeline":
        """
        Load a transform pipeline from disk.

        Args:
            path: Path to load the pipeline from (should be a directory)

        Returns:
            Loaded transform pipeline
        """

        path = Path(path)

        # Load pipeline metadata
        with open(path / "pipeline_config.json", "r") as f:
            pipeline_config = json.load(f)

        # Map of transform class names to actual classes
        transform_classes: dict[str, Any] = {
            "CorrelationFilterTransform": CorrelationFilterTransform,
            "RobustScalerTransform": RobustScalerTransform,
        }

        # Load each transform
        transforms: list[Transform] = []
        for i in range(pipeline_config["num_transforms"]):
            transform_name = pipeline_config["transform_names"][i]
            transform_path = path / f"transform_{i}_{transform_name}.json"

            # Load transform config to determine class
            with open(transform_path, "r") as f:
                transform_config = json.load(f)

            transform_class_name = transform_config.get("transform_class")
            if transform_class_name not in transform_classes:
                raise ValueError(f"Unknown transform class: {transform_class_name}")

            transform_class = transform_classes[transform_class_name]
            transform = transform_class.load(transform_path)
            transforms.append(transform)

        # Create pipeline
        pipeline = cls(transforms)
        pipeline.is_fitted = pipeline_config["is_fitted"]

        logger.info(f"Loaded transform pipeline from {path}")
        return pipeline

    def get_config(self) -> Dict[str, Any]:
        """
        Get the configuration dictionary for the entire pipeline.

        Returns:
            Dictionary containing pipeline and all transform configurations
        """
        return {
            "pipeline_type": "TransformPipeline",
            "transforms": [t.get_config() for t in self.transforms],
            "is_fitted": self.is_fitted,
        }

    def get_config_for_hashing(self) -> Dict[str, Any]:
        """
        Get stable configuration for hashing that excludes fitted state.
        
        Returns:
            Dictionary containing pipeline configuration without fitted state
        """
        stable_transforms: list[Dict[str, Any]] = []
        for transform in self.transforms:
            transform_config = transform.get_config()
            # Remove any state-dependent keys that change after fitting
            config_copy = transform_config.copy()
            
            # Remove fitted state indicators
            config_copy.pop('is_fitted', None)
            
            # Remove fitted parameters that vary after fitting
            fitted_keys = [
                # General fitted parameters
                'num_features_', 'constant_features_mask_', 'removed_features_',
                'center_', 'scale_',
                # Correlation filter specific parameters
                'keep_mask', 'non_constant_mask', 'num_original_features', 'removed_feature_indices',
                # Robust scaler specific parameters
                'median_values', 'iqr_values', 'num_features', 'constant_features_mask',
                'clip_min_values', 'clip_max_values'
            ]
            for key in fitted_keys:
                config_copy.pop(key, None)
            
            stable_transforms.append(config_copy)
        
        return {
            "pipeline_type": "TransformPipeline",
            "transforms": stable_transforms,
        }

    def __len__(self) -> int:
        """Return the number of transforms in the pipeline."""
        return len(self.transforms)

    def __getitem__(self, index: int) -> Transform:
        """Get a transform by index."""
        return self.transforms[index]

    def __iter__(self):
        """Iterate over transforms."""
        return iter(self.transforms)

    def add_transform(self, transform: Transform) -> None:
        """
        Add a transform to the end of the pipeline.

        Args:
            transform: Transform to add

        Note:
            If the pipeline is already fitted, you'll need to refit it.
        """
        self.transforms.append(transform)
        if self.is_fitted:
            logger.warning(
                "Pipeline was fitted but a new transform was added. You may need to refit."
            )

    def remove_transform(self, index: int) -> Transform:
        """
        Remove and return a transform by index.

        Args:
            index: Index of transform to remove

        Returns:
            Removed transform

        Note:
            If the pipeline is already fitted, you'll need to refit it.
        """
        if index < 0 or index >= len(self.transforms):
            raise IndexError(f"Transform index {index} out of range")

        removed_transform = self.transforms.pop(index)
        if self.is_fitted:
            logger.warning(
                "Pipeline was fitted but a transform was removed. You may need to refit."
            )

        return removed_transform
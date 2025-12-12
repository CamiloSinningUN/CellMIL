"""
Base classes for feature transforms.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
import torch
from pathlib import Path
import json


class Transform(ABC):
    """Base class for all feature transforms."""

    def __init__(self, name: str):
        """
        Initialize the transform.

        Args:
            name: Name of the transform for identification
        """
        self.name = name

    @abstractmethod
    def transform(self, features: torch.Tensor) -> torch.Tensor:
        """
        Apply the transform to features.

        Args:
            features: Input features tensor of shape (n_instances, n_features)

        Returns:
            Transformed features tensor
        """
        pass

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """
        Get the configuration dictionary for this transform.

        Returns:
            Dictionary containing transform configuration
        """
        pass

    @classmethod
    @abstractmethod
    def from_config(cls, config: Dict[str, Any]) -> "Transform":
        """
        Create transform instance from configuration dictionary.

        Args:
            config: Configuration dictionary

        Returns:
            Transform instance
        """
        pass

    def save(self, path: Path) -> None:
        """
        Save the transform to disk.

        Args:
            path: Path to save the transform
        """
        config = self.get_config()
        config["transform_class"] = self.__class__.__name__

        with open(path, "w") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "Transform":
        """
        Load transform from disk.

        Args:
            path: Path to load the transform from

        Returns:
            Transform instance
        """
        with open(path, "r") as f:
            config = json.load(f)

        # Remove class name from config since it's not needed for instantiation
        config.pop("transform_class", None)

        return cls.from_config(config)


class FittableTransform(Transform):
    """Base class for transforms that need to be fitted on training data."""

    def __init__(self, name: str):
        """
        Initialize the fittable transform.

        Args:
            name: Name of the transform for identification
        """
        super().__init__(name)
        self.is_fitted = False

    @abstractmethod
    def fit(
        self, features: torch.Tensor, feature_names: Optional[List[str]] = None
    ) -> "FittableTransform":
        """
        Fit the transform on training data.

        Args:
            features: Training features tensor of shape (n_instances, n_features)
            feature_names: Optional list of feature names for logging

        Returns:
            Self for method chaining
        """
        pass

    def fit_transform(
        self, features: torch.Tensor, feature_names: Optional[List[str]] = None
    ) -> torch.Tensor:
        """
        Fit the transform and apply it to the features.

        Args:
            features: Features to fit and transform
            feature_names: Optional list of feature names for logging

        Returns:
            Transformed features
        """
        self.fit(features, feature_names)
        return self.transform(features)

    def transform(self, features: torch.Tensor) -> torch.Tensor:
        """
        Apply the transform to features.

        Args:
            features: Input features tensor

        Returns:
            Transformed features tensor

        Raises:
            RuntimeError: If transform hasn't been fitted yet
        """
        if not self.is_fitted:
            raise RuntimeError(
                f"Transform '{self.name}' must be fitted before transforming data"
            )

        return self._transform_impl(features)

    @abstractmethod
    def _transform_impl(self, features: torch.Tensor) -> torch.Tensor:
        """
        Implementation of the transform operation.

        Args:
            features: Input features tensor

        Returns:
            Transformed features tensor
        """
        pass
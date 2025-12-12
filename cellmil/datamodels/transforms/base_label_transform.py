"""
Base classes for label transforms.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Union
from pathlib import Path
import json


class LabelTransform(ABC):
    """Base class for all label transforms."""

    def __init__(self, name: str):
        """
        Initialize the transform.

        Args:
            name: Name of the transform for identification
        """
        self.name = name

    @abstractmethod
    def transform_labels(
        self, labels: Dict[str, Union[int, Tuple[float, int]]]
    ) -> Dict[str, Union[int, Tuple[float, int]]]:
        """
        Apply the transform to labels.

        Args:
            labels: Dictionary mapping slide IDs to labels
                   For classification: int labels
                   For survival: (duration, event) tuples

        Returns:
            Transformed labels dictionary
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
    def from_config(cls, config: Dict[str, Any]) -> "LabelTransform":
        """
        Create transform instance from configuration dictionary.

        Args:
            config: Configuration dictionary

        Returns:
            LabelTransform instance
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
    def load(cls, path: Path) -> "LabelTransform":
        """
        Load transform from disk.

        Args:
            path: Path to load the transform from

        Returns:
            LabelTransform instance
        """
        with open(path, "r") as f:
            config = json.load(f)

        # Remove class name from config since it's not needed for instantiation
        config.pop("transform_class", None)

        return cls.from_config(config)


class FittableLabelTransform(LabelTransform):
    """Base class for label transforms that need to be fitted on training data."""

    def __init__(self, name: str):
        """
        Initialize the fittable label transform.

        Args:
            name: Name of the transform for identification
        """
        super().__init__(name)
        self.is_fitted = False

    @abstractmethod
    def fit(
        self, labels: Dict[str, Union[int, Tuple[float, int]]], **kwargs: Any
    ) -> "FittableLabelTransform":
        """
        Fit the transform on training labels.

        Args:
            labels: Training labels dictionary mapping slide IDs to labels
            **kwargs: Additional keyword arguments for fitting

        Returns:
            Self for method chaining
        """
        pass

    def fit_transform(
        self, labels: Dict[str, Union[int, Tuple[float, int]]], **kwargs: Any
    ) -> Dict[str, Union[int, Tuple[float, int]]]:
        """
        Fit the transform and apply it to the labels.

        Args:
            labels: Labels to fit and transform
            **kwargs: Additional keyword arguments for fitting

        Returns:
            Transformed labels
        """
        self.fit(labels, **kwargs)
        return self.transform_labels(labels)

    def transform_labels(
        self, labels: Dict[str, Union[int, Tuple[float, int]]]
    ) -> Dict[str, Union[int, Tuple[float, int]]]:
        """
        Apply the transform to labels.

        Args:
            labels: Labels dictionary to transform

        Returns:
            Transformed labels dictionary

        Raises:
            RuntimeError: If transform hasn't been fitted yet
        """
        if not self.is_fitted:
            raise RuntimeError(
                f"LabelTransform '{self.name}' must be fitted before transforming data"
            )

        return self._transform_labels_impl(labels)

    @abstractmethod
    def _transform_labels_impl(
        self, labels: Dict[str, Union[int, Tuple[float, int]]]
    ) -> Dict[str, Union[int, Tuple[float, int]]]:
        """
        Implementation of the label transform operation.

        Args:
            labels: Input labels dictionary

        Returns:
            Transformed labels dictionary
        """
        pass

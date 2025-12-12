"""
Pipeline for composing multiple label transforms.
"""

from typing import List, Union, Dict, Any, Tuple
from pathlib import Path
import json
from .base_label_transform import LabelTransform, FittableLabelTransform


class LabelTransformPipeline:
    """
    Pipeline for applying multiple label transforms in sequence.

    This class manages a sequence of label transforms, ensuring that fittable
    transforms are properly fitted on training data before being applied.
    """

    def __init__(self, transforms: List[LabelTransform]):
        """
        Initialize the pipeline with a list of transforms.

        Args:
            transforms: List of LabelTransform instances to apply in sequence
        """
        self.transforms = transforms
        self.name = "label_pipeline"

    def fit(
        self, labels: Dict[str, Union[int, Tuple[float, int]]], **kwargs: Any
    ) -> "LabelTransformPipeline":
        """
        Fit all fittable transforms in the pipeline on training labels.

        Args:
            labels: Training labels dictionary
            **kwargs: Additional keyword arguments passed to each transform's fit method

        Returns:
            Self for method chaining
        """
        for transform in self.transforms:
            if isinstance(transform, FittableLabelTransform):
                transform.fit(labels, **kwargs)
                # Apply the transform to get updated labels for next transform
                labels = transform.transform_labels(labels)

        return self

    def transform_labels(
        self, labels: Dict[str, Union[int, Tuple[float, int]]]
    ) -> Dict[str, Union[int, Tuple[float, int]]]:
        """
        Apply all transforms in the pipeline sequentially.

        Args:
            labels: Labels dictionary to transform

        Returns:
            Transformed labels dictionary after applying all transforms
        """
        result = labels
        for transform in self.transforms:
            result = transform.transform_labels(result)  # type: ignore
        return result  # type: ignore

    def fit_transform(
        self, labels: Dict[str, Union[int, Tuple[float, int]]], **kwargs: Any
    ) -> Dict[str, Union[int, Tuple[float, int]]]:
        """
        Fit the pipeline and apply it to labels.

        Args:
            labels: Labels to fit and transform
            **kwargs: Additional keyword arguments for fitting

        Returns:
            Transformed labels
        """
        self.fit(labels, **kwargs)
        return self.transform_labels(labels)

    def get_config(self) -> Dict[str, Any]:
        """
        Get configuration for all transforms in the pipeline.

        Returns:
            Dictionary containing pipeline configuration
        """
        return {
            "name": self.name,
            "transforms": [
                {"transform_class": t.__class__.__name__, "config": t.get_config()}
                for t in self.transforms
            ],
        }

    def save(self, directory: Path) -> None:
        """
        Save the pipeline and all its transforms to disk.

        Args:
            directory: Directory to save the pipeline configuration and transforms
        """
        directory.mkdir(parents=True, exist_ok=True)

        # Save pipeline config
        pipeline_config = self.get_config()
        pipeline_path = directory / "pipeline.json"

        with open(pipeline_path, "w") as f:
            json.dump(pipeline_config, f, indent=2)

        # Save individual transforms
        for idx, transform in enumerate(self.transforms):
            transform_path = directory / f"transform_{idx}.json"
            transform.save(transform_path)

    @classmethod
    def load(cls, directory: Path) -> "LabelTransformPipeline":
        """
        Load a pipeline from disk.

        Args:
            directory: Directory containing the saved pipeline

        Returns:
            LabelTransformPipeline instance
        """
        pipeline_path = directory / "pipeline.json"

        with open(pipeline_path, "r") as f:
            pipeline_config = json.load(f)

        # Import transform classes
        from . import TimeDiscretizerTransform

        transform_classes = {
            "TimeDiscretizerTransform": TimeDiscretizerTransform,
        }

        # Load transforms
        transforms: List[LabelTransform] = []
        for idx, transform_info in enumerate(pipeline_config["transforms"]):
            transform_path = directory / f"transform_{idx}.json"

            transform_class_name = transform_info.get("transform_class")
            if transform_class_name not in transform_classes:
                raise ValueError(
                    f"Unknown label transform class: {transform_class_name}"
                )

            transform_class = transform_classes[transform_class_name]
            transform = transform_class.load(transform_path)
            transforms.append(transform)

        return cls(transforms)

    def __len__(self) -> int:
        """Return the number of transforms in the pipeline."""
        return len(self.transforms)

    def __getitem__(self, idx: int) -> LabelTransform:
        """Get a transform by index."""
        return self.transforms[idx]

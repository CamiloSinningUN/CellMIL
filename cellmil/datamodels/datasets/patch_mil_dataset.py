import pandas as pd
import torch
import json
import hashlib
from tqdm import tqdm
from pathlib import Path
from typing import List, Literal, Tuple, Union, Optional, Any, Dict
from torch.utils.data import Dataset
from cellmil.interfaces.FeatureExtractorConfig import (
    ExtractorType,
    FeatureExtractionType,
)
from cellmil.utils import logger
from ..transforms import Transform, TransformPipeline, LabelTransform, LabelTransformPipeline, FittableLabelTransform
from .utils import (
    wsl_preprocess,
    column_sanity_check,
    filter_split,
    preprocess_row,
    get_feature_path,
    weights_for_sampler,
)


class PatchMILDataset(Dataset[Tuple[torch.Tensor, int | Tuple[float, int]]]):
    """
    An optimized PyTorch Dataset for Patch-based MIL (Multiple Instance Learning) tasks.

    This dataset follows PyTorch best practices by preprocessing all data once during initialization
    and storing it efficiently. This provides significant speed improvements over the previous
    implementation by avoiding repeated feature loading in __getitem__.
    
    Returns:
        For classification tasks: Tuple[torch.Tensor, int] (features, label)
        For survival prediction tasks: Tuple[torch.Tensor, Tuple[float, int]] (features, (duration, event))
    """

    def __init__(
        self,
        root: Union[str, Path],
        label: Union[str, Tuple[str, str]],
        folder: Path,
        data: pd.DataFrame,
        extractor: ExtractorType,
        split: Literal["train", "val", "test", "all"] = "all",
        force_reload: bool = False,
        transforms: Optional[Union[Transform, TransformPipeline]] = None,
        label_transforms: Optional[Union[LabelTransform, LabelTransformPipeline]] = None,
    ):
        """
        Initialize the optimized Patch MIL dataset.

        Args:
            root: Root directory where the processed dataset will be cached
            label: Label for the dataset. Either:
                - A single string (e.g., "dcr") for classification tasks
                - A tuple of two strings (e.g., ("duration", "event")) for survival prediction tasks
            folder: Path to the dataset folder
            data: DataFrame containing metadata
            extractor: Feature extractor type (must be embedding type)
            split: Dataset split (train/val/test/all). Use "all" to include all data regardless of split
            force_reload: Whether to force reprocessing even if processed files exist
            transforms: Optional Transform or TransformPipeline to apply to features before returning them
            label_transforms: Optional LabelTransform or LabelTransformPipeline to apply to labels (e.g., TimeDiscretizerTransform for survival analysis)
        """
        self.root = Path(root)
        self.label = label
        self.folder = folder
        self.raw_data = data
        self.extractor = extractor
        self.split = split
        self.force_reload = force_reload
        self.transforms = transforms
        self.label_transforms = label_transforms

        if self.extractor not in FeatureExtractionType.Embedding:
            raise ValueError(
                f"Extractor {self.extractor} not supported for PatchMILDataset. Use embedding extractors."
            )

        # TODO: Make this configurable
        self.wsl = False
        # TODO ---

        # Data structures
        self.all_slides: List[str] = []  # All slides with features
        self.slides: List[str] = []      # Only slides with labels for this task
        self.labels: Dict[str, Union[int, Tuple[float, int]]] = {}

        # Preprocessed data storage
        self.features: Dict[str, torch.Tensor] = {}

        # Create root directory
        self.root.mkdir(parents=True, exist_ok=True)

        # Check if we need to process or can load from cache
        processed_path = self._get_processed_path()

        if self.force_reload or not processed_path.exists():
            logger.info("Processing patch dataset from scratch...")
            self._process_dataset()
            self._save_processed_data(processed_path)
        else:
            logger.info(f"Loading preprocessed patch dataset from {processed_path}")
            self._load_processed_data(processed_path)

    def get_num_labels(self) -> int:
        """
        Get the number of unique labels in the dataset.
        
        Note: For survival prediction tasks, this returns 0 as there are no discrete classes.
        """
        if self.labels:
            # Check if we have survival data by looking at the first label
            first_label = next(iter(self.labels.values()))
            if isinstance(first_label, tuple):
                # Survival data - no discrete classes
                return 0
        
        # Classification data
        return len(set(self.labels.values()))

    def _get_processed_path(self) -> Path:
        """Get the path for the processed dataset file."""

        config_dict: Dict[str, Any] = {
            "extractor": str(self.extractor),
            "split": self.split,
        }

        config_str = json.dumps(config_dict, sort_keys=True)
        config_hash = hashlib.md5(config_str.encode("utf-8")).hexdigest()[:8]

        logger.info(f"Dataset configuration:{config_dict}")
        logger.info(f"Patch dataset configuration hash: {config_hash}")

        return self.root / f"data_{self.split}_{config_hash}.pt"

    def _process_dataset(self) -> None:
        """Process the entire dataset and cache results."""
        try:
            if self.wsl:
                processed_data = wsl_preprocess(self.raw_data)
            else:
                processed_data = self.raw_data.copy()

            # Perform sanity check
            column_sanity_check(processed_data, self.label)

            # Filter by split type (unless split is "all")
            if self.split != "all":
                print("-"*50)
                print(f"Filtering data for split: {self.split}")
                processed_data = filter_split(processed_data, self.split)
            else:
                logger.info(f"Using all data: {len(processed_data)} slides")

            # Extract valid slides and labels
            logger.info("Validating patch slides...")
            for _, row in tqdm(
                processed_data.iterrows(),
                total=len(processed_data),
                desc="Validating slides",
            ):
                try:
                    slide_name, _ = preprocess_row(
                        row,
                        self.label,  # Still need to validate that label exists
                        self.folder,
                        self.extractor,
                    )

                    if slide_name is not None:
                        self.all_slides.append(slide_name)
                except Exception as e:
                    raise ValueError(f"Failed to validate row: {e}")

            logger.info(
                f"Found {len(self.all_slides)} valid patch slides out of {len(processed_data)} total slides"
            )

            if not self.all_slides:
                raise ValueError("No valid slides found")

            # Preprocess all features
            logger.info("Preprocessing patch features for all slides...")
            valid_slides: list[str] = []
            for slide_name in tqdm(self.all_slides, desc="Processing patch features"):
                try:
                    features = self._preprocess_slide_features(slide_name)
                    if features is not None:
                        self.features[slide_name] = features
                        valid_slides.append(slide_name)
                except Exception as e:
                    logger.error(f"Failed to preprocess slide {slide_name}: {e}")

            # Update all_slides list to only include those with successfully processed features
            self.all_slides = valid_slides
            logger.info(f"Successfully preprocessed {len(self.features)} patch slides")

            # Now get labels and filter slides to only those with labels
            self.labels = self._get_labels()
            self.slides = [slide for slide in self.all_slides if slide in self.labels]
            logger.info(
                f"Extracted {len(self.labels)} labels for {len(self.slides)} slides with labels"
            )

            if not self.slides:
                raise ValueError("No slides found with labels for this task")

        except Exception as e:
            logger.error(f"Failed to process patch dataset: {e}")
            raise ValueError(f"Failed to process patch dataset: {e}")

    def _get_labels(self) -> Dict[str, Union[int, Tuple[float, int]]]:
        """
        Extract labels from the DataFrame based on current configuration.
        This allows labels to be extracted fresh without being cached with features.
        
        Returns:
            Dictionary mapping slide names to labels (either int for classification or (duration, event) for survival)
        """
        try:
            # Process the DataFrame same way as in process()
            if self.wsl:
                processed_data = wsl_preprocess(self.raw_data)
            else:
                processed_data = self.raw_data.copy()

            # Filter by split type (unless split is "all")
            if self.split != "all":
                processed_data = filter_split(processed_data, self.split)

            # Extract labels for valid slides as a dictionary
            labels: Dict[str, Union[int, Tuple[float, int]]] = {}
            for _, row in tqdm(processed_data.iterrows(), total=len(processed_data), desc="Extracting labels"):
                try:
                    slide_name, label_value = preprocess_row(
                        row,
                        self.label,
                        self.folder,
                        self.extractor,
                        do_validate_features=False
                    )

                    if slide_name in self.all_slides and label_value is not None:
                        labels[slide_name] = label_value

                except Exception as e:
                    raise ValueError(f"Failed to extract label from row: {e}")

            return labels

        except Exception as e:
            logger.error(f"Failed to extract labels: {e}")
            raise

    def _preprocess_slide_features(self, slide_name: str) -> Optional[torch.Tensor]:
        """
        Preprocess features for a single slide.

        Args:
            slide_name: Name of the slide to process

        Returns:
            Preprocessed features tensor or None if processing fails
        """
        try:
            features = torch.load(
                get_feature_path(self.folder, slide_name, self.extractor),
                map_location="cpu",
                weights_only=False,
            )["features"]

            if features is None or features.numel() == 0:
                raise ValueError(
                    f"No features found for slide {slide_name} with extractor {self.extractor}"
                )
                return None

            return features

        except Exception as e:
            logger.error(f"Error preprocessing slide {slide_name}: {e}")
            return None

    def _save_processed_data(self, path: Path) -> None:
        """Save preprocessed data to disk (all slides with features, label-independent)."""
        data_dict: dict[str, Any] = {
            "all_slides": self.all_slides,
            "features": self.features,
        }

        torch.save(data_dict, path)
        logger.info(f"Saved preprocessed patch dataset to {path}")

    def _load_processed_data(self, path: Path) -> None:
        """Load preprocessed data from disk and extract labels for current task."""
        data_dict = torch.load(path, map_location="cpu", weights_only=False)

        self.all_slides = data_dict["all_slides"]
        self.features = data_dict.get("features", {})
        
        # Extract labels for current task and filter slides
        self.labels = self._get_labels()
        self.slides = [slide for slide in self.all_slides if slide in self.labels]

        logger.info(f"Loaded {len(self.all_slides)} total slides, {len(self.slides)} with labels for current task")

    def get_weights_for_sampler(self) -> torch.Tensor:
        """
        Get weights for WeightedRandomSampler to handle class imbalance.
        
        Note: Only applicable for classification tasks. For survival prediction,
        returns uniform weights.
        """
        # Convert dictionary values to list in the same order as slides
        labels_list = [self.labels[slide] for slide in self.slides]
        
        # Check if we have survival data
        if labels_list and isinstance(labels_list[0], tuple):
            # Survival data - return uniform weights
            logger.warning("Uniform weights used for survival prediction tasks")
            return torch.ones(len(labels_list), dtype=torch.float32)
        
        # Classification data
        return weights_for_sampler(labels_list)  # type: ignore

    def create_subset(self, indices: List[int]) -> "PatchMILDataset":
        """
        Create a subset of the dataset using the specified indices.

        This is useful for creating train/val/test splits when using split="all".
        The subset will share the same cached features but only include the specified samples.

        Args:
            indices: List of indices to include in the subset

        Returns:
            New PatchMILDataset instance containing only the specified samples

        Raises:
            ValueError: If any index is out of range
        """
        if not indices:
            raise ValueError("Indices list cannot be empty")

        max_idx = len(self.slides) - 1
        invalid_indices = [idx for idx in indices if idx < 0 or idx > max_idx]
        if invalid_indices:
            raise ValueError(
                f"Invalid indices {invalid_indices}. Valid range is 0-{max_idx}"
            )

        # Create a new instance with the same configuration
        subset = PatchMILDataset.__new__(PatchMILDataset)

        # Copy configuration
        subset.root = self.root
        subset.label = self.label
        subset.folder = self.folder
        subset.raw_data = self.raw_data
        subset.extractor = self.extractor
        subset.split = self.split
        subset.force_reload = self.force_reload
        subset.transforms = self.transforms
        subset.label_transforms = self.label_transforms
        subset.wsl = self.wsl

        # Create subset data
        subset.slides = [self.slides[i] for i in indices]
        subset.all_slides = self.all_slides  # Share the same all_slides

        # Share the same features cache
        subset.features = self.features

        # Extract labels fresh for the subset
        subset.labels = subset._get_labels()

        logger.info(
            f"Created subset with {len(subset.slides)} samples from {len(self.slides)} total samples"
        )

        return subset

    def __len__(self) -> int:
        return len(self.slides)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Union[int, Tuple[float, int]]]:
        """
        Get a sample from the dataset.

        Args:
            index: Index of the sample to retrieve

        Returns:
            For classification tasks:
                Tuple of (features, label) where features is a tensor of shape (n_patches, n_features)
                and label is an int.
            
            For survival prediction tasks:
                Tuple of (features, (duration, event)) where features is a tensor of shape (n_patches, n_features),
                duration is a float, and event is an int.
        """
        slide_name = self.slides[index]
        label = self.labels[slide_name]

        try:
            # Get features from cache
            features = self.features[
                slide_name
            ].clone()  # Clone to avoid modifying cached data

            # Apply transforms if provided
            if self.transforms is not None:
                features = self.transforms.transform(features)
            
            # Apply label transforms if provided
            if self.label_transforms is not None:
                labels_dict = {slide_name: label}
                transformed = self.label_transforms.transform_labels(labels_dict)
                label = transformed[slide_name]

            return features, label
        except KeyError:
            logger.error(f"No features found for slide {slide_name}")
            raise ValueError(f"No features found for slide {slide_name}")
        except Exception as e:
            logger.error(f"Failed to get item for index {index}: {e}")
            raise ValueError(f"Failed to get item for index {index}: {e}")

    def get_normalization_params(self) -> None:
        """Return None for compatibility - patch datasets don't use normalization."""
        return None

    def get_correlation_mask(self) -> None:
        """Return None for compatibility - patch datasets don't use correlation filtering."""
        return None
    
    def create_train_val_datasets(
        self, 
        train_indices: List[int], 
        val_indices: List[int],
        transforms: Optional[Union[Transform, TransformPipeline]] = None,
        label_transforms: Optional[Union[LabelTransform, LabelTransformPipeline]] = None
    ) -> Tuple["PatchMILDataset", "PatchMILDataset"]:
        """
        Create train and validation datasets with transforms fitted on training data only.
        
        This function prevents data leakage by ensuring transforms are fitted only on
        the training set before being applied to both training and validation sets.
        
        Args:
            train_indices: List of indices for training data
            val_indices: List of indices for validation data
            transforms: Optional Transform or TransformPipeline for features (fitted on training data)
            label_transforms: Optional LabelTransform or LabelTransformPipeline for labels (e.g., TimeDiscretizerTransform for survival analysis)
            
        Returns:
            Tuple of (train_dataset, val_dataset) with properly fitted transforms
            
        Raises:
            ValueError: If indices are invalid or transforms cannot be fitted
        """
        
        if transforms is not None:
            logger.warning("Transforms are ignored...")
        
        # Create train dataset
        train_dataset = self.create_subset(train_indices)
        
        # Fit label transforms on training labels only
        if label_transforms is not None:
            
            # Get training labels
            train_labels = {train_dataset.slides[i]: train_dataset.labels[train_dataset.slides[i]] 
                           for i in range(len(train_dataset.slides))}
            
            # Fit label transforms on training data
            if isinstance(label_transforms, FittableLabelTransform):
                label_transforms.fit(train_labels)
                logger.info(f"Fitted label transform on {len(train_labels)} training labels")
            elif isinstance(label_transforms, LabelTransformPipeline):
                label_transforms.fit(train_labels)
                logger.info(f"Fitted label transform pipeline on {len(train_labels)} training labels")
            
            # Apply to both datasets
            train_dataset.label_transforms = label_transforms
        
        # Create validation dataset with the same fitted transforms
        val_dataset = self.create_subset(val_indices)
        
        # Share the fitted label transforms with validation dataset
        if label_transforms is not None:
            val_dataset.label_transforms = label_transforms

        
        logger.info(f"Created train dataset with {len(train_dataset)} samples and val dataset with {len(val_dataset)} samples")
        
        return train_dataset, val_dataset

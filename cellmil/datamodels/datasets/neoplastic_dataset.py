import pandas as pd
import torch
from tqdm import tqdm
from pathlib import Path
from typing import List, Literal, Tuple, Union, Optional, Dict
from torch.utils.data import Dataset
from cellmil.interfaces.FeatureExtractorConfig import ExtractorType
from cellmil.interfaces.GraphCreatorConfig import GraphCreatorType
from cellmil.interfaces.CellSegmenterConfig import ModelType
from cellmil.utils import logger
from .utils import (
    wsl_preprocess,
    filter_split,
    get_cell_types,
    get_cell_features,
    compute_normalization,
    correlation_filter,
)


class NeoplasticDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    """
    A PyTorch Dataset for neoplastic cell classification tasks.
    This dataset treats each individual cell as a sample, creating a cell-level dataset
    where each item is a single cell with its features and binary neoplastic label.

    The target is binary:
    - Neoplastic cells (type 1): label = 1
    - All other cell types (types 2, 3, 4, 5): label = 0

    The dataset concatenates all cells from all slides in the split, hiding the
    slide-level structure from the model.
    """

    def __init__(
        self,
        folder: Path,
        data: pd.DataFrame,
        extractor: Union[ExtractorType, List[ExtractorType]],
        graph_creator: GraphCreatorType | None = None,
        segmentation_model: ModelType | None = None,
        split: Literal["train", "val", "test"] = "train",
        max_workers: int = 8,
        correlation_filter: bool = False,
        correlation_threshold: float = 0.9,
        correlation_mask: Optional[torch.Tensor] = None,
        normalize_feature: bool = False,
        normalization_params: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        """
        Initialize the dataset with data and cell-level neoplastic labels.

        Args:
            folder: Path to the dataset folder
            data: DataFrame containing metadata
            extractor: Feature extractor type or list of types to use for feature extraction.
            graph_creator: Optional graph creator type, needed for some extractors
            segmentation_model: Segmentation model type, required for cell type information
            split: Dataset split (train/val/test)
            max_workers: Maximum number of threads for parallel processing
            correlation_filter: Whether to apply correlation filtering to remove highly correlated features
            correlation_threshold: Correlation threshold above which features will be filtered (default: 0.9)
            correlation_mask: Optional tensor mask to apply correlation filtering
            normalize_feature: Whether to apply min-max normalization to features
            normalization_params: Optional tuple of (min_values, max_values) for normalization
        """
        self.folder = folder
        self.data = data
        self.extractor = extractor
        self.graph_creator = graph_creator
        self.segmentation_model = segmentation_model
        self.split = split
        self.max_workers = max_workers
        self.correlation_filter = correlation_filter
        self.correlation_threshold = correlation_threshold
        self.correlation_mask = correlation_mask
        self.normalize_feature = normalize_feature
        self.normalization_params = normalization_params

        # Validate that segmentation model is provided for cell type information
        if self.segmentation_model not in ["cellvit", "hovernet"]:
            raise ValueError(
                f"Neoplastic classification requires 'cellvit' or 'hovernet' segmentation model. "
                f"Got '{self.segmentation_model}'"
            )

        # TODO: Make this configurable
        self.wsl = False
        # TODO ---

        # Cell-level data structures
        self.cell_features: List[torch.Tensor] = []  # Features for each cell
        self.cell_labels: List[int] = []  # Binary labels for each cell
        self.slide_names: List[
            str
        ] = []  # Keep track of which slide each cell came from

        self._read_data()

    def _read_data(self) -> None:
        """Read the data specified in the configuration."""
        try:
            if self.wsl:
                self.data = wsl_preprocess(self.data)

            # Perform sanity check - we don't need a specific label column for neoplastic classification
            if self.data.empty:
                raise ValueError("Data is empty")

            required_columns = ["FULL_PATH", "SPLIT"]
            for col in required_columns:
                if col not in self.data.columns:
                    raise ValueError(f"Required column '{col}' not found in data")

            # Filter by split type
            self.data = filter_split(self.data, self.split)

            # Process slides and collect all cells
            slides_processed: list[str] = []
            for _, (_, row) in tqdm(
                enumerate(self.data.iterrows()),
                total=len(self.data),
                desc="Loading slides for neoplastic classification",
            ):
                slide_name = self._preprocess_row_for_neoplastic(row)
                if slide_name is not None:
                    slides_processed.append(slide_name)

            logger.info(
                f"Found {len(slides_processed)} valid slides with non-empty features out of {len(self.data)} total slides"
            )

            # Now process each slide to extract individual cells
            logger.info("Extracting individual cells from slides...")
            for slide_name in tqdm(slides_processed, desc="Processing cells"):
                # Get features and cell types for this slide
                features, cell_indices, _ = get_cell_features(
                    self.folder,
                    slide_name,
                    self.extractor,
                    self.graph_creator,
                    self.segmentation_model,
                )

                if features is None or cell_indices is None:
                    logger.warning(
                        f"No features or cell indices found for slide {slide_name}"
                    )
                    continue

                # Get cell types for this slide
                if self.segmentation_model is not None:
                    cell_types = get_cell_types(
                        self.folder, slide_name, self.segmentation_model
                    )
                else:
                    logger.warning(
                        f"No segmentation model provided for slide {slide_name}"
                    )
                    continue

                if not cell_types:
                    logger.warning(f"No cell types found for slide {slide_name}")
                    continue

                # Extract individual cells
                for cell_id, tensor_idx in cell_indices.items():
                    if cell_id in cell_types:
                        # Get cell features (single row from features tensor)
                        cell_feature = features[
                            tensor_idx : tensor_idx + 1
                        ]  # Shape: (1, n_features)
                        cell_feature = cell_feature.squeeze(0)  # Shape: (n_features,)

                        # Get cell label (binary: 1 for neoplastic, 0 for others)
                        cell_type = cell_types[cell_id]
                        cell_label = 1 if cell_type == 1 else 0  # Type 1 is neoplastic

                        # Store cell data
                        self.cell_features.append(cell_feature)
                        self.cell_labels.append(cell_label)
                        self.slide_names.append(slide_name)

            logger.info(
                f"Loaded {len(self.cell_features)} individual cells from {len(slides_processed)} slides"
            )

            # Compute correlation filter mask if enabled
            if (
                self.correlation_filter
                and len(self.cell_features) > 0
                and self.split == "train"
            ):
                self._compute_correlation_filter()

            # Compute normalization parameters if enabled
            if (
                self.normalize_feature
                and len(self.cell_features) > 0
                and self.split == "train"
            ):
                self._compute_normalization_params()

        except Exception as e:
            logger.error(f"Failed to read data: {e}")
            raise ValueError(f"Failed to read data: {e}")

    def _preprocess_row_for_neoplastic(self, row: pd.Series) -> Optional[str]:
        """
        Process a single slide row to extract slide name and validate features.
        Modified version that doesn't require a label column.

        Args:
            row: A pandas Series representing a row from the Excel file

        Returns:
            slide_name if valid, None otherwise
        """
        try:
            # Extract slide name from file path
            file_path = Path(row["FULL_PATH"])
            slide_name = self._extract_slide_name(file_path)

            # Validate that features exist for this slide
            if self._validate_features(self.folder, slide_name):
                return slide_name
            else:
                logger.warning(f"No valid features found for slide {slide_name}")
                return None

        except Exception as e:
            logger.warning(f"Error processing slide row: {e}")
            return None

    def _extract_slide_name(self, file_path: Path) -> str:
        """Extract slide name from file path."""
        return file_path.stem

    def _validate_features(self, folder: Path, slide_name: str) -> bool:
        """
        Validate that features exist for the given slide.
        """
        try:
            features, cell_indices, _ = get_cell_features(
                folder,
                slide_name,
                self.extractor,
                self.graph_creator,
                self.segmentation_model,
            )
            return (
                features is not None
                and cell_indices is not None
                and len(cell_indices) > 0
            )
        except Exception as e:
            logger.warning(f"Error validating features for slide {slide_name}: {e}")
            return False

    def _create_neoplastic_labels(
        self, cell_types: Dict[int, int], cell_indices: Dict[int, int]
    ) -> torch.Tensor:
        """
        Create binary neoplastic labels for cells.

        Args:
            cell_types: Dictionary mapping cell_id to cell_type
            cell_indices: Dictionary mapping cell_id to tensor index

        Returns:
            Tensor of shape (n_cells,) with binary labels:
            - 1 for neoplastic cells (type 1)
            - 0 for all other cell types (types 2, 3, 4, 5)
        """
        n_cells = len(cell_indices)
        labels = torch.zeros(n_cells, dtype=torch.float32)

        for cell_id, tensor_idx in cell_indices.items():
            cell_type = cell_types.get(cell_id, 0)  # Default to 0 if not found
            # Neoplastic cells have type 1, all others get label 0
            if cell_type == 1:  # Neoplastic
                labels[tensor_idx] = 1.0
            # All other types (2, 3, 4, 5) remain 0

        return labels

    def _compute_correlation_filter(self) -> None:
        """
        Compute correlation filter mask based on training data.
        Features with correlation > threshold will have one feature removed.
        """
        logger.info("Loading features to compute correlation matrix...")

        if len(self.cell_features) == 0:
            logger.warning("No cell features available for correlation computation")
            return

        # Use a sample of cells for correlation computation to save memory
        sample_size = min(len(self.cell_features), 50000)
        sampled_indices = torch.randperm(len(self.cell_features))[:sample_size]

        sampled_features = [self.cell_features[i] for i in sampled_indices]

        logger.info(f"Using {len(sampled_features)} cells for correlation computation")

        # Stack features into a tensor
        combined_features = torch.stack(
            sampled_features, dim=0
        )  # Shape: (n_cells, n_features)

        keep_mask, non_constant_mask = correlation_filter(
            combined_features, self.correlation_threshold
        )
        self.correlation_mask = keep_mask

        logger.info(
            f"Correlation filtering: kept {keep_mask.sum()} out of {len(keep_mask)} features"
        )

        # Final memory cleanup
        del non_constant_mask

    def _compute_normalization_params(self) -> None:
        """
        Compute min-max normalization parameters based on training data.
        These parameters can be used to normalize validation and test sets consistently.
        """
        logger.info("Loading features to compute normalization parameters...")

        if len(self.cell_features) == 0:
            logger.warning("No cell features available for normalization computation")
            return

        # Use a sample of cells for normalization computation to save memory
        sample_size = min(len(self.cell_features), 100000)
        sampled_indices = torch.randperm(len(self.cell_features))[:sample_size]

        sampled_features = [self.cell_features[i] for i in sampled_indices]

        logger.info(
            f"Using {len(sampled_features)} cells for normalization computation"
        )

        # Stack features into a tensor
        combined_features = torch.stack(
            sampled_features, dim=0
        )  # Shape: (n_cells, n_features)

        min_values, max_values = compute_normalization(combined_features)
        self.normalization_params = (min_values, max_values)

        logger.info(
            f"Computed normalization parameters: min range = {min_values.min():.6f}, max range = {max_values.max():.6f}"
        )

    def get_normalization_params(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get the normalization parameters (min, max) computed from training data.

        Returns:
            Tuple of (min_values, max_values) tensors, or None if not computed.
        """
        return self.normalization_params

    def get_correlation_mask(self):
        """
        Get the mask for features retained after correlation filtering.

        Returns:
            A boolean tensor indicating which features are kept.
        """
        return self.correlation_mask

    def get_class_distribution(self) -> Dict[str, int]:
        """
        Get the distribution of neoplastic vs non-neoplastic cells.

        Returns:
            Dictionary with counts for each class.
        """
        neoplastic_count = sum(1 for label in self.cell_labels if label == 1)
        non_neoplastic_count = sum(1 for label in self.cell_labels if label == 0)

        return {
            "neoplastic": neoplastic_count,
            "non_neoplastic": non_neoplastic_count,
            "total": neoplastic_count + non_neoplastic_count,
        }

    def get_weights_for_sampler(self) -> torch.Tensor:
        """
        Compute weights for WeightedRandomSampler to handle class imbalance.
        Since this is cell-level classification, weights are computed based on
        the number of neoplastic vs non-neoplastic cells.

        Returns:
            torch.Tensor: Weights for each cell in the dataset, with shape (len(dataset),).
        """
        if len(self.cell_labels) == 0:
            logger.warning("No cell labels found for weight computation")
            return torch.ones(1, dtype=torch.float32)

        # Get class distribution
        class_dist = self.get_class_distribution()
        total_neoplastic = class_dist["neoplastic"]
        total_non_neoplastic = class_dist["non_neoplastic"]

        if total_neoplastic == 0 or total_non_neoplastic == 0:
            logger.warning("One class has no samples, using uniform weights")
            return torch.ones(len(self.cell_labels), dtype=torch.float32)

        # Compute weight for each cell: inverse of its class frequency
        weights = torch.zeros(len(self.cell_labels), dtype=torch.float32)

        for i, label in enumerate(self.cell_labels):
            if label == 1:  # Neoplastic
                weights[i] = 1.0 / total_neoplastic
            else:  # Non-neoplastic
                weights[i] = 1.0 / total_non_neoplastic

        # Normalize weights
        weights = weights * len(weights) / weights.sum()

        logger.info("Computed sampling weights for neoplastic classification:")
        logger.info(f"  Total neoplastic cells: {total_neoplastic}")
        logger.info(f"  Total non-neoplastic cells: {total_non_neoplastic}")
        logger.info(
            f"  Class balance ratio: {total_neoplastic / (total_neoplastic + total_non_neoplastic):.4f}"
        )

        return weights

    def __len__(self) -> int:
        """
        Return the number of cells in the dataset.
        """
        return len(self.cell_features)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset.

        Args:
            idx: Index of the cell to retrieve

        Returns:
            Tuple of (features, label) where:
            - features is a tensor of shape (n_features,) for a single cell
            - label is an int with binary neoplastic label (0 or 1)
        """
        if idx >= len(self.cell_features):
            raise IndexError(
                f"Index {idx} out of range for dataset of size {len(self.cell_features)}"
            )

        # Get cell features
        features = self.cell_features[idx]  # Shape: (n_features,)

        # Get cell label
        label = torch.tensor([self.cell_labels[idx]], dtype=torch.long)  # Binary: 0 or 1

        # Apply correlation filter mask if available
        if self.correlation_mask is not None:
            if features.size(0) == len(self.correlation_mask):
                features = features[self.correlation_mask]
            else:
                logger.warning(
                    f"Feature dimension mismatch for cell {idx}: expected {len(self.correlation_mask)}, got {features.size(0)}"
                )

        # Apply normalization if enabled and parameters are available
        if self.normalize_feature and self.normalization_params is not None:
            min_values, max_values = self.normalization_params

            # Ensure dimensions match
            if features.size(0) == min_values.size(0) == max_values.size(0):
                # Avoid division by zero
                range_values = max_values - min_values
                range_values = torch.where(
                    range_values == 0, torch.ones_like(range_values), range_values
                )
                features = (features - min_values) / range_values
            else:
                logger.warning(
                    f"Normalization dimension mismatch for cell {idx}: "
                    f"features={features.size(0)}, min_values={min_values.size(0)}, max_values={max_values.size(0)}"
                )

        return features, label

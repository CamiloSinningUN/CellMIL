"""
CellTypeDataset: A PyTorch Dataset for multi-class cell type classification.

This dataset treats each individual cell as a sample, creating a cell-level dataset
where each item is a single cell with its features and one-hot encoded cell type label.

Supports label smoothing for specific cell types to handle annotation uncertainty.
"""

import pandas as pd
import torch
import json
import hashlib
from tqdm import tqdm
from pathlib import Path
from typing import List, Literal, Tuple, Union, Optional, Dict, Any
from torch.utils.data import Dataset

from cellmil.interfaces.FeatureExtractorConfig import ExtractorType
from cellmil.interfaces.GraphCreatorConfig import GraphCreatorType
from cellmil.interfaces.CellSegmenterConfig import ModelType, TYPE_NUCLEI_DICT
from cellmil.utils import logger
from ..transforms import (
    TransformPipeline,
    Transform,
)
from .utils import (
    wsl_preprocess,
    filter_split,
    get_cell_types,
    get_cell_features,
)


# Number of cell types (Neoplastic, Inflammatory, Connective, Dead, Epithelial)
NUM_CELL_TYPES = len(TYPE_NUCLEI_DICT)


class CellTypeDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    """
    A PyTorch Dataset for multi-class cell type classification.

    This dataset treats each individual cell as a sample, creating a cell-level dataset
    where each item is a single cell with its features and one-hot encoded cell type label.

    The labels are one-hot encoded for the 5 cell types:
    - Type 1: Neoplastic
    - Type 2: Inflammatory
    - Type 3: Connective
    - Type 4: Dead
    - Type 5: Epithelial

    Supports label smoothing for specific cell types to handle annotation uncertainty.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (features, label) where:
            - features is a tensor of shape (n_features,) for a single cell
            - label is a one-hot encoded tensor of shape (n_cell_types,) with optional label smoothing
    """

    def __init__(
        self,
        root: Union[str, Path],
        folder: Path,
        data: pd.DataFrame,
        extractor: Union[ExtractorType, List[ExtractorType]],
        graph_creator: GraphCreatorType | None = None,
        segmentation_model: ModelType | None = None,
        split: Literal["train", "val", "test", "all"] = "all",
        transforms: Optional[TransformPipeline | Transform] = None,
        cell_types_to_keep: Optional[List[str]] = None,
        label_smoothing: Optional[Union[float, Dict[str, float]]] = None,
        max_workers: int = 8,
        force_reload: bool = False,
    ):
        """
        Initialize the CellTypeDataset.

        Args:
            root: Root directory where the processed dataset will be cached
            folder: Path to the dataset folder containing slide data
            data: DataFrame containing metadata with at least 'FULL_PATH' and 'SPLIT' columns
            extractor: Feature extractor type or list of types to use for feature extraction
            graph_creator: Optional graph creator type, needed for some extractors
            segmentation_model: Segmentation model type ('cellvit' or 'hovernet'), required for cell type info
            split: Dataset split ('train', 'val', 'test', or 'all')
            transforms: Optional TransformPipeline or Transform to apply to features at getitem time
            cell_types_to_keep: Optional list of cell type names to keep (e.g., ["Neoplastic", "Connective"]).
                Valid names: "Neoplastic", "Inflammatory", "Connective", "Dead", "Epithelial" (case-insensitive).
                If provided, only cells of these types will be included.
            label_smoothing: Label smoothing configuration. Can be:
                - None or 0.0: No smoothing applied
                - float (0.0 to 1.0): Same smoothing value applied to all cell types
                - Dict[str, float]: Custom smoothing value for each cell type, e.g.,
                    {"Neoplastic": 0.0, "Inflammatory": 0.1, "Dead": 0.2, "Epithelial": 0.15}
                  Cell types not in the dict will have no smoothing applied.
            max_workers: Maximum number of threads for parallel processing
            force_reload: Whether to force reprocessing even if processed files exist
        """
        self.root = Path(root)
        self.folder = folder
        self.raw_data = data
        self.extractor = extractor
        self.graph_creator = graph_creator
        self.segmentation_model = segmentation_model
        self.split = split
        self.max_workers = max_workers
        self.force_reload = force_reload
        self.transforms = transforms

        # Validate segmentation model
        if self.segmentation_model not in ["cellvit", "hovernet"]:
            raise ValueError(
                f"Cell type classification requires 'cellvit' or 'hovernet' segmentation model. "
                f"Got '{self.segmentation_model}'"
            )

        # Process and validate label smoothing
        self.label_smoothing_per_type = self._process_label_smoothing(label_smoothing)

        # Convert cell type names to indices (1-based as in TYPE_NUCLEI_DICT)
        self.cell_types_to_keep_indices = self._cell_type_names_to_indices(
            cell_types_to_keep
        )

        # TODO: Make this configurable
        self.wsl = False

        # Data structures
        self.all_slides: List[str] = []  # All slides with features
        self.slides: List[str] = []  # Slides after filtering

        # Cell-level data (populated after processing)
        self.cell_features: Dict[
            str, torch.Tensor
        ] = {}  # slide_name -> features tensor
        self.cell_types: Dict[
            str, torch.Tensor
        ] = {}  # slide_name -> cell types tensor (1-based indices)
        self.cell_slide_indices: Dict[
            str, List[int]
        ] = {}  # slide_name -> list of global indices

        # Flat list for indexing
        self._global_indices: List[
            Tuple[str, int]
        ] = []  # (slide_name, local_idx) for each cell

        # Create root directory
        self.root.mkdir(parents=True, exist_ok=True)

        # Check if we need to process or can load from cache
        data_path = self._get_data_path()

        if self.force_reload or not data_path.exists():
            logger.info("Processing dataset from scratch...")
            self._process_dataset()
            self._save_data(data_path)
        else:
            logger.info(f"Loading preprocessed dataset from {data_path}")
            self._load_data(data_path)

    def _process_label_smoothing(
        self, label_smoothing: Optional[Union[float, Dict[str, float]]]
    ) -> Dict[int, float]:
        """
        Process label smoothing configuration into a per-type dictionary.

        Args:
            label_smoothing: Either None, a float, or a dict mapping cell type names to smoothing values

        Returns:
            Dictionary mapping cell type indices (1-based) to smoothing values
        """
        # Default: no smoothing for any type
        smoothing_per_type: Dict[int, float] = {
            i: 0.0 for i in range(1, NUM_CELL_TYPES + 1)
        }

        if label_smoothing is None:
            return smoothing_per_type

        if isinstance(label_smoothing, (int, float)):
            # Single value: apply to all types
            smoothing_value = float(label_smoothing)
            if not 0.0 <= smoothing_value <= 1.0:
                raise ValueError(
                    f"label_smoothing must be between 0.0 and 1.0. Got {smoothing_value}"
                )
            for i in range(1, NUM_CELL_TYPES + 1):
                smoothing_per_type[i] = smoothing_value
        elif isinstance(label_smoothing, dict):  # type: ignore
            # Dict: custom value per type
            name_to_idx: Dict[str, int] = {
                v.lower(): k for k, v in TYPE_NUCLEI_DICT.items()
            }

            for type_name, smoothing_value in label_smoothing.items():
                type_name_lower = type_name.lower()
                if type_name_lower not in name_to_idx:
                    valid_names = list(TYPE_NUCLEI_DICT.values())
                    raise ValueError(
                        f"Invalid cell type name in label_smoothing: '{type_name}'. "
                        f"Valid names are: {valid_names}"
                    )
                if not 0.0 <= smoothing_value <= 1.0:
                    raise ValueError(
                        f"label_smoothing value for '{type_name}' must be between 0.0 and 1.0. "
                        f"Got {smoothing_value}"
                    )
                type_idx = name_to_idx[type_name_lower]
                smoothing_per_type[type_idx] = smoothing_value
        else:
            raise TypeError(
                f"label_smoothing must be None, float, or Dict[str, float]. "
                f"Got {type(label_smoothing)}"
            )

        return smoothing_per_type

    def _cell_type_names_to_indices(
        self, cell_type_names: Optional[List[str]]
    ) -> Optional[List[int]]:
        """
        Convert cell type names to their corresponding indices.

        Args:
            cell_type_names: List of cell type names (case-insensitive)

        Returns:
            List of cell type indices (1-based as in TYPE_NUCLEI_DICT), or None if input is None
        """
        if cell_type_names is None:
            return None

        # Create reverse mapping from name to index
        name_to_idx: Dict[str, int] = {
            v.lower(): k for k, v in TYPE_NUCLEI_DICT.items()
        }

        indices: List[int] = []
        for name in cell_type_names:
            name_lower = name.lower()
            if name_lower not in name_to_idx:
                valid_names = list(TYPE_NUCLEI_DICT.values())
                raise ValueError(
                    f"Invalid cell type name: '{name}'. Valid names are: {valid_names}"
                )
            indices.append(name_to_idx[name_lower])

        return indices

    def _get_data_path(self) -> Path:
        """Get the path for the processed dataset file."""
        # Create a hash of the configuration to ensure cache invalidation
        extractor_str = (
            json.dumps(self.extractor, sort_keys=True)
            if isinstance(self.extractor, list)
            else str(self.extractor)
        )

        config_dict: Dict[str, Any] = {
            "extractor": extractor_str,
            "graph_creator": self.graph_creator,
            "segmentation_model": self.segmentation_model,
            "split": self.split,
            "cell_types_to_keep": sorted(self.cell_types_to_keep_indices)
            if self.cell_types_to_keep_indices
            else None,
        }

        config_str = json.dumps(config_dict, sort_keys=True)
        config_hash = hashlib.md5(config_str.encode("utf-8")).hexdigest()[:8]

        logger.info(f"Dataset configuration: {config_dict}")
        logger.info(f"Dataset configuration hash: {config_hash}")

        return self.root / f"celltype_{self.split}_{config_hash}.pt"

    def _process_dataset(self) -> None:
        """Process the entire dataset and cache results."""
        try:
            if self.wsl:
                processed_data = wsl_preprocess(self.raw_data)
            else:
                processed_data = self.raw_data.copy()

            # Validate required columns
            if processed_data.empty:
                raise ValueError("Data is empty")

            required_columns = (
                ["FULL_PATH", "SPLIT"] if self.split != "all" else ["FULL_PATH"]
            )
            for col in required_columns:
                if col not in processed_data.columns:
                    raise ValueError(f"Required column '{col}' not found in data")

            # Filter by split type
            if self.split != "all":
                processed_data = filter_split(processed_data, self.split)
            else:
                logger.info(f"Using all data: {len(processed_data)} slides")

            # Process each slide
            logger.info("Processing slides and extracting cells...")
            global_idx = 0

            for _, row in tqdm(
                processed_data.iterrows(),
                total=len(processed_data),
                desc="Processing slides",
            ):
                slide_name = self._process_slide(row, global_idx)
                if slide_name is not None:
                    self.all_slides.append(slide_name)
                    # Update global index based on number of cells added
                    if slide_name in self.cell_slide_indices:
                        global_idx += len(self.cell_slide_indices[slide_name])

            # Filter slides to those with valid features
            self.slides = [
                s
                for s in self.all_slides
                if s in self.cell_features and len(self.cell_features[s]) > 0
            ]

            logger.info(
                f"Found {len(self.slides)} valid slides out of {len(processed_data)} total slides"
            )
            logger.info(f"Total cells: {len(self._global_indices)}")

            if not self.slides:
                raise ValueError("No valid slides found")

            # Log class distribution
            self._log_class_distribution()

        except Exception as e:
            logger.error(f"Failed to process dataset: {e}")
            raise ValueError(f"Failed to process dataset: {e}")

    def _process_slide(self, row: pd.Series, start_global_idx: int) -> Optional[str]:
        """
        Process a single slide and extract cell-level features and labels.

        Args:
            row: A pandas Series representing a row from the DataFrame
            start_global_idx: Starting global index for cells in this slide

        Returns:
            slide_name if successful, None otherwise
        """
        try:
            file_path = Path(row["FULL_PATH"])
            slide_name = file_path.stem

            # Get features and cell indices
            features, cell_indices, _ = get_cell_features(
                self.folder,
                slide_name,
                self.extractor,
                self.graph_creator,
                self.segmentation_model,
            )

            if features is None or cell_indices is None or len(cell_indices) == 0:
                logger.debug(f"No features found for slide {slide_name}")
                return None

            # Get cell types
            if self.segmentation_model is not None:
                cell_types = get_cell_types(
                    self.folder, slide_name, self.segmentation_model
                )
            else:
                logger.warning(f"No segmentation model for slide {slide_name}")
                return None

            if not cell_types:
                logger.debug(f"No cell types found for slide {slide_name}")
                return None

            # Filter and extract cell-level data
            filtered_features: List[torch.Tensor] = []
            filtered_cell_types: List[int] = []
            local_indices: List[int] = []

            for cell_id, tensor_idx in cell_indices.items():
                if cell_id not in cell_types:
                    continue

                cell_type = cell_types[cell_id]  # 1-based index

                # Filter by cell types to keep if specified
                if self.cell_types_to_keep_indices is not None:
                    if cell_type not in self.cell_types_to_keep_indices:
                        continue

                # Get cell features
                cell_feature = features[tensor_idx]  # Shape: (n_features,)

                filtered_features.append(cell_feature)
                filtered_cell_types.append(cell_type)
                local_indices.append(len(self._global_indices))
                self._global_indices.append((slide_name, len(filtered_features) - 1))

            if not filtered_features:
                logger.debug(f"No cells after filtering for slide {slide_name}")
                return None

            # Store as tensors
            self.cell_features[slide_name] = torch.stack(filtered_features, dim=0)
            self.cell_types[slide_name] = torch.tensor(
                filtered_cell_types, dtype=torch.long
            )
            self.cell_slide_indices[slide_name] = local_indices

            return slide_name

        except Exception as e:
            logger.warning(f"Error processing slide row: {e}")
            return None

    def _create_label(self, cell_type: int) -> torch.Tensor:
        """
        Create a one-hot encoded label with optional label smoothing.

        Args:
            cell_type: Cell type index (1-based, 1-5)

        Returns:
            One-hot encoded tensor of shape (NUM_CELL_TYPES,) with optional smoothing
        """
        # Convert to 0-based index
        idx = cell_type - 1

        # Create base one-hot encoding
        label = torch.zeros(NUM_CELL_TYPES, dtype=torch.float32)
        label[idx] = 1.0

        # Apply label smoothing for this specific cell type
        smoothing = self.label_smoothing_per_type.get(cell_type, 0.0)
        if smoothing > 0:
            # Label smoothing: (1 - smoothing) * one_hot + smoothing / num_classes
            label = (1.0 - smoothing) * label + smoothing / NUM_CELL_TYPES

        return label

    def _save_data(self, path: Path) -> None:
        """Save processed data to disk (slides is derived, not saved)."""
        data_dict: Dict[str, Any] = {
            "all_slides": self.all_slides,
            "cell_features": self.cell_features,
            "cell_types": self.cell_types,
            "cell_slide_indices": self.cell_slide_indices,
            "_global_indices": self._global_indices,
        }

        torch.save(data_dict, path)
        logger.info(f"Saved dataset to {path}")

    def _load_data(self, path: Path) -> None:
        """Load processed data from disk and derive slides list."""
        data_dict = torch.load(path, map_location="cpu", weights_only=False)

        self.all_slides = data_dict["all_slides"]
        self.cell_features = data_dict["cell_features"]
        self.cell_types = data_dict["cell_types"]
        self.cell_slide_indices = data_dict["cell_slide_indices"]
        self._global_indices = data_dict["_global_indices"]

        # Derive slides from all_slides (those with valid features)
        self.slides = [
            s
            for s in self.all_slides
            if s in self.cell_features and len(self.cell_features[s]) > 0
        ]

        logger.info(
            f"Loaded {len(self.slides)} slides with {len(self._global_indices)} total cells"
        )

    def _log_class_distribution(self) -> None:
        """Log the distribution of cell types in the dataset."""
        type_counts: Dict[int, int] = {i: 0 for i in range(1, NUM_CELL_TYPES + 1)}

        for slide_name in self.slides:
            cell_types_tensor = self.cell_types[slide_name]
            for cell_type in cell_types_tensor:
                type_counts[int(cell_type.item())] += 1

        total = sum(type_counts.values())
        logger.info("Cell type distribution:")
        for cell_type, count in type_counts.items():
            type_name = TYPE_NUCLEI_DICT[cell_type]
            percentage = (count / total * 100) if total > 0 else 0
            logger.info(f"  {type_name}: {count} ({percentage:.2f}%)")

    def get_class_distribution(self) -> Dict[str, int]:
        """
        Get the distribution of cell types in the dataset.

        Returns:
            Dictionary with cell type names as keys and counts as values
        """
        type_counts: Dict[str, int] = {}

        for _, name in TYPE_NUCLEI_DICT.items():
            type_counts[name] = 0

        for slide_name in self.slides:
            cell_types_tensor = self.cell_types[slide_name]
            for cell_type in cell_types_tensor:
                type_name = TYPE_NUCLEI_DICT[int(cell_type.item())]
                type_counts[type_name] += 1

        type_counts["total"] = sum(type_counts.values())
        return type_counts

    def get_num_classes(self) -> int:
        """
        Get the number of cell type classes.

        Returns:
            Number of cell type classes (5 for all types, or fewer if filtered)
        """
        if self.cell_types_to_keep_indices is not None:
            return len(self.cell_types_to_keep_indices)
        return NUM_CELL_TYPES

    def get_weights_for_sampler(self) -> torch.Tensor:
        """
        Compute weights for WeightedRandomSampler to handle class imbalance.

        Returns:
            torch.Tensor: Weights for each cell in the dataset, shape (len(dataset),)
        """
        if len(self._global_indices) == 0:
            logger.warning("No cells found for weight computation")
            return torch.ones(1, dtype=torch.float32)

        # Count cells per type (using 1-based indices)
        type_counts: Dict[int, int] = {i: 0 for i in range(1, NUM_CELL_TYPES + 1)}
        cell_types_list: List[int] = []

        for slide_name, local_idx in self._global_indices:
            cell_type = int(self.cell_types[slide_name][local_idx].item())  # 1-based
            type_counts[cell_type] += 1
            cell_types_list.append(cell_type)

        # Compute weights (inverse of class frequency)
        total_cells = len(cell_types_list)
        weights = torch.zeros(total_cells, dtype=torch.float32)

        for i, cell_type in enumerate(cell_types_list):
            count = type_counts[cell_type]
            if count > 0:
                weights[i] = 1.0 / count
            else:
                weights[i] = 1.0

        # Normalize weights
        weights = weights * len(weights) / weights.sum()

        logger.info("Computed sampling weights for cell type classification:")
        for cell_type, count in type_counts.items():
            type_name = TYPE_NUCLEI_DICT.get(cell_type, f"Type {cell_type}")
            logger.info(f"  {type_name}: {count} cells")

        return weights

    def __len__(self) -> int:
        """Return the number of cells in the dataset."""
        return len(self._global_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset.

        Args:
            idx: Index of the cell to retrieve

        Returns:
            Tuple of (features, label) where:
                - features is a tensor of shape (n_features,) for a single cell
                - label is a one-hot encoded tensor of shape (n_cell_types,)
        """
        if idx >= len(self._global_indices):
            raise IndexError(
                f"Index {idx} out of range for dataset of size {len(self._global_indices)}"
            )

        slide_name, local_idx = self._global_indices[idx]

        # Get cell features
        features = self.cell_features[slide_name][local_idx].clone()

        # Get cell type and create one-hot label with optional smoothing
        cell_type = int(self.cell_types[slide_name][local_idx].item())  # 1-based
        label = self._create_label(cell_type)

        # Apply transforms if provided
        if self.transforms is not None:
            # Transforms expect (n_instances, n_features), so add batch dimension
            features = features.unsqueeze(0)
            features = self.transforms.transform(features)
            features = features.squeeze(0)

        return features, label

    def create_subset(self, indices: List[int]) -> "CellTypeDataset":
        """
        Create a subset of the dataset using the specified indices.

        Args:
            indices: List of cell indices to include in the subset

        Returns:
            New CellTypeDataset instance containing only the specified cells
        """
        if not indices:
            raise ValueError("Indices list cannot be empty")

        max_idx = len(self._global_indices) - 1
        invalid_indices = [idx for idx in indices if idx < 0 or idx > max_idx]
        if invalid_indices:
            raise ValueError(
                f"Invalid indices {invalid_indices[:5]}... Valid range is 0-{max_idx}"
            )

        # Create a new instance with the same configuration
        subset = CellTypeDataset.__new__(CellTypeDataset)

        # Copy configuration
        subset.root = self.root
        subset.folder = self.folder
        subset.raw_data = self.raw_data
        subset.extractor = self.extractor
        subset.graph_creator = self.graph_creator
        subset.segmentation_model = self.segmentation_model
        subset.split = self.split
        subset.max_workers = self.max_workers
        subset.force_reload = self.force_reload
        subset.transforms = self.transforms
        subset.label_smoothing_per_type = self.label_smoothing_per_type
        subset.cell_types_to_keep_indices = self.cell_types_to_keep_indices
        subset.wsl = self.wsl

        # Share feature/label data (read-only, avoid duplication)
        subset.cell_features = self.cell_features
        subset.cell_types = self.cell_types
        subset.cell_slide_indices = self.cell_slide_indices
        subset.all_slides = self.all_slides

        # Create subset indices
        subset._global_indices = [self._global_indices[i] for i in indices]

        # Determine which slides are in the subset
        subset_slide_set = set(slide for slide, _ in subset._global_indices)
        subset.slides = [s for s in self.slides if s in subset_slide_set]

        logger.info(
            f"Created subset with {len(subset._global_indices)} cells from {len(subset.slides)} slides"
        )

        return subset

    def create_train_val_datasets(
        self,
        train_indices: List[int],
        val_indices: List[int],
        transforms: Optional[Union[TransformPipeline, Transform]] = None,
    ) -> Tuple["CellTypeDataset", "CellTypeDataset"]:
        """
        Create train and validation datasets from specified indices.

        Args:
            train_indices: List of cell indices for training set
            val_indices: List of cell indices for validation set
            transforms: Optional pre-fitted transforms to apply to both datasets.
                       These should already be fitted on training data before calling this method.

        Returns:
            Tuple of (train_dataset, val_dataset) with the provided transforms
        """
        if not train_indices:
            raise ValueError("Train indices list cannot be empty")
        if not val_indices:
            raise ValueError("Val indices list cannot be empty")

        # Validate indices
        max_idx = len(self._global_indices) - 1
        invalid_train = [idx for idx in train_indices if idx < 0 or idx > max_idx]
        invalid_val = [idx for idx in val_indices if idx < 0 or idx > max_idx]

        if invalid_train:
            raise ValueError(
                f"Invalid train indices {invalid_train[:5]}... Valid range is 0-{max_idx}"
            )
        if invalid_val:
            raise ValueError(
                f"Invalid val indices {invalid_val[:5]}... Valid range is 0-{max_idx}"
            )

        # Validate that transforms are fitted if provided
        if transforms is not None:
            if hasattr(transforms, "is_fitted") and not transforms.is_fitted:  # type: ignore
                raise ValueError(
                    "Transforms must be fitted before passing to create_train_val_datasets. "
                    "Fit the transforms on training data first using transforms.fit()."
                )

        logger.info(
            f"Creating train/val datasets with {len(train_indices)} train and {len(val_indices)} val cells"
        )

        # Create datasets
        train_dataset = self.create_subset(train_indices)
        val_dataset = self.create_subset(val_indices)

        # Apply pre-fitted transforms if provided
        if transforms is not None:
            train_dataset.transforms = transforms
            val_dataset.transforms = transforms

        logger.info(f"Train dataset: {len(train_dataset)} cells")
        logger.info(f"Val dataset: {len(val_dataset)} cells")

        return train_dataset, val_dataset

    def create_train_val_datasets_by_slides(
        self,
        train_slides: List[str],
        val_slides: List[str],
        transforms: Optional[Union[TransformPipeline, Transform]] = None,
    ) -> Tuple["CellTypeDataset", "CellTypeDataset"]:
        """
        Create train and validation datasets based on slide names.

        This method is useful when you want to split the CellTypeDataset
        based on the same slide-level split used for another dataset (e.g., MILDataset).
        All cells from slides in train_slides go to training, and all cells from
        slides in val_slides go to validation.

        Args:
            train_slides: List of slide names for training set
            val_slides: List of slide names for validation set
            transforms: Optional pre-fitted transforms to apply to both datasets.
                       These should already be fitted on training data before calling this method.

        Returns:
            Tuple of (train_dataset, val_dataset) with the provided transforms
        """
        if not train_slides:
            raise ValueError("Train slides list cannot be empty")
        if not val_slides:
            raise ValueError("Val slides list cannot be empty")

        # Convert to sets for faster lookup
        train_slides_set = set(train_slides)
        val_slides_set = set(val_slides)

        # Validate that slides exist in dataset
        available_slides = set(self.slides)
        invalid_train = [s for s in train_slides if s not in available_slides]
        invalid_val = [s for s in val_slides if s not in available_slides]

        if invalid_train:
            logger.warning(
                f"Some train slides not found in dataset: {invalid_train[:5]}..."
            )
        if invalid_val:
            logger.warning(
                f"Some val slides not found in dataset: {invalid_val[:5]}..."
            )

        # Find cell indices for each split based on slide membership
        train_indices: List[int] = [idx for idx, (slide_name, _) in enumerate(self._global_indices) if slide_name in train_slides_set]
        val_indices: List[int] = [idx for idx, (slide_name, _) in enumerate(self._global_indices) if slide_name in val_slides_set]

        if not train_indices:
            raise ValueError(
                "No cells found for training slides. Check that slide names match."
            )
        if not val_indices:
            raise ValueError(
                "No cells found for validation slides. Check that slide names match."
            )

        logger.info(
            f"Split by slides: {len(train_indices)} train cells from {len([s for s in train_slides if s in available_slides])} slides, "
            f"{len(val_indices)} val cells from {len([s for s in val_slides if s in available_slides])} slides"
        )

        # Use the existing method to create the datasets
        return self.create_train_val_datasets(
            train_indices=train_indices,
            val_indices=val_indices,
            transforms=transforms,
        )

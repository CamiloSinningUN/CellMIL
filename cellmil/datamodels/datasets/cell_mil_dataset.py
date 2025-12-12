import pandas as pd
import torch
import json
import random
import hashlib
from tqdm import tqdm
from pathlib import Path
from typing import List, Literal, Tuple, Union, Optional, Dict, Any
from torch.utils.data import Dataset

from cellmil.interfaces.FeatureExtractorConfig import (
    ExtractorType,
    FeatureExtractionType,
)
from cellmil.interfaces.GraphCreatorConfig import GraphCreatorType
from cellmil.interfaces.CellSegmenterConfig import ModelType, TYPE_NUCLEI_DICT
from cellmil.utils import logger
from ..transforms import (
    TransformPipeline,
    Transform,
    FittableTransform,
    LabelTransform,
    LabelTransformPipeline
)
from .utils import (
    wsl_preprocess,
    column_sanity_check,
    filter_split,
    preprocess_row,
    get_cell_types,
    get_centroids,
    cell_types_to_tensor,
    get_cell_features,
    weights_for_sampler,
    cell_type_name_to_index,
    load_roi_for_slide,
    filter_cells_by_roi,
)


class CellMILDataset(
    Dataset[Tuple[torch.Tensor, int | Tuple[float, int]] | Tuple[torch.Tensor, torch.Tensor, int | Tuple[float, int]]]
):
    """
    An PyTorch Dataset for MIL (Multiple Instance Learning) tasks.

    Returns:
        For classification tasks:
            When cell_type=False or return_cell_types=False: Tuple[torch.Tensor, int] (features, label)
            When cell_type=True and return_cell_types=True: Tuple[torch.Tensor, torch.Tensor, int] (features, cell_types, label)
        
        For survival prediction tasks:
            When cell_type=False or return_cell_types=False: Tuple[torch.Tensor, Tuple[float, int]] (features, (duration, event))
            When cell_type=True and return_cell_types=True: Tuple[torch.Tensor, torch.Tensor, Tuple[float, int]] (features, cell_types, (duration, event))
    """

    def __init__(
        self,
        root: Union[str, Path],
        label: Union[str, Tuple[str, str]],
        folder: Path,
        data: pd.DataFrame,
        extractor: Union[ExtractorType, List[ExtractorType]],
        graph_creator: GraphCreatorType | None = None,
        segmentation_model: ModelType | None = None,
        split: Literal["train", "val", "test", "all"] = "all",
        transforms: Optional[TransformPipeline | Transform] = None,
        label_transforms: Optional[LabelTransform | LabelTransformPipeline] = None,
        cell_type: bool = False,
        cell_types_to_keep: Optional[List[str]] = None,
        return_cell_types: bool = True,
        roi_folder: Optional[Path] = None,
        max_workers: int = 8,
        force_reload: bool = False,
    ):
        """
        Initialize the optimized MIL dataset.

        Args:
            root: Root directory where the processed dataset will be cached
            label: Label for the dataset. Either:
                - A single string (e.g., "dcr") for classification tasks
                - A tuple of two strings (e.g., ("duration", "event")) for survival prediction tasks
            folder: Path to the dataset folder
            data: DataFrame containing metadata. If roi_folder is provided, must contain 'ID', 'I3LUNG_ID', and 'CENTER' columns.
            extractor: Feature extractor type or list of types to use for feature extraction.
            graph_creator: Optional graph creator type, needed for some extractors
            segmentation_model: Optional Segmentation model type, needed for some extractors
            split: Dataset split (train/val/test/all). Use "all" to include all data regardless of split
            cell_type: Whether to add cell types as one-hot encoded columns to the feature tensor.
                Only available for 'cellvit' and 'hovernet' segmentation models.
            cell_types_to_keep: Optional list of cell type names to keep (e.g., ["Neoplastic", "Connective"]).
                Valid names: "Neoplastic", "Inflammatory", "Connective", "Dead", "Epithelial" (case-insensitive).
                If provided, only cells of these types will be included. Requires cell_type=True.
            roi_folder: Optional path to the directory containing ROI CSV files organized by center folders.
                If provided, cells will be filtered to only include those within ROI boundaries.
                Requires 'ID', 'I3LUNG_ID', and 'CENTER' columns in the data DataFrame.
                Slides with no cells matching these types will be excluded from the dataset.
            return_cell_types: Whether to return cell types in __getitem__. If False, only returns (features, label).
                If True and cell_type=True, returns (features, cell_types, label). Default is True for backward compatibility.
            max_workers: Maximum number of threads for parallel processing
            force_reload: Whether to force reprocessing even if processed files exist
            transforms: Optional TransformPipeline to apply to features at getitem time
            label_transform: Optional transform to apply to labels (e.g., TimeDiscretizerTransform for binning survival times)
        """
        self.root = Path(root)
        self.label = label
        self.folder = folder
        self.raw_data = data
        self.extractor = extractor
        self.graph_creator = graph_creator
        self.segmentation_model = segmentation_model
        self.split = split
        self.cell_type = cell_type
        self.return_cell_types = return_cell_types
        self.label_transforms = label_transforms

        # Convert cell type names to indices
        if cell_types_to_keep is not None:
            self.cell_types_to_keep_indices = cell_type_name_to_index(
                cell_types_to_keep
            )
        else:
            self.cell_types_to_keep_indices = None

        # ROI filtering setup
        self.roi_folder = Path(roi_folder) if roi_folder is not None else None
        
        # Validate ROI parameters
        if self.roi_folder is not None:
            if not self.roi_folder.exists():
                raise ValueError(f"ROI folder does not exist: {self.roi_folder}")
            
            # Check if data DataFrame has required columns for ROI filtering
            required_roi_columns = ['ID', 'I3LUNG_ID', 'CENTER']
            missing_columns = [col for col in required_roi_columns if col not in data.columns]
            
            if missing_columns:
                raise ValueError(
                    f"ROI filtering requires the following columns in data DataFrame: {required_roi_columns}. "
                    f"Missing columns: {missing_columns}"
                )
            
            logger.info(f"ROI filtering enabled. ROI folder: {self.roi_folder}")

        self.max_workers = max_workers
        self.force_reload = force_reload
        self.transforms = transforms

        if (
            isinstance(self.extractor, ExtractorType)
            and self.extractor in FeatureExtractionType.Embedding
        ) or (
            isinstance(self.extractor, list)
            and any(ext in FeatureExtractionType.Embedding for ext in self.extractor)
        ):
            raise ValueError("Embedding extractor is not supported for this dataset.")

        # Validate that cell types are only used with compatible segmentation models
        if self.cell_type and self.segmentation_model not in ["cellvit", "hovernet"]:
            raise ValueError(
                f"Cell types can only be used with 'cellvit' or 'hovernet' segmentation models. "
                f"Got '{self.segmentation_model}'"
            )

        # Validate that cell_types_to_keep requires cell_type=True
        if self.cell_types_to_keep_indices is not None and not self.cell_type:
            raise ValueError(
                "cell_types_to_keep requires cell_type=True. "
                "Please set cell_type=True to enable cell type filtering."
            )

        # TODO: Make this configurable
        self.wsl = False
        # TODO ---

        # Data structures
        self.all_slides: List[str] = []  # All slides with features
        self.slides: List[str] = []  # Only slides with labels for this task
        self.labels: Dict[str, Union[int, Tuple[float, int]]] = {}
        self.cell_types: Dict[str, Dict[int, int]] = {}

        # Raw data storage (transforms applied at getitem time)
        self.features: Dict[str, torch.Tensor] = {}
        self.cell_types_tensors: Dict[
            str, torch.Tensor
        ] = {}  # Store cell types separately
        self.cell_indices: Dict[str, Dict[int, int]] = {}

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
        
        # Apply ROI filtering if enabled (after loading/processing complete dataset)
        if self.roi_folder is not None:
            self._apply_roi_filtering()

    def _get_data_path(self) -> Path:
        """Get the path for the processed dataset file."""
        # Create a hash of the configuration to ensure cache invalidation
        # when parameters change
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
            "cell_type": self.cell_type,
        }

        config_str = json.dumps(config_dict, sort_keys=True)
        config_hash = hashlib.md5(config_str.encode("utf-8")).hexdigest()[:8]

        logger.info(f"Dataset configuration:{config_dict}")
        logger.info(f"Dataset configuration hash: {config_hash}")

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
                processed_data = filter_split(processed_data, self.split)
            else:
                logger.info(f"Using all data: {len(processed_data)} slides")

            # Extract valid slides and labels
            logger.info("Validating slides...")
            for _, row in tqdm(
                processed_data.iterrows(),
                total=len(processed_data),
                desc="Validating slides",
            ):
                try:
                    slide_name, _ = preprocess_row(
                        row,
                        self.label,
                        self.folder,
                        self.extractor,
                        self.graph_creator,
                        self.segmentation_model,
                    )

                    if slide_name is not None:
                        self.all_slides.append(slide_name)
                except Exception as e:
                    raise ValueError(f"Error processing row: {e}")

            logger.info(
                f"Found {len(self.all_slides)} valid slides out of {len(processed_data)} total slides"
            )

            if not self.all_slides:
                raise ValueError("No valid slides found")

            # Precompute cell types if enabled
            if self.cell_type and self.segmentation_model:
                logger.info("Precomputing cell types for all slides...")
                for slide_name in tqdm(self.all_slides, desc="Loading cell types"):
                    _cell_types = get_cell_types(
                        self.folder, slide_name, self.segmentation_model
                    )

                    if _cell_types is None:
                        logger.warning(f"No cell types found for slide {slide_name}")
                        self.cell_types[slide_name] = {}
                    else:
                        self.cell_types[slide_name] = _cell_types

            # Load all features (without applying transforms)
            logger.info("Loading raw features for all slides...")
            valid_slides: List[str] = []
            for slide_name in tqdm(self.all_slides, desc="Loading features"):
                try:
                    features = self._load_slide_features(slide_name)
                    if features is not None:
                        # Store raw features
                        self.features[slide_name] = features
                        valid_slides.append(slide_name)
                except Exception as e:
                    logger.error(f"Failed to load slide {slide_name}: {e}")

            # Update all_slides to only include those with successfully loaded features
            self.all_slides = valid_slides

            # Now get labels and filter slides to only those with labels
            self.labels = self._get_labels()
            self.slides = [slide for slide in self.all_slides if slide in self.labels]

            # Filter out slides with no matching cell types if cell_types_to_keep is specified
            if self.cell_types_to_keep_indices is not None:
                logger.info("Filtering slides with no matching cell types...")
                slides_with_matching_cells: list[str] = []

                for slide_name in tqdm(self.slides, desc="Checking cell types"):
                    cell_types_tensor = self.cell_types_tensors.get(slide_name)

                    if cell_types_tensor is None or cell_types_tensor.shape[0] == 0:
                        raise ValueError(
                            f"Missing or empty cell types tensor for slide {slide_name}"
                        )

                    # Get the cell type for each cell (argmax of one-hot encoding)
                    cell_type_tensor_indices = torch.argmax(cell_types_tensor, dim=1)

                    # Convert 1-based TYPE_NUCLEI_DICT indices to 0-based tensor indices
                    tensor_indices_to_keep = [
                        idx - 1 for idx in self.cell_types_to_keep_indices
                    ]

                    # Check if any cells match the requested types
                    has_matching_cells = any(
                        (cell_type_tensor_indices == tensor_idx).any().item()
                        for tensor_idx in tensor_indices_to_keep
                    )

                    if has_matching_cells:
                        slides_with_matching_cells.append(slide_name)
                    else:
                        logger.debug(
                            f"Skipping slide {slide_name}: no cells of requested types"
                        )

                self.slides = slides_with_matching_cells
                logger.info(f"Kept {len(self.slides)} slides with matching cell types")

            logger.info(f"Successfully loaded {len(self.features)} slides")
            logger.info(f"Found {len(self.slides)} slides with labels for current task")

        except Exception as e:
            logger.error(f"Failed to process dataset: {e}")
            raise ValueError(f"Failed to process dataset: {e}")

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
            for _, row in processed_data.iterrows():
                try:
                    slide_name, label_value = preprocess_row(
                        row,
                        self.label,
                        self.folder,
                        self.extractor,
                        self.graph_creator,
                        self.segmentation_model,
                        do_validate_features=False,
                    )

                    if slide_name in self.all_slides and label_value is not None:
                        labels[slide_name] = label_value

                except Exception as e:
                    logger.warning(f"Error extracting label: {e}")

            return labels

        except Exception as e:
            logger.error(f"Failed to extract labels: {e}")
            raise

    def _load_slide_features(self, slide_name: str) -> Optional[torch.Tensor]:
        """
        Load raw features for a single slide without applying transforms.

        Args:
            slide_name: Name of the slide to process

        Returns:
            Raw features tensor or None if loading fails
        """
        try:
            # Load features from one or many extractors and concatenate column-wise
            features, cell_indices, _ = get_cell_features(
                self.folder,
                slide_name,
                self.extractor,
                self.graph_creator,
                self.segmentation_model,
            )

            if cell_indices is None or features is None:
                raise ValueError("Missing features or cell_indices")

            # Store cell indices for potential future use
            self.cell_indices[slide_name] = cell_indices

            # Store cell types as a separate tensor if requested
            if self.cell_type:
                if len(cell_indices) == 0:
                    logger.warning(
                        f"Missing cell_indices for slide {slide_name}; adding zero cell types."
                    )
                    n_cell_types = len(TYPE_NUCLEI_DICT)
                    cell_types_onehot = torch.zeros(
                        features.size(0), n_cell_types, dtype=torch.float32
                    )
                else:
                    cell_types_onehot = self._get_cell_types_tensor(
                        slide_name, cell_indices
                    )

                # Store cell types separately instead of concatenating
                self.cell_types_tensors[slide_name] = cell_types_onehot

            return features

        except Exception as e:
            logger.error(f"Error loading slide {slide_name}: {e}")
            return None

    def _get_cell_types_tensor(
        self, slide_name: str, cell_indices: Dict[int, int]
    ) -> torch.Tensor:
        """
        Get cell types for a given slide and convert to one-hot encoding.

        Args:
            slide_name: Name of the slide
            cell_indices: Dictionary mapping cell_id to tensor index

        Returns:
            Tensor of shape (n_cells, n_cell_types) containing one-hot encoded cell types
        """
        n_cells = len(cell_indices)
        n_cell_types = len(TYPE_NUCLEI_DICT)

        # Get cached data from the dictionary cache
        cell_types = self.cell_types.get(slide_name)

        if cell_types is None:
            logger.warning(f"No cached cell types found for slide {slide_name}")
            cell_types_onehot = torch.zeros(n_cells, n_cell_types, dtype=torch.float32)
        elif len(cell_types) == 0:
            # Empty dict means no cell types available
            cell_types_onehot = torch.zeros(n_cells, n_cell_types, dtype=torch.float32)
        else:
            # Convert cached dict to one-hot tensor
            cell_types_onehot = cell_types_to_tensor(cell_types, cell_indices)

        return cell_types_onehot

    def _save_data(self, path: Path) -> None:
        """Save data to disk (all slides with features, label-independent)."""
        data_dict: dict[str, Any] = {
            "all_slides": self.all_slides,
            "cell_types": self.cell_types,
            "features": self.features,
            "cell_types_tensors": self.cell_types_tensors,
            "cell_indices": self.cell_indices,
        }

        torch.save(data_dict, path)
        logger.info(f"Saved dataset to {path}")

    def _load_data(self, path: Path) -> None:
        """Load data from disk and extract labels for current task."""
        data_dict = torch.load(path, map_location="cpu", weights_only=False)

        self.all_slides = data_dict["all_slides"]
        self.cell_types = data_dict["cell_types"]

        self.features = data_dict.get("features", {})
        self.cell_types_tensors = data_dict.get("cell_types_tensors", {})
        self.cell_indices = data_dict.get("cell_indices", {})

        # Extract labels for current task and filter slides
        self.labels = self._get_labels()
        
        # Apply label transform if provided (e.g., discretize survival times)
        if self.label_transforms is not None and isinstance(self.label, tuple):
            # For survival tasks, apply discretization
            self.labels = self.label_transforms.transform_labels(self.labels)
        
        self.slides = [slide for slide in self.all_slides if slide in self.labels]

        logger.info(
            f"Loaded {len(self.all_slides)} total slides, {len(self.slides)} with labels for current task"
        )

    def _apply_roi_filtering(self) -> None:
        """
        Apply ROI filtering to the loaded dataset.
        This filters cells in each slide to keep only those within ROI boundaries.
        Creates roi_filtered versions of features, cell_types_tensors, and cell_indices.
        """
        logger.info("Applying ROI filtering to loaded dataset...")
        
        if self.segmentation_model is None:
            raise ValueError("segmentation_model must be specified for ROI filtering")
        
        if self.roi_folder is None:
            raise ValueError("roi_folder must be specified for ROI filtering")
        
        slides_to_remove: list[str] = []
        total_cells_before = 0
        total_cells_after = 0
        
        for slide_name in tqdm(self.slides, desc="Applying ROI filtering"):
            # Load ROI for this slide
            roi_df = load_roi_for_slide(slide_name, self.roi_folder, self.raw_data)
            
            if roi_df is None:
                logger.warning(
                    f"No ROI found for slide {slide_name}, removing slide from dataset"
                )
                slides_to_remove.append(slide_name)
                continue
            
            # Get cell centroids
            centroids = get_centroids(self.folder, slide_name, self.segmentation_model)
            
            if centroids is None:
                logger.warning(f"Could not load centroids for slide {slide_name}, keeping all cells")
                continue
            
            # Filter cells by ROI
            cells_to_keep = filter_cells_by_roi(centroids, roi_df)
            
            total_cells_before += len(centroids)
            total_cells_after += len(cells_to_keep)
            
            if len(cells_to_keep) == 0:
                logger.warning(f"No cells within ROI for slide {slide_name}, removing slide from dataset")
                slides_to_remove.append(slide_name)
                continue
            
            # Get original cell_indices for this slide
            cell_indices = self.cell_indices[slide_name]
            
            # Filter cell_indices to keep only cells within ROI
            filtered_cell_indices = {
                cell_id: idx 
                for cell_id, idx in cell_indices.items() 
                if cell_id in cells_to_keep
            }
            
            # Create a mapping from old indices to new indices
            old_to_new_idx = {
                old_idx: new_idx 
                for new_idx, old_idx in enumerate(sorted(filtered_cell_indices.values()))
            }
            
            # Filter features tensor
            indices_to_keep = sorted(filtered_cell_indices.values())
            self.features[slide_name] = self.features[slide_name][indices_to_keep]
            
            # Update cell_indices with new sequential indices
            self.cell_indices[slide_name] = {
                cell_id: old_to_new_idx[old_idx] 
                for cell_id, old_idx in filtered_cell_indices.items()
            }
            
            # Filter cell types if they exist
            if slide_name in self.cell_types_tensors:
                self.cell_types_tensors[slide_name] = self.cell_types_tensors[slide_name][indices_to_keep]
            
            logger.debug(
                f"ROI filtering for {slide_name}: kept {len(cells_to_keep)}/{len(centroids)} cells "
                f"({len(cells_to_keep)/len(centroids)*100:.1f}%)"
            )
        
        # Remove slides with no cells in ROI
        for slide_name in slides_to_remove:
            self.slides.remove(slide_name)
            if slide_name in self.labels:
                del self.labels[slide_name]
        
        logger.info(
            f"ROI filtering complete: kept {total_cells_after}/{total_cells_before} cells "
            f"({total_cells_after/total_cells_before*100:.1f}%) across {len(self.slides)} slides"
        )
        if slides_to_remove:
            logger.info(f"Removed {len(slides_to_remove)} slides with no cells in ROI")

    def get_num_labels(self) -> int:
        """
        Get the number of unique labels in the dataset.
        
        Note: For survival prediction tasks, this returns 0 as there are no discrete classes.
        """
        # Check if we have survival data by looking at the first label
        if self.labels:
            first_label = next(iter(self.labels.values()))
            if isinstance(first_label, tuple):
                # Survival data - no discrete classes
                return 0
        
        # Classification data
        return len(set(self.labels.values()))

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

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.
        """
        return len(self.slides)

    def __getitem__(
        self, idx: int
    ) -> Union[
        Tuple[torch.Tensor, int | Tuple[float, int]], 
        Tuple[torch.Tensor, torch.Tensor, int | Tuple[float, int]]
    ]:
        """
        Get a sample from the dataset.

        Args:
            idx: Index of the sample to retrieve

        Returns:
            For classification tasks:
                If cell_type=False or return_cell_types=False:
                    Tuple of (features, label) where features is a tensor of shape (n_instances, n_features)
                If cell_type=True and return_cell_types=True:
                    Tuple of (features, cell_types, label) where:
                    - features is a tensor of shape (n_instances, n_features)
                    - cell_types is a tensor of shape (n_instances, n_cell_types) with one-hot encoded cell types
                    - label is the sample label (int)
            
            For survival prediction tasks:
                If cell_type=False or return_cell_types=False:
                    Tuple of (features, (duration, event)) where features is a tensor and (duration, event) is survival data
                If cell_type=True and return_cell_types=True:
                    Tuple of (features, cell_types, (duration, event)) where:
                    - features is a tensor of shape (n_instances, n_features)
                    - cell_types is a tensor of shape (n_instances, n_cell_types) with one-hot encoded cell types
                    - (duration, event) is the survival data tuple
        """
        slide_name = self.slides[idx]

        # Get raw features from cache
        features = self.features[
            slide_name
        ].clone()  # Clone to avoid modifying cached data

        label = self.labels[slide_name]

        # Handle cell type filtering if enabled
        if self.cell_type:
            # Get cell types tensor
            if slide_name in self.cell_types_tensors:
                cell_types = self.cell_types_tensors[slide_name].clone()
            else:
                raise ValueError(f"Missing cell types tensor for slide {slide_name}")

            # Filter by cell types if requested
            if self.cell_types_to_keep_indices is not None:
                # Get the cell type for each cell (argmax of one-hot encoding)
                # Note: cell_types tensor uses 0-based indexing (TYPE_NUCLEI_DICT keys are 1-based)
                cell_type_tensor_indices = torch.argmax(
                    cell_types, dim=1
                )  # (n_cells,) values 0-4

                # Convert 1-based TYPE_NUCLEI_DICT indices to 0-based tensor indices
                tensor_indices_to_keep = [
                    idx - 1 for idx in self.cell_types_to_keep_indices
                ]

                # Create mask for cells to keep
                mask = torch.zeros(cell_type_tensor_indices.shape[0], dtype=torch.bool)
                for tensor_idx in tensor_indices_to_keep:
                    mask |= cell_type_tensor_indices == tensor_idx

                # Apply mask to filter both features and cell_types
                features = features[mask]
                cell_types = cell_types[mask]

            # Apply transforms after filtering
            if self.transforms is not None:
                features = self.transforms.transform(features)

            # Return based on return_cell_types parameter
            if self.return_cell_types:
                return features, cell_types, label
            else:
                return features, label
        else:
            # Apply transforms if provided
            if self.transforms is not None:
                features = self.transforms.transform(features)

            return features, label

    def create_subset(self, indices: List[int]) -> "CellMILDataset":
        """
        Create a subset of the dataset using the specified indices.

        This is useful for creating train/val/test splits when using split="all".
        The subset will share the same cached features but only include the specified samples.

        Args:
            indices: List of indices to include in the subset

        Returns:
            New CellMILDataset instance containing only the specified samples

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
        subset = CellMILDataset.__new__(CellMILDataset)

        # Copy configuration
        subset.root = self.root
        subset.label = self.label
        subset.folder = self.folder
        subset.raw_data = self.raw_data
        subset.extractor = self.extractor
        subset.graph_creator = self.graph_creator
        subset.segmentation_model = self.segmentation_model
        subset.split = self.split
        subset.cell_type = self.cell_type
        subset.return_cell_types = self.return_cell_types
        subset.cell_types_to_keep_indices = self.cell_types_to_keep_indices
        subset.max_workers = self.max_workers
        subset.force_reload = self.force_reload
        subset.transforms = self.transforms
        subset.wsl = self.wsl

        # Create subset data
        subset.slides = [self.slides[i] for i in indices]
        subset.all_slides = self.all_slides  # Share the same all_slides
        subset.cell_types = self.cell_types  # Share cell types

        # Share the same features cache (no need to copy)
        subset.features = self.features
        subset.cell_types_tensors = self.cell_types_tensors
        subset.cell_indices = self.cell_indices

        # Extract labels fresh for the subset
        subset.labels = subset._get_labels()

        logger.info(
            f"Created subset with {len(subset.slides)} samples from {len(self.slides)} total samples"
        )

        return subset

    def create_train_val_datasets(
        self,
        train_indices: List[int],
        val_indices: List[int],
        transforms: Optional[Union[TransformPipeline, Transform]] = None,
        label_transforms: Optional[LabelTransform | LabelTransformPipeline] = None,
    ) -> Tuple["CellMILDataset", "CellMILDataset"]:
        """
        Create train and validation datasets with transforms fitted only on training data.

        This method prevents data leakage by ensuring that any fittable transforms
        (like normalization or feature selection) are fitted only on the training set
        and then applied to both train and validation sets.

        Args:
            train_indices: List of indices for training set
            val_indices: List of indices for validation set
            transforms: Optional transforms to apply. If provided, any FittableTransform
                       will be fitted on training data only
            label_transforms: Optional label transform (e.g., TimeDiscretizerTransform) to apply.
                           Will be fitted on training labels only

        Returns:
            Tuple of (train_dataset, val_dataset) with properly fitted transforms

        Raises:
            ValueError: If indices lists are empty or contain invalid indices
        """
        if not train_indices:
            raise ValueError("train_indices cannot be empty")
        if not val_indices:
            raise ValueError("val_indices cannot be empty")

        # Validate indices
        max_idx = len(self.slides) - 1
        invalid_train = [idx for idx in train_indices if idx < 0 or idx > max_idx]
        invalid_val = [idx for idx in val_indices if idx < 0 or idx > max_idx]

        if invalid_train:
            raise ValueError(
                f"Invalid train indices {invalid_train}. Valid range is 0-{max_idx}"
            )
        if invalid_val:
            raise ValueError(
                f"Invalid val indices {invalid_val}. Valid range is 0-{max_idx}"
            )

        logger.info(
            f"Creating train/val datasets with {len(train_indices)} train and {len(val_indices)} val samples"
        )

        # Fit label transform on training labels if provided (before feature transforms)
        if label_transforms is not None and isinstance(self.label, tuple):
            logger.info("Fitting label transform on training survival data...")
            
            # Get training labels
            train_labels = {
                self.slides[idx]: self.labels[self.slides[idx]] 
                for idx in train_indices
            }
            
            # Fit the label transform on training labels
            label_transforms.fit(train_labels) # type: ignore
            logger.info(f"Label transform fitted with {label_transforms.n_bins} bins") # type: ignore

        # If no transforms provided, create simple subsets
        if transforms is None and label_transforms is None:
            train_dataset = self.create_subset(train_indices)
            val_dataset = self.create_subset(val_indices)
            return train_dataset, val_dataset

        # Always fit fittable transforms on training data, regardless of is_fitted
        logger.info("Fitting transforms on training data to prevent data leakage...")

        # Collect training features for fitting
        all_train_features: List[torch.Tensor] = []
        max_samples_per_slide = 100_000
        sample_size = min(len(train_indices), 20)
        indices = random.sample(train_indices, sample_size)

        for idx in indices:
            try:
                result = self[idx]
                features = result[0]

                # Subsample if needed to prevent memory issues
                if features.shape[0] > max_samples_per_slide:
                    perm_indices = torch.randperm(features.shape[0])[
                        :max_samples_per_slide
                    ]
                    features = features[perm_indices]

                all_train_features.append(features)

            except Exception as e:
                logger.warning(
                    f"Failed to load sample {idx} for transform fitting: {e}"
                )
                continue

        if not all_train_features:
            raise ValueError("No valid training samples found for fitting transforms")

        # Concatenate all training features
        combined_train_features = torch.cat(all_train_features, dim=0)
        logger.info(
            f"Fitting transforms on {combined_train_features.shape[0]} training instances with {combined_train_features.shape[1]} features"
        )

        # Fit the transforms if they are fittable
        if isinstance(transforms, (TransformPipeline, FittableTransform)):
            transforms.fit(combined_train_features)
        else:
            # If not fittable, just assign
            pass

        # Create datasets with fitted transforms
        train_dataset = self.create_subset(train_indices)
        val_dataset = self.create_subset(val_indices)

        train_dataset.transforms = transforms
        val_dataset.transforms = transforms
        
        # Apply label transform to both datasets if provided
        if label_transforms is not None:
            train_dataset.label_transforms = label_transforms
            val_dataset.label_transforms = label_transforms
            
            # Apply label transformation to the labels
            if isinstance(self.label, tuple):
                train_dataset.labels = label_transforms.transform_labels(train_dataset.labels)
                val_dataset.labels = label_transforms.transform_labels(val_dataset.labels)

        logger.info("Successfully created train/val datasets with fitted transforms")
        logger.info(f"Train dataset: {len(train_dataset)} samples")
        logger.info(f"Val dataset: {len(val_dataset)} samples")

        return train_dataset, val_dataset

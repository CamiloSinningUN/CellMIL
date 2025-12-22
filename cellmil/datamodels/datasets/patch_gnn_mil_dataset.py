import torch
import pandas as pd
import json
import hashlib
from tqdm import tqdm
from pathlib import Path
from typing import List, Literal, Tuple, Union, Optional, Dict, Callable, Any
from torch_geometric.data import InMemoryDataset, Data  # type: ignore

from cellmil.interfaces.FeatureExtractorConfig import ExtractorType, FeatureExtractionType
from cellmil.utils import logger
from ..transforms import Transform, TransformPipeline, LabelTransform, LabelTransformPipeline, FittableLabelTransform
from .utils import (
    wsl_preprocess,
    column_sanity_check,
    filter_split,
    preprocess_row,
    get_feature_path
)


class PatchGNNMILDataset(InMemoryDataset):
    def __init__(
        self, 
        root: Union[str, Path], 
        folder: Union[str, Path], 
        label: Union[str, Tuple[str, str]], 
        data: pd.DataFrame, 
        extractor: ExtractorType, 
        split: Literal["train", "val", "test", "all"] = "all", 
        transform: Optional[Callable[[Data], Data]] = None, 
        pre_transform: Optional[Callable[[Data], Data]] = None, 
        pre_filter: Optional[Callable[[Data], bool]] = None, 
        force_reload: bool = False,
        transforms: Optional[Union[Transform, TransformPipeline]] = None,
        label_transforms: Optional[Union[LabelTransform, LabelTransformPipeline]] = None,
    ):

        self.folder = Path(folder)
        self.label = label
        self.raw_data = data
        self.extractor = extractor
        self.split = split
        self.transform = transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter
        self.force_reload = force_reload
        self.transforms = transforms
        self.label_transforms = label_transforms
        
        # TODO: Make this configurable
        self.wsl = False
        # TODO: -------
        
        if self.extractor not in FeatureExtractionType.Embedding:
            raise ValueError(f"Extractor {self.extractor} not supported for PatchMILDataset. Use embedding extractors.")
        
        super().__init__( #type: ignore
            root=str(root),
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
            force_reload=force_reload
        )
        
        # Load the processed data
        self.load(self.processed_paths[0])
        
        # Extract slide information
        if not hasattr(self, 'all_slides') or not self.all_slides:
            # Extract all slides from cached data
            self.all_slides = self._get_slides()
        
        # Extract labels fresh from DataFrame (not cached with features)
        self.labels = self._get_labels()
        
        # Filter slides to only those with labels for this task
        self.slides = [slide for slide in self.all_slides if slide in self.labels]
        
        logger.info(f"Loaded optimized GNN+MIL Dataset with {len(self)} total samples, {len(self.slides)} with labels for current task")
    
    @property
    def raw_file_names(self) -> List[str]:
        """
        Required by InMemoryDataset.
        Since we're working with pre-existing data, we return an empty list.
        """
        return []
    
    @property 
    def processed_file_names(self) -> List[str]:
        """
        Return the name of the processed file that will contain our data.
        Create a stable hash based on the configuration.
        """
        
        config_dict: dict[str, Any] = {
            'extractor': str(self.extractor),
            'split': self.split
        }

        # Convert config_dict to a JSON string, which ensures deterministic representation
        config_str = json.dumps(config_dict, sort_keys=True)
        
        # Use hashlib to generate a stable hash from the string
        config_hash = hashlib.md5(config_str.encode('utf-8')).hexdigest()[:8]  # Shorten hash to 8 characters
        
        logger.info(f"Dataset configuration:{config_dict}")
        logger.info(f"Dataset configuration hash: {config_hash}")
        
        return [f'data_{self.split}_{config_hash}.pt']
    
    def download(self) -> None:
        """
        Required by InMemoryDataset.
        Since we're working with pre-existing data, we don't need to download anything.
        """
        pass
    
    def get_config(self) -> dict[str, Any]:
        """Get dataset configuration as a dictionary."""
        return {
            "dataset_type": self.__class__.__name__,
            "label": str(self.label),
            "extractor": str(self.extractor),
            "split": self.split,
        }

    def get_num_labels(self) -> int:
        """
        Get the number of unique labels in the dataset.
        
        Note: For survival prediction tasks, this returns 0 as there are no discrete classes.
        """
        if hasattr(self, 'labels') and self.labels:
            # Check if we have survival data by looking at the first label
            first_label = next(iter(self.labels.values()))
            if isinstance(first_label, tuple):
                # Survival data - no discrete classes
                return 0
            # Classification data
            return len(set(self.labels.values()))
        return self.num_classes
    
    def process(self) -> None:
        """
        Required by InMemoryDataset.
        Process all the raw data and save it as a list of Data objects.
        This is where the real work happens and it's only done once.
        """
        logger.info("Processing GNN+MIL dataset...")
        
        # Process the DataFrame
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

            # Extract valid slides
            all_slides: list[str] = []

            for _, row in tqdm(processed_data.iterrows(), total=len(processed_data), desc="Validating slides"):
                try:
                    slide_name, _ = preprocess_row(
                        row,
                        self.label,
                        self.folder,
                        self.extractor
                    )

                    if slide_name is not None:
                        all_slides.append(slide_name)
                    else:
                        logger.warning(f"Invalid slide: {slide_name}")

                except Exception as e:
                    logger.warning(f"Error validating slide: {e}")

            self.all_slides = all_slides
            logger.info(f"Found {len(self.all_slides)} valid slides")

            if not self.all_slides:
                raise ValueError("No valid slides found")

            # Process all graphs
            data_list: list[Data] = []
            for _, slide_name in enumerate(tqdm(
                self.all_slides, 
                total=len(self.all_slides), 
                desc="Processing graphs"
            )):
                try:
                    graph_data = self._process_single_graph(slide_name)
                    if graph_data is not None:
                        data_list.append(graph_data)
                except Exception as e:
                    logger.error(f"Failed to process slide {slide_name}: {e}")
                    continue

            logger.info(f"Successfully processed {len(data_list)} graphs")

            if not data_list:
                raise ValueError("No graphs were successfully processed")


            # Apply pre_filter if provided
            if self.pre_filter is not None: # type: ignore
                data_list = [data for data in data_list if self.pre_filter(data)] # type: ignore
                logger.info(f"After filtering: {len(data_list)} graphs remain")

            # Apply pre_transform if provided
            if self.pre_transform is not None: # type: ignore
                data_list = [self.pre_transform(data) for data in data_list] # type: ignore

            # Save the processed data
            self.save(data_list, self.processed_paths[0])
            logger.info(f"Saved processed dataset to {self.processed_paths[0]}")

        except Exception as e:
            logger.error(f"Failed to process dataset: {e}")
            raise

    def _get_slides(self) -> List[str]:
        """
        Extract slide names from the cached graph data.
        
        Returns:
            List of slide names corresponding to all processed graphs
        """
        try:
            all_slides: List[str] = []
            for i in range(len(self)):
                # Get the data object directly from cache
                data = super().get(i)
                if hasattr(data, 'slide_name'):
                    all_slides.append(data.slide_name)
                else:
                    raise ValueError(f"Graph at index {i} missing slide_name attribute")
            
            logger.info(f"Extracted {len(all_slides)} slide names from cached data")
            return all_slides
            
        except Exception as e:
            logger.error(f"Failed to extract slide names: {e}")
            raise

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
                        do_validate_features=False
                    )

                    if slide_name in self.all_slides and label_value is not None:
                        labels[slide_name] = label_value

                except Exception as e:
                    logger.warning(f"Error extracting label: {e}")

            return labels

        except Exception as e:
            logger.error(f"Failed to extract labels: {e}")
            raise

    def _process_single_graph(
        self, 
        slide_name: str
    ) -> Optional[Data]:
        """
        Process a single slide into a graph Data object without labels.
        Labels are handled separately to enable caching across different classification tasks.
        
        Args:
            slide_name: Name of the slide
            
        Returns:
            Processed Data object without labels or None if processing fails
        """
        try:
            # Load features
            data = torch.load(
                get_feature_path(self.folder, slide_name, self.extractor),
                map_location="cpu",
                weights_only=False,
            )
            features = data["features"]
            coordinates = data["patch_indices"]
            
            positions = self._get_positions_tensor(coordinates)

            # Merge graph with features
            graph_data = self._merge_graph_with_features(
                features, positions
            )

            # Store slide name for reference
            graph_data.slide_name = slide_name

            return graph_data

        except Exception as e:
            logger.error(f"Error processing slide {slide_name}: {e}")
            return None

    def _get_positions_tensor(self, coordinates: list[str]) -> torch.Tensor:
        """Get positions tensor for a slide."""
        return torch.tensor([
            [int(coord.split("_")[0]), int(coord.split("_")[1])]
            for coord in coordinates
        ])
        
    def _compute_graph(self):
        pass
    
    def _merge_graph_with_features(
        self, 
        features: torch.Tensor, 
        positions: torch.Tensor
    ) -> Data:
        """Create graph with spatial connectivity based on patch positions."""
        try:
            num_nodes = features.shape[0]
            
            if num_nodes != positions.shape[0]:
                raise ValueError(f"Mismatch between features ({num_nodes}) and positions ({positions.shape[0]})")
            
            # Create edges between spatially adjacent patches
            edge_list: list[tuple[int, int]] = []
            
            # Convert positions to a dictionary for fast lookup
            pos_to_idx: Dict[Tuple[int, int], int] = {}
            for idx, pos in enumerate(positions):
                pos_key = (int(pos[0].item()), int(pos[1].item()))
                pos_to_idx[pos_key] = idx
            
            # For each patch, check for neighbors (±1 in x or y direction)
            for idx, pos in enumerate(positions):
                x, y = int(pos[0].item()), int(pos[1].item())
                
                # Check 4-connected neighbors (up, down, left, right)
                neighbors: list[tuple[int, int]] = [
                    (x + 1, y),     # right
                    (x - 1, y),     # left
                    (x, y + 1),     # up
                    (x, y - 1),     # down
                    (x + 1, y + 1), # up-right
                    (x - 1, y + 1), # up-left
                    (x + 1, y - 1), # down-right
                    (x - 1, y - 1), # down-left
                ]
                
                for neighbor_pos in neighbors:
                    if neighbor_pos in pos_to_idx:
                        neighbor_idx = pos_to_idx[neighbor_pos]
                        # Add edge (undirected, so add both directions)
                        edge_list.append((idx, neighbor_idx))
                        edge_list.append((neighbor_idx, idx))
            
            # Remove duplicate edges and convert to tensor
            edge_set = set(edge_list)
            if edge_set:
                edge_index = torch.tensor(list(edge_set), dtype=torch.long).t().contiguous()
            else:
                # If no edges, create empty edge_index with correct shape
                edge_index = torch.empty((2, 0), dtype=torch.long)
            
            # Create the graph data object
            graph_data = Data(
                x=features,
                pos=positions.float(),
                edge_index=edge_index,
                num_nodes=num_nodes,
            )
            
            return graph_data

        except Exception as e:
            raise ValueError(f"Failed to create graph with features: {e}")
    
    def __len__(self) -> int:
        """Return the number of slides with labels for this task."""
        if hasattr(self, 'slides') and hasattr(self, 'labels') and self.labels:
            return len(self.slides)
        else:
            return super().__len__()
    
    def get(self, idx: int) -> Data:
        """
        Override get method to apply feature transforms and attach labels dynamically.
        
        Args:
            idx: Index of the sample to retrieve (based on slides with labels)
            
        Returns:
            Data object with transforms and labels applied
        """
        # Map from logical index (slides with labels) to actual cache index (all slides)
        if hasattr(self, 'slides') and idx < len(self.slides):
            slide_name = self.slides[idx]
            # Find the actual index in the cached data
            actual_idx = self.all_slides.index(slide_name)
        else:
            actual_idx = idx
        
        # Get the original data from the parent class
        data = super().get(actual_idx)
        
        # Clone to avoid modifying the cached data
        data = data.clone()
        
        # Attach label dynamically (not cached with features)
        if hasattr(self, 'labels') and hasattr(data, 'slide_name') and data.slide_name in self.labels:
            label = self.labels[data.slide_name]
            
            # Apply label transforms if provided
            if self.label_transforms is not None:
                labels_dict = {data.slide_name: label}
                transformed = self.label_transforms.transform_labels(labels_dict)
                label = transformed[data.slide_name]
            
            data.y = torch.tensor([label], dtype=torch.long) if isinstance(label, int) else label
        
        # Apply feature transforms if provided
        if self.transforms is not None and hasattr(data, 'x') and data.x is not None:
            # Clone to avoid modifying the cached data
            data = data.clone()
            data.x = self.transforms.transform(data.x)
        
        return data  # type: ignore
    
    def create_subset(self, indices: List[int]) -> "SubsetPatchGNNMILDataset":
        """
        Create a subset of the dataset using the specified indices.
        
        This is useful for creating train/val/test splits when using split="all".
        Note: This creates a lightweight wrapper that references the original data.
        
        Args:
            indices: List of indices to include in the subset
            
        Returns:
            New PatchGNNMILDataset instance containing only the specified samples
            
        Raises:
            ValueError: If any index is out of range
        """
        # Allow empty indices for validation-less training (e.g., final model on all data)
        if not indices:
            subset = SubsetPatchGNNMILDataset(self, [])
            logger.info("Created empty GNN subset (0 samples)")
            return subset
            
        max_idx = len(self) - 1
        invalid_indices = [idx for idx in indices if idx < 0 or idx > max_idx]
        if invalid_indices:
            raise ValueError(f"Invalid indices {invalid_indices}. Valid range is 0-{max_idx}")
        
        # Create a simple subset wrapper
        subset = SubsetPatchGNNMILDataset(self, indices)
        
        logger.info(f"Created GNN subset with {len(indices)} samples from {len(self)} total samples")
        
        return subset
    
    def create_train_val_datasets(
        self, 
        train_indices: List[int], 
        val_indices: List[int],
        transforms: Optional[Union[Transform, TransformPipeline]] = None,
        label_transforms: Optional[Union[LabelTransform, LabelTransformPipeline]] = None
    ) -> Tuple["SubsetPatchGNNMILDataset", "SubsetPatchGNNMILDataset"]:
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
            train_labels = {train_dataset.parent_dataset.slides[train_indices[i]]: 
                           train_dataset.parent_dataset.labels[train_dataset.parent_dataset.slides[train_indices[i]]] 
                           for i in range(len(train_indices))}
            
            # Fit label transforms on training data
            if isinstance(label_transforms, FittableLabelTransform):
                label_transforms.fit(train_labels)
                logger.info(f"Fitted label transform on {len(train_labels)} training labels")
            elif isinstance(label_transforms, LabelTransformPipeline):
                label_transforms.fit(train_labels)
                logger.info(f"Fitted label transform pipeline on {len(train_labels)} training labels")
            
            # Apply to both datasets
            train_dataset.parent_dataset.label_transforms = label_transforms
        
        # Create validation dataset with the same fitted transforms
        val_dataset = self.create_subset(val_indices)

        
        logger.info(f"Created train dataset with {len(train_dataset)} samples and val dataset with {len(val_dataset)} samples")
        
        return train_dataset, val_dataset
    
    def get_normalization_params(self) -> None:
        return None
    
    def get_correlation_mask(self) -> None:
        return None


class SubsetPatchGNNMILDataset:
    """
    A lightweight wrapper for creating subsets of PatchGNNMILDataset.
    This avoids the complexity of properly initializing InMemoryDataset subsets.
    """
    
    def __init__(self, parent_dataset: PatchGNNMILDataset, indices: List[int]):
        self.parent_dataset = parent_dataset
        self.indices = indices
        
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Data:
        if idx < 0 or idx >= len(self.indices):
            raise IndexError(f"Index {idx} out of range for subset of length {len(self.indices)}")
        original_idx = self.indices[idx]
        return self.parent_dataset.get(original_idx)
    
    def get(self, idx: int) -> Data:
        """Alias for __getitem__ to match PyTorch Geometric interface."""
        return self.__getitem__(idx)
    
    @property
    def num_classes(self) -> int:
        """Get number of classes from parent dataset."""
        return self.parent_dataset.num_classes
    
    def get_num_labels(self) -> int:
        """Get number of labels from parent dataset."""
        return self.parent_dataset.get_num_labels()

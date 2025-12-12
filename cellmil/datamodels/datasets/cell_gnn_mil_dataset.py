import torch
import pandas as pd
import json
import hashlib
import random
from tqdm import tqdm
from pathlib import Path
from typing import List, Literal, Tuple, Union, Optional, Dict, Callable, Any
from torch_geometric.data import InMemoryDataset, Data  # type: ignore

from cellmil.interfaces.FeatureExtractorConfig import (
    ExtractorType,
    FeatureExtractionType,
)
from cellmil.interfaces.GraphCreatorConfig import GraphCreatorType
from cellmil.interfaces.CellSegmenterConfig import ModelType, TYPE_NUCLEI_DICT
from cellmil.utils import logger
from ..transforms import (
    Transform,
    TransformPipeline,
    FittableTransform,
    LabelTransform,
    LabelTransformPipeline,
)
from .utils import (
    wsl_preprocess,
    column_sanity_check,
    filter_split,
    preprocess_row,
    get_cell_types,
    cell_types_to_tensor,
    get_centroids,
    centroids_to_tensor,
    get_cell_features,
    load_precomputed_graph,
    merge_graph_with_features,
    cell_type_name_to_index,
    load_roi_for_slide,
    filter_cells_by_roi,
)


class CellGNNMILDataset(InMemoryDataset):
    """
    An optimized PyTorch Geometric InMemoryDataset for GNN+MIL (Graph Neural Network + Multiple Instance Learning) tasks.

    This dataset follows the official PyTorch Geometric pattern by processing all data once in the `process()` method
    and storing it efficiently. This provides significant speed improvements over the previous implementation.
    """

    def __init__(
        self,
        root: Union[str, Path],
        folder: Union[str, Path],
        label: Union[str, Tuple[str, str]],
        data: pd.DataFrame,
        extractor: Union[ExtractorType, List[ExtractorType]],
        graph_creator: GraphCreatorType,
        segmentation_model: ModelType,
        split: Literal["train", "val", "test", "all"] = "all",
        cell_type: bool = False,
        cell_types_to_keep: Optional[List[str]] = None,
        return_cell_types: bool = True,
        centroid: bool = False,
        roi_folder: Optional[Path] = None,
        max_workers: int = 8,
        transforms: Optional[TransformPipeline | Transform] = None,
        label_transforms: Optional[Union[LabelTransform, LabelTransformPipeline]] = None,
        transform: Optional[Callable[[Data], Data]] = None,
        pre_transform: Optional[Callable[[Data], Data]] = None,
        pre_filter: Optional[Callable[[Data], bool]] = None,
        force_reload: bool = False,
    ):
        """
        Initialize the optimized GNN+MIL dataset.

        Args:
            root: Root directory where the processed dataset will be cached
            folder: Path to the original dataset folder containing the raw data
            label: Label for the dataset. Either:
                - A single string (e.g., "dcr") for classification tasks
                - A tuple of two strings (e.g., ("duration", "event")) for survival prediction tasks
            data: DataFrame containing metadata
            extractor: Feature extractor type(s)
            graph_creator: Graph creator type - required for locating pre-computed graphs
            segmentation_model: Segmentation model used for cell detection
            split: Dataset split (train/val/test/all). Use "all" to include all data regardless of split
            cell_type: Whether to include cell type features
            cell_types_to_keep: Optional list of cell type names to keep (e.g., ["Neoplastic", "Connective"]).
                Valid names: "Neoplastic", "Inflammatory", "Connective", "Dead", "Epithelial" (case-insensitive).
                If provided, only cells of these types will be included. Requires cell_type=True.
                Slides with no cells matching these types will be excluded from the dataset.
            return_cell_types: Whether to store cell types in the graph data. If False, cell_types attribute is not added.
                If True and cell_type=True, graph data will have a cell_types attribute. Default is True for backward compatibility.
            centroid: Whether to include centroid features
            roi_folder: Optional path to the directory containing ROI CSV files organized by center folders.
                If provided, cells will be filtered to only include those within ROI boundaries.
                Requires 'ID', 'I3LUNG_ID', and 'CENTER' columns in the data DataFrame.
                Slides with no cells in ROI will be excluded from the dataset.
            max_workers: Number of worker threads
            transform: A function/transform that takes in a Data object and returns a transformed version
            pre_transform: A function/transform applied before caching
            pre_filter: A function that filters data objects
            force_reload: Whether to force reprocessing even if processed files exist
            transforms: Optional TransformPipeline to apply to node features at getitem time
            label_transforms: Optional LabelTransform or LabelTransformPipeline to apply to labels (e.g., TimeDiscretizerTransform for survival analysis)
        """
        # Store parameters before calling super().__init__()
        self.folder = Path(folder)
        self.label = label
        self.raw_data = data
        self.extractor = extractor
        self.graph_creator = graph_creator
        self.segmentation_model = segmentation_model
        self.split = split
        self.cell_type = cell_type
        self.return_cell_types = return_cell_types

        # Convert cell type names to indices
        if cell_types_to_keep is not None:
            self.cell_types_to_keep_indices = cell_type_name_to_index(
                cell_types_to_keep
            )
        else:
            self.cell_types_to_keep_indices = None

        self.centroid = centroid

        # ROI filtering setup
        self.roi_folder = Path(roi_folder) if roi_folder is not None else None

        # Validate ROI parameters
        if self.roi_folder is not None:
            if not self.roi_folder.exists():
                raise ValueError(f"ROI folder does not exist: {self.roi_folder}")

            # Check if data DataFrame has required columns for ROI filtering
            required_roi_columns = ["ID", "I3LUNG_ID", "CENTER"]
            missing_columns = [
                col for col in required_roi_columns if col not in data.columns
            ]

            if missing_columns:
                raise ValueError(
                    f"ROI filtering requires the following columns in data DataFrame: {required_roi_columns}. "
                    f"Missing columns: {missing_columns}"
                )

            logger.info(f"ROI filtering enabled. ROI folder: {self.roi_folder}")

        self.max_workers = max_workers
        self.force_reload = force_reload
        self.transforms = transforms
        self.label_transforms = label_transforms

        # TODO: Make this configurable
        self.wsl = False
        # TODO: -------

        if (
            isinstance(self.extractor, ExtractorType)
            and self.extractor in FeatureExtractionType.Embedding
        ) or (
            isinstance(self.extractor, list)
            and any(ext in FeatureExtractionType.Embedding for ext in self.extractor)
        ):
            raise ValueError("Embedding extractor is not supported for this dataset.")

        # Validate parameters
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

        # Initialize data structures that will be populated in process()
        self.all_slides: List[str] = []  # All slides with features
        self.slides: List[str] = []  # Only slides with labels for this task
        self.labels: Dict[str, Union[int, Tuple[float, int]]] = {}
        self.cell_types: Dict[str, Dict[int, int]] = {}
        self.centroids: Dict[str, Dict[int, Tuple[float, float]]] = {}
        self.roi_filtered_graphs: Dict[
            str, Any
        ] = {}  # ROI-filtered graphs (in-memory only)

        # Call parent constructor
        super().__init__(  # type: ignore
            root=str(root),
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
            force_reload=force_reload,
        )

        # Load the processed data
        self.load(self.processed_paths[0])

        # Extract slide information
        if not hasattr(self, "all_slides") or not self.all_slides:
            # Extract all slides from cached data
            self.all_slides = self._get_slides()

        # Extract labels fresh from DataFrame (not cached with features)
        self.labels = self._get_labels()

        # Filter slides to only those with labels for this task
        self.slides = [slide for slide in self.all_slides if slide in self.labels]

        # Filter out slides with no matching cell types if cell_types_to_keep is specified
        if self.cell_types_to_keep_indices is not None:
            logger.info("Filtering slides with no matching cell types...")
            slides_with_matching_cells: list[str] = []

            for slide_name in tqdm(self.slides, desc="Checking cell types"):
                # Get the graph data for this slide
                try:
                    actual_idx = self.all_slides.index(slide_name)
                    graph_data = super().get(actual_idx)

                    if (
                        not hasattr(graph_data, "cell_types")
                        or graph_data.cell_types is None
                    ):
                        logger.debug(
                            f"Skipping slide {slide_name}: no cell types available"
                        )
                        continue

                    cell_types_tensor = graph_data.cell_types

                    if cell_types_tensor.shape[0] == 0:
                        logger.debug(
                            f"Skipping slide {slide_name}: empty cell types tensor"
                        )
                        continue

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

                except Exception as e:
                    logger.warning(
                        f"Error checking cell types for slide {slide_name}: {e}"
                    )
                    continue

            self.slides = slides_with_matching_cells
            logger.info(f"Kept {len(self.slides)} slides with matching cell types")

        # Apply ROI filtering if enabled (after loading/processing complete dataset)
        if self.roi_folder is not None:
            self._apply_roi_filtering()

        logger.info(
            f"Loaded optimized GNN+MIL Dataset with {len(self)} total samples, {len(self.slides)} with labels for current task"
        )

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
        Return the name of the processed files that will contain our data.
        Create a stable hash based on the configuration.
        """
        # Ensure that the extractor (which is a list) is represented consistently
        extractor_str = json.dumps(self.extractor, sort_keys=True)

        config_dict: dict[str, Any] = {
            "extractor": extractor_str,  # Use stringified extractor to ensure stability
            "graph_creator": self.graph_creator,
            "segmentation_model": self.segmentation_model,
            "split": self.split,
            "cell_type": self.cell_type,
            "centroid": self.centroid,
        }

        # Convert config_dict to a JSON string, which ensures deterministic representation
        config_str = json.dumps(config_dict, sort_keys=True)

        # Use hashlib to generate a stable hash from the string
        config_hash = hashlib.md5(config_str.encode("utf-8")).hexdigest()[
            :8
        ]  # Shorten hash to 8 characters

        logger.info(f"Dataset configuration:{config_dict}")
        logger.info(f"Dataset configuration hash: {config_hash}")

        return [f"data_{self.split}_{config_hash}.pt"]

    def download(self) -> None:
        """
        Required by InMemoryDataset.
        Since we're working with pre-existing data, we don't need to download anything.
        """
        pass

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
                        all_slides.append(slide_name)

                except Exception as e:
                    raise ValueError(f"Error validating slide: {e}")

            self.all_slides = all_slides
            logger.info(f"Found {len(self.all_slides)} valid slides")

            if not self.all_slides:
                raise ValueError("No valid slides found")

            # Precompute cell types if enabled
            if self.cell_type:
                logger.info("Precomputing cell types for all slides...")
                for slide_name in tqdm(self.all_slides, desc="Loading cell types"):
                    cell_types = get_cell_types(
                        self.folder, slide_name, self.segmentation_model
                    )
                    self.cell_types[slide_name] = cell_types or {}

            # Precompute centroids if enabled
            if self.centroid:
                logger.info("Precomputing centroids for all slides...")
                for slide_name in tqdm(self.all_slides, desc="Loading centroids"):
                    centroids = get_centroids(
                        self.folder, slide_name, self.segmentation_model
                    )
                    self.centroids[slide_name] = centroids or {}

            # Process all graphs
            data_list: list[Data] = []
            for _, slide_name in enumerate(
                tqdm(
                    self.all_slides,
                    total=len(self.all_slides),
                    desc="Processing graphs",
                )
            ):
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
            if self.pre_filter is not None:  # type: ignore
                data_list = [data for data in data_list if self.pre_filter(data)]  # type: ignore
                logger.info(f"After filtering: {len(data_list)} graphs remain")

            # Apply pre_transform if provided
            if self.pre_transform is not None:  # type: ignore
                data_list = [self.pre_transform(data) for data in data_list]  # type: ignore

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
                if hasattr(data, "slide_name"):
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
                        self.graph_creator,
                        self.segmentation_model,
                        do_validate_features=False,
                    )

                    if slide_name in self.all_slides and label_value is not None:
                        labels[slide_name] = label_value

                except Exception as e:
                    raise ValueError(f"Error extracting label: {e}")

            return labels

        except Exception as e:
            logger.error(f"Failed to extract labels: {e}")
            raise

    def _apply_roi_filtering(self) -> None:
        """
        Apply ROI filtering to the loaded dataset in-memory only.
        This filters cells within ROI boundaries and keeps filtered graphs in memory.
        Also removes slides that have no cells within ROI from self.slides.
        Does NOT modify the cached files on disk.
        """
        logger.info("Applying ROI filtering to dataset (in-memory only)...")

        if self.roi_folder is None:
            raise ValueError("roi_folder must be specified for ROI filtering")

        if not self.centroid:
            raise ValueError("ROI filtering requires centroid=True")

        # Store filtered graphs in memory
        self.roi_filtered_graphs: Dict[str, Any] = {}

        slides_to_remove: list[str] = []
        total_cells_before = 0
        total_cells_after = 0

        for slide_name in tqdm(self.slides, desc="Applying ROI filtering"):
            # Get the graph data for this slide
            try:
                actual_idx = self.all_slides.index(slide_name)
                graph_data = super().get(actual_idx)

                # Clone to avoid modifying cached data
                graph_data = graph_data.clone()

                # Load ROI for this slide
                roi_df = load_roi_for_slide(slide_name, self.roi_folder, self.raw_data)

                if roi_df is None:
                    logger.debug(
                        f"No ROI found for slide {slide_name}, keeping all cells"
                    )
                    self.roi_filtered_graphs[slide_name] = graph_data
                    continue

                # Get cell centroids (original pixel coordinates)
                centroids_dict = get_centroids(
                    self.folder, slide_name, self.segmentation_model
                )

                if centroids_dict is None or len(centroids_dict) == 0:
                    logger.warning(
                        f"Could not load centroids for slide {slide_name}, keeping all cells"
                    )
                    self.roi_filtered_graphs[slide_name] = graph_data
                    continue

                # Filter cells by ROI using actual pixel coordinates
                cells_to_keep = filter_cells_by_roi(centroids_dict, roi_df)

                total_cells_before += len(centroids_dict)

                if len(cells_to_keep) == 0:
                    logger.debug(
                        f"No cells within ROI for slide {slide_name}, removing slide"
                    )
                    slides_to_remove.append(slide_name)
                    continue

                total_cells_after += len(cells_to_keep)

                # Check if graph has required attributes
                if not hasattr(graph_data, "cell_ids") or graph_data.cell_ids is None:
                    logger.warning(
                        f"Graph for {slide_name} missing cell_ids, keeping all cells"
                    )
                    self.roi_filtered_graphs[slide_name] = graph_data
                    continue

                # Convert cell_ids tensor to list and create mask based on cells_to_keep
                cell_ids_list = graph_data.cell_ids.tolist()

                # Create mask for nodes to keep (cells within ROI)
                mask = torch.tensor(
                    [cell_id in cells_to_keep for cell_id in cell_ids_list],
                    dtype=torch.bool,
                )

                if not mask.any():
                    logger.debug(
                        f"No matching cells in graph for slide {slide_name}, removing slide"
                    )
                    slides_to_remove.append(slide_name)
                    continue

                # Filter node features
                graph_data.x = graph_data.x[mask]

                # Filter position data
                if hasattr(graph_data, "pos") and graph_data.pos is not None:
                    graph_data.pos = graph_data.pos[mask]

                # Filter cell_ids
                graph_data.cell_ids = graph_data.cell_ids[mask]

                # Filter cell types if they exist
                if (
                    hasattr(graph_data, "cell_types")
                    and graph_data.cell_types is not None
                ):
                    graph_data.cell_types = graph_data.cell_types[mask]

                # Update edge indices to match filtered nodes
                # Create mapping from old indices to new indices
                num_nodes = mask.shape[0]
                old_to_new = torch.full((num_nodes,), -1, dtype=torch.long)
                old_to_new[mask] = torch.arange(mask.sum().item())

                # Filter edges: keep only edges where both nodes are in the mask
                if (
                    hasattr(graph_data, "edge_index")
                    and graph_data.edge_index is not None
                ):
                    edge_mask = (
                        mask[graph_data.edge_index[0]] & mask[graph_data.edge_index[1]]
                    )
                    graph_data.edge_index = graph_data.edge_index[:, edge_mask]
                    # Remap edge indices to new node indices
                    graph_data.edge_index = old_to_new[graph_data.edge_index]

                    # Filter edge attributes if they exist
                    if (
                        hasattr(graph_data, "edge_attr")
                        and graph_data.edge_attr is not None
                    ):
                        graph_data.edge_attr = graph_data.edge_attr[edge_mask]

                # Store filtered graph
                self.roi_filtered_graphs[slide_name] = graph_data

                logger.debug(
                    f"ROI filtering for {slide_name}: kept {len(cells_to_keep)}/{len(centroids_dict)} cells "
                    f"({len(cells_to_keep) / len(centroids_dict) * 100:.1f}%)"
                )

            except Exception as e:
                logger.error(f"Error applying ROI filtering to slide {slide_name}: {e}")
                slides_to_remove.append(slide_name)
                continue

        # Remove slides with no cells in ROI from slides list
        for slide_name in slides_to_remove:
            if slide_name in self.slides:
                self.slides.remove(slide_name)
            if slide_name in self.labels:
                del self.labels[slide_name]

        logger.info(
            f"ROI filtering complete: kept {total_cells_after}/{total_cells_before} cells "
            f"({total_cells_after / total_cells_before * 100:.1f}% if total_cells_before > 0 else 0) across {len(self.slides)} slides"
        )
        if slides_to_remove:
            logger.info(f"Removed {len(slides_to_remove)} slides with no cells in ROI")

    def _process_single_graph(self, slide_name: str) -> Optional[Data]:
        """
        Process a single graph by merging pre-computed graph structure with features.
        Labels are handled separately to enable caching across different classification tasks.

        Args:
            slide_name: Name of the slide

        Returns:
            Processed Data object without labels or None if processing fails
        """
        try:
            # Load features
            features, cell_indices, _ = get_cell_features(
                self.folder,
                slide_name,
                self.extractor,
                self.graph_creator,
                self.segmentation_model,
            )

            if cell_indices is None or features is None:
                raise ValueError(
                    f"Missing features or cell_indices for slide {slide_name}"
                )

            # Get centroids
            if len(cell_indices) == 0:
                raise ValueError(
                    f"No cell indices for slide {slide_name}, using zero centroids"
                )
                centroids = torch.zeros(features.size(0), 2, dtype=torch.float32)
            elif not self.centroid:
                centroids = torch.zeros(features.size(0), 2, dtype=torch.float32)
            else:
                centroids = self._get_centroids_tensor(slide_name, cell_indices)
                # Normalize centroids to [0, 1] per slide
                min_vals = centroids.min(dim=0).values
                max_vals = centroids.max(dim=0).values
                denom = max_vals - min_vals
                denom[denom == 0] = 1.0  # Prevent division by zero
                centroids = (centroids - min_vals) / denom

            # Store cell types as separate attribute if requested (not concatenated to features)
            cell_types_tensor: Optional[torch.Tensor] = None
            if self.cell_type:
                if len(cell_indices) == 0:
                    raise ValueError(
                        f"Missing cell_indices for slide {slide_name}; adding zero cell types."
                    )
                    n_cell_types = len(TYPE_NUCLEI_DICT)
                    cell_types_tensor = torch.zeros(
                        features.size(0), n_cell_types, dtype=torch.float32
                    )
                else:
                    cell_types_tensor = self._get_cell_types_tensor(
                        slide_name, cell_indices
                    )

            # Load pre-computed graph
            precomputed_graph = load_precomputed_graph(
                self.folder, slide_name, self.graph_creator, self.segmentation_model
            )

            # Merge graph with features
            graph_data = merge_graph_with_features(
                precomputed_graph, features, cell_indices, centroids
            )

            # Store slide name for reference
            graph_data.slide_name = slide_name

            # Store cell types as a separate attribute if enabled and return_cell_types is True
            if cell_types_tensor is not None and self.return_cell_types:
                graph_data.cell_types = cell_types_tensor

            return graph_data

        except Exception as e:
            logger.error(f"Error processing slide {slide_name}: {e}")
            return None

    def _get_cell_types_tensor(
        self, slide_name: str, cell_indices: Dict[int, int]
    ) -> torch.Tensor:
        """Get cell types tensor for a slide."""
        cell_types = self.cell_types.get(slide_name)

        if cell_types is None or len(cell_types) == 0:
            n_cells = len(cell_indices)
            n_cell_types = len(TYPE_NUCLEI_DICT)
            return torch.zeros(n_cells, n_cell_types, dtype=torch.float32)

        return cell_types_to_tensor(cell_types, cell_indices)

    def _get_centroids_tensor(
        self, slide_name: str, cell_indices: Dict[int, int]
    ) -> torch.Tensor:
        """Get centroids tensor for a slide."""
        centroids = self.centroids.get(slide_name)

        if centroids is None or len(centroids) == 0:
            n_cells = len(cell_indices)
            return torch.zeros(n_cells, 2, dtype=torch.float32)

        return centroids_to_tensor(centroids, cell_indices)

    def __len__(self) -> int:
        """Return the number of slides with labels for this task."""
        if hasattr(self, "slides") and hasattr(self, "labels") and self.labels:
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
        if hasattr(self, "slides") and idx < len(self.slides):
            slide_name = self.slides[idx]

            # Use ROI filtered graph if available
            if (
                hasattr(self, "roi_filtered_graphs")
                and slide_name in self.roi_filtered_graphs
            ):
                data = self.roi_filtered_graphs[slide_name].clone()
            else:
                # Find the actual index in the cached data
                actual_idx = self.all_slides.index(slide_name)
                # Get the original data from the parent class
                data = super().get(actual_idx)
                # Clone to avoid modifying the cached data
                data = data.clone()
        else:
            actual_idx = idx
            # Get the original data from the parent class
            data = super().get(actual_idx)
            # Clone to avoid modifying the cached data
            data = data.clone()

        # Filter by cell types if requested (only if not already filtered during process)
        if (
            self.cell_types_to_keep_indices is not None
            and hasattr(data, "cell_types")
            and data.cell_types is not None
        ):
            # Get the cell type for each cell (argmax of one-hot encoding)
            # Note: cell_types tensor uses 0-based indexing (TYPE_NUCLEI_DICT keys are 1-based)
            cell_type_tensor_indices = torch.argmax(
                data.cell_types, dim=1
            )  # (n_cells,) values 0-4

            # Convert 1-based TYPE_NUCLEI_DICT indices to 0-based tensor indices
            tensor_indices_to_keep = [
                idx - 1 for idx in self.cell_types_to_keep_indices
            ]

            # Create mask for cells to keep
            mask = torch.zeros(cell_type_tensor_indices.shape[0], dtype=torch.bool)
            for tensor_idx in tensor_indices_to_keep:
                mask |= cell_type_tensor_indices == tensor_idx

            # Apply mask to filter node features and cell_types
            data.x = data.x[mask]
            if self.return_cell_types:
                data.cell_types = data.cell_types[mask]

            # Update edge indices to match filtered nodes
            # Create mapping from old indices to new indices
            old_to_new = torch.full((mask.shape[0],), -1, dtype=torch.long)
            old_to_new[mask] = torch.arange(mask.sum().item())

            # Filter edges: keep only edges where both nodes are in the mask
            if hasattr(data, "edge_index") and data.edge_index is not None:
                edge_mask = mask[data.edge_index[0]] & mask[data.edge_index[1]]
                data.edge_index = data.edge_index[:, edge_mask]
                # Remap edge indices to new node indices
                data.edge_index = old_to_new[data.edge_index]

                # Filter edge attributes if they exist
                if hasattr(data, "edge_attr") and data.edge_attr is not None:
                    data.edge_attr = data.edge_attr[edge_mask]

        # Attach label dynamically (not cached with features)
        if (
            hasattr(self, "labels")
            and hasattr(data, "slide_name")
            and data.slide_name in self.labels
        ):
            label = self.labels[data.slide_name]
            
            # Apply label transforms if provided
            if self.label_transforms is not None:
                labels_dict = {data.slide_name: label}
                transformed = self.label_transforms.transform_labels(labels_dict)
                label = transformed[data.slide_name]
            
            data.y = torch.tensor([label], dtype=torch.long) if isinstance(label, int) else label

        # Note: Transform application is now handled by subsets to allow
        # different transforms per fold without contamination

        return data  # type: ignore

    def create_subset(self, indices: List[int]) -> "SubsetCellGNNMILDataset":
        """
        Create a subset of the dataset using the specified indices.

        This is useful for creating train/val/test splits when using split="all".
        Note: This creates a lightweight wrapper that references the original data.

        Args:
            indices: List of indices to include in the subset

        Returns:
            New SubsetCellGNNMILDataset instance containing only the specified samples

        Raises:
            ValueError: If any index is out of range
        """
        if not indices:
            raise ValueError("Indices list cannot be empty")

        max_idx = len(self) - 1
        invalid_indices = [idx for idx in indices if idx < 0 or idx > max_idx]
        if invalid_indices:
            raise ValueError(
                f"Invalid indices {invalid_indices}. Valid range is 0-{max_idx}"
            )

        # Create a simple subset wrapper
        subset = SubsetCellGNNMILDataset(self, indices)

        logger.info(
            f"Created GNN subset with {len(indices)} samples from {len(self)} total samples"
        )

        return subset

    def create_train_val_datasets(
        self,
        train_indices: List[int],
        val_indices: List[int],
        transforms: Optional[Union[TransformPipeline, Transform]] = None,
        label_transforms: Optional[Union[LabelTransform, LabelTransformPipeline]] = None,
    ) -> Tuple["SubsetCellGNNMILDataset", "SubsetCellGNNMILDataset"]:
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
            label_transforms: Optional LabelTransform or LabelTransformPipeline for labels (e.g., TimeDiscretizerTransform for survival analysis)

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
        max_idx = len(self) - 1
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
            f"Creating train/val GNN datasets with {len(train_indices)} train and {len(val_indices)} val samples"
        )

        # If no transforms provided, create simple subsets
        if transforms is None:
            train_dataset = self.create_subset(train_indices)
            val_dataset = self.create_subset(val_indices)
            return train_dataset, val_dataset

        # Always fit fittable transforms on training data
        logger.info("Fitting transforms on training data to prevent data leakage...")

        # Collect training features for fitting
        all_train_features: List[torch.Tensor] = []
        max_samples_per_slide = 100_000
        sample_size = min(len(train_indices), 20)
        indices = random.sample(train_indices, sample_size)

        for idx in indices:
            try:
                # Get graph data and extract node features
                graph_data = self.get(idx)
                features = graph_data.x  # Node features from graph

                if features is None:
                    raise ValueError(f"No features found for sample {idx}")

                # Subsample if needed to prevent memory issues
                if features.shape[0] > max_samples_per_slide:
                    perm_indices = torch.randperm(features.shape[0])[
                        :max_samples_per_slide
                    ]
                    features = features[perm_indices]

                all_train_features.append(features)

            except Exception as e:
                raise ValueError(
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

        # Store transforms on individual subset datasets, not parent dataset
        train_dataset.transforms = transforms
        val_dataset.transforms = transforms
        
        # Fit label transforms on training labels only
        if label_transforms is not None:
            from ..transforms import FittableLabelTransform
            
            # Get training labels
            train_labels = {self.slides[train_indices[i]]: self.labels[self.slides[train_indices[i]]] 
                           for i in range(len(train_indices))}
            
            # Fit label transforms on training data
            if isinstance(label_transforms, FittableLabelTransform):
                label_transforms.fit(train_labels)
                logger.info(f"Fitted label transform on {len(train_labels)} training labels")
            elif isinstance(label_transforms, LabelTransformPipeline):
                label_transforms.fit(train_labels)
                logger.info(f"Fitted label transform pipeline on {len(train_labels)} training labels")
            
            # Apply to parent dataset (shared by both train and val subsets)
            self.label_transforms = label_transforms

        logger.info(
            "Successfully created train/val GNN datasets with fitted transforms"
        )
        logger.info(f"Train dataset: {len(train_dataset)} samples")
        logger.info(f"Val dataset: {len(val_dataset)} samples")

        return train_dataset, val_dataset

    def get_num_labels(self) -> int:
        """
        Get the number of unique labels in the dataset.
        
        Note: For survival prediction tasks, this returns 0 as there are no discrete classes.
        """
        if hasattr(self, "labels") and self.labels:
            # Check if we have survival data by looking at the first label
            first_label = next(iter(self.labels.values()))
            if isinstance(first_label, tuple):
                # Survival data - no discrete classes
                return 0
            # Classification data
            return len(set(self.labels.values()))
        return self.num_classes


class SubsetCellGNNMILDataset:
    """
    A lightweight wrapper for creating subsets of CellGNNMILDataset.
    This avoids the complexity of properly initializing InMemoryDataset subsets.
    """

    def __init__(self, parent_dataset: CellGNNMILDataset, indices: List[int]):
        self.parent_dataset = parent_dataset
        self.indices = indices
        self.transforms: Optional[Union[TransformPipeline, Transform]] = (
            None  # Allow subset to have its own transforms
        )
        # Note: cell_types_to_keep filtering is handled by parent dataset's get() method

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Data:
        if idx < 0 or idx >= len(self.indices):
            raise IndexError(
                f"Index {idx} out of range for subset of length {len(self.indices)}"
            )
        original_idx = self.indices[idx]

        # Get the data from parent dataset
        data = self.parent_dataset.get(original_idx)

        # Apply subset-specific transforms if they exist
        if self.transforms is not None and hasattr(data, "x") and data.x is not None:
            # Clone to avoid modifying the cached data
            data = data.clone()
            data.x = self.transforms.transform(data.x)  # type: ignore

        return data

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

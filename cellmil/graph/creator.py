import torch
import math
import torch_cluster  # type: ignore
import numpy as np
from tqdm import tqdm
from abc import ABC, abstractmethod
from typing import List, Any, Tuple, Callable, cast, Literal
from cellmil.utils import logger
from cellmil.interfaces.GraphCreatorConfig import GraphCreatorType
from scipy.spatial import Delaunay, cKDTree  # type: ignore
import cv2


class Creator:
    """Base class for graph creation."""

    def __init__(self, method: GraphCreatorType, device: str):
        self.method = method
        self.device = device

        # TODO: Make configurable
        if method == "knn":
            self.edge_creator = KNNEdgeCreator(device, k=8)
        elif method == "radius":
            self.edge_creator = RadiusEdgeCreator(device, radius=100)
        elif method == "delaunay_radius":
            self.edge_creator = DelaunayEdgeCreator(device, limit_radius=4000)
        elif method == "similarity":
            self.edge_creator = SimilarityEdgeCreator(
                device,
                similarity_threshold=3,
                alpha=0.6,
                distance_sigma=300.0,
                combination_method="additive",
                distance_metric="gaussian",
                feature_metric="gaussian",
                feature_sigma=1,
                # device, similarity_threshold=0.8, alpha=0.8, distance_sigma=500.0
            )
        else:
            self.edge_creator = DilateEdgeCreator(device, dilation=40)
        # TODO: --------

    def create(
        self, cells: List[dict[str, Any]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create graph from cells data."

        Args:
            cells: List of cell dictionaries

        Returns:
            node_features: Tensor of shape [N, F] where N is number of cells and F is feature dimension
            edge_indices: Tensor of shape [2, E] where E is number of edges
            edge_features: Tensor of shape [E, F] where F is edge feature dimension
        """
        if not cells or len(cells) == 0:
            logger.warning("No cells found. Creating empty graph.")
            return self._create_empty_graph()

        n_cells = len(cells)
        logger.info(f"Creating graph for {n_cells} cells")

        logger.info("Extracting node features...")
        node_features, positions = self._extract_node_features(cells)

        positions = positions.to(self.device)

        logger.info("Creating edges...")
        edge_indices, edge_features = self.edge_creator.create_edges(positions, cells)

        # Move results back to CPU if needed
        if self.device != "cpu":
            edge_indices = edge_indices.cpu()
            edge_features = edge_features.cpu()
            positions = positions.cpu()

        logger.info(
            f"Created graph with {node_features.shape[0]} nodes and {edge_indices.shape[1]} edges"
        )

        return node_features, edge_indices, edge_features

    def _extract_node_features(
        self, cells: List[dict[str, Any]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract node features from cells."""
        n_cells = len(cells)

        # Pre-allocate tensors for efficiency
        node_features = torch.zeros((n_cells, 1), dtype=torch.long)
        positions = torch.zeros((n_cells, 2), dtype=torch.float32)

        # Vectorized extraction using numpy for speed
        centroids = np.array([cell["centroid"] for cell in cells])
        cell_ids = np.array([cell["cell_id"] for cell in cells])

        # Store in tensors
        node_features[:, 0] = torch.from_numpy(cell_ids)  # type: ignore
        positions[:, 0] = torch.from_numpy(centroids[:, 0])  # type: ignore
        positions[:, 1] = torch.from_numpy(centroids[:, 1])  # type: ignore

        logger.info(f"Extracted features for {n_cells} cells")
        return node_features, positions

    def _create_empty_graph(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create an empty graph when no cells are found."""
        return (
            torch.empty((0, 1), dtype=torch.long),  # node_features (cell_id)
            torch.empty((2, 0), dtype=torch.long),  # edge_indices
            torch.empty((0, 3), dtype=torch.float32),  # edge features
        )


class EdgeCreator(ABC):
    """Abstract base class for edge creation between cells.

    This class provides a framework for creating different types of graphs from cell position data.
    It implements common functionality like batched processing and edge feature calculation,
    while allowing subclasses to define specific edge creation strategies.

    The class supports various graph creation methods:
    - KNN: Connect each cell to its k nearest neighbors
    - Radius: Connect cells within a specified radius
    - Delaunay + Radius: Use Delaunay triangulation with distance filtering
    - Dilate: Dilate the nuclei to approximate cell boundaries.

    All edge creators produce graphs with:
    - Node features: Cell IDs
    - Edge features: [distance, direction_x, direction_y] where direction is a unit vector

    Attributes:
        device (str): Computing device ('cpu' or 'cuda:X')
        batch_size (int): Number of cells to process per batch for memory efficiency
        k (int | None): Number of nearest neighbors for KNN method
        radius (float | None): Maximum distance for radius-based connections
        limit_radius (float | None): Maximum distance filter for Delaunay triangulation
        dilation (int | None): Dilation factor of nuclei for edge creation
    """

    def __init__(
        self,
        device: str,
        k: int | None = None,
        radius: float | None = None,
        limit_radius: float | None = None,
        dilation: int | None = None,
        batch_size: int = 2_000_000,
    ):
        self.device = device
        self.batch_size = batch_size
        self.k = k
        self.radius = radius
        self.limit_radius = limit_radius
        self.dilation = dilation

    @abstractmethod
    def create_edges(
        self, positions: torch.Tensor, cells: List[dict[str, Any]] | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create edges between cells.

        Args:
            positions: Tensor of shape [N, 2] containing the (x, y) positions of the cells
            cells: Optional list of cell dictionaries for contour-based methods

        Returns:
            Tuple of (edge_indices, edge_features) where:
            - edge_indices: Tensor of shape (2, num_edges) containing source and target node indices
            - edge_features: Tensor of shape (num_edges, feature_dim) containing edge features
        """
        pass

    def _process_batched(
        self,
        positions: torch.Tensor,
        edge_computation_fn: Callable[
            [torch.Tensor], Tuple[torch.Tensor, torch.Tensor]
        ],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process positions in batches and compute edges.

        Args:
            positions: Tensor of shape [N, 2] containing the (x, y) positions of the cells
            edge_computation_fn: Function that takes batch_positions and returns (edge_indices, edge_features)

        Returns:
            Tuple of (edge_indices, edge_features)
        """
        n_cells = positions.shape[0]
        logger.info(f"Processing {n_cells} cells in batches of {self.batch_size}")

        all_edge_indices: list[torch.Tensor] = []
        all_edge_features: list[torch.Tensor] = []

        # Calculate total number of batches for progress bar
        total_batches = math.ceil(n_cells / self.batch_size)

        # Progress bar for batch processing
        batch_iterator = range(0, n_cells, self.batch_size)
        progress_bar = tqdm(
            batch_iterator, desc="Processing batches", unit="batch", total=total_batches
        )

        for start_idx in progress_bar:
            end_idx = min(start_idx + self.batch_size, n_cells)
            batch_positions = positions[start_idx:end_idx]

            # Update progress bar description
            progress_bar.set_description(
                f"Processing batch {start_idx // self.batch_size + 1}/{total_batches}"
            )

            # Compute edges for this batch using the provided function
            batch_edge_indices, batch_edge_features = edge_computation_fn(
                batch_positions
            )

            # Adjust indices to global indexing and collect results
            if batch_edge_indices.shape[1] > 0:
                batch_edge_indices = batch_edge_indices + start_idx
                all_edge_indices.append(batch_edge_indices)
                all_edge_features.append(batch_edge_features)

            # Clear GPU cache to manage memory
            if self.device != "cpu":
                torch.cuda.empty_cache()

        # Concatenate all batches
        if all_edge_indices:
            edge_indices = torch.cat(all_edge_indices, dim=1)
            edge_features = torch.cat(all_edge_features, dim=0)
        else:
            edge_indices = torch.empty((2, 0), dtype=torch.long, device=self.device)
            edge_features = torch.empty((0, 3), dtype=torch.float32, device=self.device)

        logger.info(
            f"Completed batched processing: {edge_indices.shape[1]} total edges"
        )
        return edge_indices, edge_features

    def _calculate_edge_features(
        self, edge_indices: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        """Calculate edge features based on positions.

        Args:
            edge_indices: Tensor of shape (2, num_edges) containing source and target node indices
            positions: Tensor of shape (N, 2) containing the (x, y) positions of the cells

        Returns:
            Tensor of shape (num_edges, 3) containing edge features:
            - distance: Euclidean distance between cells
            - direction_x: x-component of unit direction vector (from source to target)
            - direction_y: y-component of unit direction vector (from source to target)
        """
        src_positions = positions[edge_indices[0].long()]
        dst_positions = positions[edge_indices[1].long()]

        # Calculate direction vector (from source to target)
        direction_vector = dst_positions - src_positions

        # Calculate distances
        distances = cast(torch.Tensor, direction_vector.norm(dim=1, keepdim=True))  # type: ignore

        # Calculate unit direction vector, handling zero distances
        unit_direction = torch.zeros_like(direction_vector)
        non_zero_mask = (
            distances.squeeze() > 1e-8
        )  # Small epsilon to avoid division by zero

        if non_zero_mask.any():
            unit_direction[non_zero_mask] = (
                direction_vector[non_zero_mask] / distances[non_zero_mask]
            )

        # Combine distance and direction features
        edge_features = torch.cat([distances, unit_direction], dim=1)

        return edge_features


class KNNEdgeCreator(EdgeCreator):
    """KNN-based edge creator for cell graphs."""

    def create_edges(
        self, positions: torch.Tensor, cells: List[dict[str, Any]] | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logger.info("Creating KNN edges...")
        if self.k is None:
            raise ValueError("K value is not set")

        def _compute_knn_batch(
            batch_positions: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """Compute KNN edges for a batch."""
            batch_n_cells = batch_positions.shape[0]
            batch_actual_k = (
                min(self.k if self.k is not None else 0, batch_n_cells - 1)
                if batch_n_cells > 1
                else 0
            )

            if batch_actual_k == 0:
                logger.info(
                    "Too few cells for KNN graph creation, creating empty graph"
                )
                return (
                    torch.empty((2, 0), dtype=torch.long, device=self.device),
                    torch.empty((0, 3), dtype=torch.float32, device=self.device),
                )

            logger.info(
                f"Creating KNN graph with k={batch_actual_k} (requested k={self.k}, n_cells={batch_n_cells})"
            )

            # Use torch_cluster directly for KNN
            batch_edge_indices = torch_cluster.knn_graph(
                batch_positions, k=batch_actual_k, loop=False
            )

            # Calculate edge features
            batch_edge_features = self._calculate_edge_features(
                batch_edge_indices, batch_positions
            )

            logger.info(f"Created {batch_edge_indices.shape[1]} KNN edges")

            return batch_edge_indices, batch_edge_features

        # Use the generic batched processing
        return self._process_batched(positions, _compute_knn_batch)


class RadiusEdgeCreator(EdgeCreator):
    """Radius-based edge creator for cell graphs."""

    def create_edges(
        self, positions: torch.Tensor, cells: List[dict[str, Any]] | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logger.info("Creating radius-based edges...")

        if self.radius is None:
            raise ValueError("Radius value is not set")

        def _compute_radius_batch(
            batch_positions: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """Compute radius edges for a batch."""
            batch_n_cells = batch_positions.shape[0]
            if batch_n_cells == 0:
                logger.info("No cells in batch, creating empty graph")
                return (
                    torch.empty((2, 0), dtype=torch.long, device=self.device),
                    torch.empty((0, 3), dtype=torch.float32, device=self.device),
                )

            batch_actual_radius = self.radius if self.radius is not None else 0

            logger.info(
                f"Creating radius graph with radius={batch_actual_radius} (n_cells={batch_n_cells})"
            )

            # Use torch_cluster directly for radius
            batch_edge_indices = torch_cluster.radius_graph(
                batch_positions, r=batch_actual_radius, loop=False
            )

            # Calculate edge features
            batch_edge_features = self._calculate_edge_features(
                batch_edge_indices, batch_positions
            )

            logger.info(f"Created {batch_edge_indices.shape[1]} radius edges")

            return batch_edge_indices, batch_edge_features

        # Use the generic batched processing
        return self._process_batched(positions, _compute_radius_batch)


class DelaunayEdgeCreator(EdgeCreator):
    """Delaunay + Radius-based edge creator for cell graphs."""

    def create_edges(
        self, positions: torch.Tensor, cells: List[dict[str, Any]] | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logger.info("Creating Delaunay triangulation edges...")
        logger.info("This method could be slow since it's not done in GPU")

        if self.limit_radius is None:
            raise ValueError("Limit radius value is not set")

        n_cells = positions.shape[0]

        if n_cells < 3:
            logger.info(
                "Too few cells for Delaunay triangulation, creating empty graph"
            )
            return (
                torch.empty((2, 0), dtype=torch.long, device=self.device),
                torch.empty((0, 3), dtype=torch.float32, device=self.device),
            )

        logger.info(
            f"Creating Delaunay triangulation with limit_radius={self.limit_radius} (n_cells={n_cells})"
        )

        # Convert to numpy for Delaunay triangulation (always done on CPU)
        positions_np = cast(
            np.ndarray[Any, Any],
            positions.cpu().numpy(),  # type: ignore
        )

        # Create Delaunay triangulation
        tri = Delaunay(positions_np)

        # Extract edges from triangulation
        edges_set: set[tuple[int]] = set()
        for simplex in tqdm(tri.simplices, desc="Processing Delaunay triangles"):
            # Add all edges from the triangle
            for i in range(3):
                for j in range(i + 1, 3):
                    edge = tuple(sorted([simplex[i], simplex[j]]))
                    edges_set.add(edge)

        if not edges_set:
            logger.info("No edges found in Delaunay triangulation")
            return (
                torch.empty((2, 0), dtype=torch.long, device=self.device),
                torch.empty((0, 3), dtype=torch.float32, device=self.device),
            )

        # Convert edges to tensor
        edges_list = list(edges_set)
        edge_indices = torch.tensor(
            edges_list, dtype=torch.long, device=self.device
        ).t()

        # Calculate edge features
        edge_features = self._calculate_edge_features(edge_indices, positions)

        # Filter edges by radius limit
        distances = edge_features[:, 0]  # First column is distance
        radius_mask = distances <= self.limit_radius

        # Apply radius filter
        filtered_edge_indices = edge_indices[:, radius_mask]
        filtered_edge_features = edge_features[radius_mask]

        logger.info(
            f"Created {edge_indices.shape[1]} Delaunay edges, filtered to {filtered_edge_indices.shape[1]} within radius {self.limit_radius}"
        )

        return filtered_edge_indices, filtered_edge_features


class DilateEdgeCreator(EdgeCreator):
    def create_edges(
        self, positions: torch.Tensor, cells: List[dict[str, Any]] | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create edges between cells whose dilated contours intersect.

        Args:
            positions (torch.Tensor): Cell centroid positions [N, 2]
            cells (List[dict] | None): List of cell dictionaries containing contour information

        Returns:
            edge_indices (torch.Tensor): Edge indices [2, E]
            edge_features (torch.Tensor): Edge features [E, 3] (distance, dx, dy)
        """
        logger.info(
            "The following process could take some time since it's run on the CPU"
        )

        if cells is None:
            logger.warning("No cell data provided for contour-based edge creation")
            return torch.zeros(
                (2, 0), dtype=torch.long, device=self.device
            ), torch.zeros((0, 3), dtype=torch.float32, device=self.device)

        n_cells = positions.shape[0]
        if n_cells < 2:
            logger.info("Less than 2 cells, creating empty edge list")
            return torch.zeros(
                (2, 0), dtype=torch.long, device=self.device
            ), torch.zeros((0, 3), dtype=torch.float32, device=self.device)

        # Make sure we have a valid dilation value
        dilation_px = self.dilation
        if dilation_px is None:
            dilation_px = 40  # Default fallback

        logger.info(
            f"Creating dilated contour edges for {n_cells} cells with dilation={dilation_px}px"
        )

        # Extract contours and prepare for individual mask processing
        edge_list: List[List[int]] = []

        # First pass: collect valid cells with contours
        valid_cells: list[tuple[int, dict[str, Any]]] = []
        centroids: list[list[float]] = []
        for i, cell in enumerate(cells):
            contour = cell.get("contour", [])
            if contour and len(contour) >= 3:
                valid_cells.append((i, cell))
                centroids.append(cell["centroid"])

        if not valid_cells:
            logger.warning("No valid contours found, creating empty edge list")
            return torch.zeros(
                (2, 0), dtype=torch.long, device=self.device
            ), torch.zeros((0, 3), dtype=torch.float32, device=self.device)

        # Build spatial index for efficient neighbor finding
        centroids_array = np.array(centroids)
        tree = cKDTree(centroids_array)

        # Conservative search radius: 2 * dilation + reasonable cell size estimate
        max_cell_size = 150  # pixels
        search_radius = 2 * dilation_px + max_cell_size

        logger.info(
            f"Processing {len(valid_cells)} cells with individual masks (search radius: {search_radius}px)"
        )

        def create_individual_mask(
            cell_data: tuple[int, dict[str, Any]],
        ) -> tuple[int, np.ndarray[Any, Any], int, int]:
            """Create individual dilated mask for a single cell."""
            original_idx, cell = cell_data
            contour = cell.get("contour", [])

            # Convert contour to numpy array
            contour_np = np.array(contour, dtype=np.int32)

            # Calculate bounding box for this cell only + dilation margin
            min_x, min_y = contour_np.min(axis=0) - dilation_px - 5
            max_x, max_y = contour_np.max(axis=0) + dilation_px + 5

            # Create small mask size for individual cell (typically ~100-200 pixels)
            mask_width = max_x - min_x
            mask_height = max_y - min_y

            # Adjust contour coordinates to local mask coordinates
            local_contour = contour_np.copy()
            local_contour[:, 0] -= min_x
            local_contour[:, 1] -= min_y

            # Create and fill mask
            mask = np.zeros((mask_height, mask_width), dtype=np.uint8)
            cv2.fillPoly(mask, [local_contour], (255,))

            # Dilate the mask
            if dilation_px > 0:
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (2 * dilation_px + 1, 2 * dilation_px + 1)
                )
                mask = cv2.dilate(mask, kernel, iterations=1)

            return original_idx, mask, min_x, min_y

        # Create individual masks for all valid cells
        cell_masks: dict[
            int, tuple[np.ndarray[Any, Any], int, int]
        ] = {}  # Dict: original_cell_index -> (mask, min_x, min_y)
        for cell_data in tqdm(valid_cells, desc="Creating individual dilated masks"):
            original_idx, mask, min_x, min_y = create_individual_mask(cell_data)
            cell_masks[original_idx] = (mask, min_x, min_y)

        logger.info(
            f"Using KDTree spatial indexing with search radius={search_radius:.1f}px for efficient intersection detection"
        )

        # Check intersections between nearby cells using individual masks
        for idx, (original_i, _) in enumerate(
            tqdm(valid_cells, desc="Detecting intersections")
        ):
            # Query nearby cells within potential interaction distance
            nearby_indices = cast(
                list[int],
                tree.query_ball_point(centroids_array[idx], search_radius),  # type: ignore
            )

            for nearby_idx in nearby_indices:
                if (
                    nearby_idx > idx
                ):  # Only check each pair once, avoid self-intersection
                    original_j, _ = valid_cells[nearby_idx]

                    # Get masks and their global coordinates
                    mask_i, min_x_i, min_y_i = cell_masks[original_i]
                    mask_j, min_x_j, min_y_j = cell_masks[original_j]

                    # Calculate bounding boxes in global coordinates
                    max_x_i = min_x_i + mask_i.shape[1]
                    max_y_i = min_y_i + mask_i.shape[0]
                    max_x_j = min_x_j + mask_j.shape[1]
                    max_y_j = min_y_j + mask_j.shape[0]

                    # Quick rejection test: check if bounding boxes overlap
                    if (
                        max_x_i <= min_x_j
                        or max_x_j <= min_x_i
                        or max_y_i <= min_y_j
                        or max_y_j <= min_y_i
                    ):
                        continue  # No overlap, skip expensive intersection test

                    # Calculate intersection region in global coordinates
                    intersect_min_x = max(min_x_i, min_x_j)
                    intersect_min_y = max(min_y_i, min_y_j)
                    intersect_max_x = min(max_x_i, max_x_j)
                    intersect_max_y = min(max_y_i, max_y_j)

                    if (
                        intersect_max_x <= intersect_min_x
                        or intersect_max_y <= intersect_min_y
                    ):
                        continue  # No valid intersection region

                    # Extract overlapping regions from both masks (convert to local coordinates)
                    region_i = mask_i[
                        intersect_min_y - min_y_i : intersect_max_y - min_y_i,
                        intersect_min_x - min_x_i : intersect_max_x - min_x_i,
                    ]
                    region_j = mask_j[
                        intersect_min_y - min_y_j : intersect_max_y - min_y_j,
                        intersect_min_x - min_x_j : intersect_max_x - min_x_j,
                    ]

                    # Check if there's actual pixel intersection
                    if region_i.shape == region_j.shape and np.any(
                        cv2.bitwise_and(region_i, region_j) > 0
                    ):
                        edge_list.append([original_i, original_j])

        if not edge_list:
            logger.info(
                "No intersecting dilated contours found, creating empty edge list"
            )
            return torch.zeros(
                (2, 0), dtype=torch.long, device=self.device
            ), torch.zeros((0, 3), dtype=torch.float32, device=self.device)

        # Convert to tensors
        edge_indices = torch.tensor(edge_list, dtype=torch.long).to(self.device).T

        # Calculate edge features (distances and directions)
        edge_features = self._calculate_edge_features(
            edge_indices, positions.to(self.device)
        )

        logger.info(f"Created {edge_indices.shape[1]} edges from dilated contours")
        return edge_indices, edge_features


class SimilarityEdgeCreator(EdgeCreator):
    """Similarity-based edge creator for cell graphs.

    Creates edges based on both spatial distance and morphological feature similarity.
    Uses correlation filtering to reduce feature redundancy before computing similarity.
    """

    def __init__(
        self,
        device: str,
        similarity_threshold: float = 0.5,
        distance_sigma: float = 200.0,
        alpha: float = 0.5,
        combination_method: Literal["additive", "multiplicative"] = "additive",
        distance_metric: Literal[
            "gaussian", "laplacian", "inverse", "inverse_square"
        ] = "gaussian",
        feature_metric: Literal[
            "cosine", "correlation", "euclidean", "gaussian"
        ] = "cosine",
        feature_sigma: float = 1.0,
        batch_size: int = 1024,
        max_gpu_memory_fraction: float = 0.8,
    ):
        """Initialize similarity-based edge creator.

        Args:
            device: Computing device ('cpu' or 'cuda:X')
            similarity_threshold: If < 1, used as threshold filter. If >= 1 (integer), used as KNN parameter
            distance_sigma: Gaussian kernel width for distance-based similarity
            alpha: Weight for similarity vs distance (0=distance only, 1=similarity only)
            combination_method: Method to combine similarity and distance ('additive' or 'multiplicative')
            distance_metric: Metric for distance-based similarity ('gaussian', 'laplacian', 'inverse', 'inverse_square')
            feature_metric: Metric for feature-based similarity ('cosine', 'correlation', 'euclidean', 'gaussian')
            feature_sigma: Gaussian kernel width for feature-based similarity (only used when feature_metric='gaussian')
            batch_size: Number of cells to process per batch (will be adjusted dynamically based on GPU memory)
            max_gpu_memory_fraction: Maximum fraction of available GPU memory to use (default 0.8)
        """
        super().__init__(device, batch_size=batch_size)
        self.similarity_threshold = similarity_threshold
        self.distance_sigma = distance_sigma
        self.alpha = alpha
        self.combination_method = combination_method
        self.distance_metric = distance_metric
        self.feature_metric = feature_metric
        self.feature_sigma = feature_sigma
        self.max_gpu_memory_fraction = max_gpu_memory_fraction

        # Determine if we're using threshold or KNN mode
        self.use_knn_mode = similarity_threshold >= 1.0
        if self.use_knn_mode:
            self.k = int(similarity_threshold)
            log_msg = (
                f"SimilarityEdgeCreator initialized in KNN mode with k={self.k}, "
                f"method={combination_method}, distance_metric={distance_metric} (distance_sigma={distance_sigma}), "
                f"feature_metric={feature_metric}"
            )
            if feature_metric == "gaussian":
                log_msg += f" (feature_sigma={feature_sigma})"
            logger.info(log_msg)
        else:
            log_msg = (
                f"SimilarityEdgeCreator initialized in threshold mode with threshold={similarity_threshold}, "
                f"method={combination_method}, distance_metric={distance_metric} (distance_sigma={distance_sigma}), "
                f"feature_metric={feature_metric}"
            )
            if feature_metric == "gaussian":
                log_msg += f" (feature_sigma={feature_sigma})"
            logger.info(log_msg)

    def _get_available_gpu_memory(self) -> float:
        """Get available GPU memory in bytes.

        Returns:
            Available memory in bytes, or 0 if not on GPU
        """
        if not self.device.startswith("cuda"):
            return 0.0

        try:
            free_mem, _ = torch.cuda.mem_get_info(torch.device(self.device))
            return float(free_mem)
        except Exception as e:
            logger.warning(f"Could not get GPU memory info: {e}")
            return 0.0

    def _estimate_memory_usage(
        self, batch_size: int, n_cells: int, feature_dim: int
    ) -> float:
        """Estimate memory usage for processing a batch.

        Args:
            batch_size: Number of cells in batch
            n_cells: Total number of cells
            feature_dim: Feature dimension

        Returns:
            Estimated memory usage in bytes
        """
        # Main memory consumers:
        # 1. positions diff: [batch_size, n_cells, 2] * 4 bytes (float32)
        # 2. distances: [batch_size, n_cells] * 4 bytes
        # 3. feature similarity: [batch_size, n_cells] * 4 bytes
        # 4. edge_weights: [batch_size, n_cells] * 4 bytes
        # 5. Intermediate feature computations: varies by metric

        bytes_per_float = 4

        # Position-based tensors
        positions_mem = batch_size * n_cells * 2 * bytes_per_float  # diff tensor
        distances_mem = batch_size * n_cells * bytes_per_float

        # Feature-based tensors (estimate worst case)
        if self.feature_metric in ["cosine", "correlation"]:
            # Normalized features + matmul result
            feature_mem = batch_size * feature_dim * bytes_per_float  # normalized batch
            feature_mem += n_cells * feature_dim * bytes_per_float  # normalized all
            feature_mem += batch_size * n_cells * bytes_per_float  # similarity matrix
        elif self.feature_metric in ["euclidean", "gaussian"]:
            # Expanded tensors for pairwise distance
            feature_mem = batch_size * n_cells * feature_dim * bytes_per_float * 2
            feature_mem += batch_size * n_cells * bytes_per_float  # distance result
        else:
            feature_mem = batch_size * n_cells * bytes_per_float

        # Edge weights and masks
        weights_mem = batch_size * n_cells * bytes_per_float

        # Add 30% overhead for intermediate operations and gradients (even in inference)
        total_mem = (positions_mem + distances_mem + feature_mem + weights_mem) * 1.3

        return total_mem

    def _calculate_safe_batch_size(
        self, n_cells: int, feature_dim: int, initial_batch_size: int
    ) -> tuple[int, str]:
        """Calculate a safe batch size that won't exceed GPU memory.

        Args:
            n_cells: Total number of cells
            feature_dim: Feature dimension
            initial_batch_size: Requested batch size

        Returns:
            Tuple of (safe_batch_size, device_to_use)
        """
        if not self.device.startswith("cuda"):
            return initial_batch_size, self.device

        available_mem = self._get_available_gpu_memory()
        if available_mem == 0:
            logger.warning("Could not determine GPU memory, using CPU fallback")
            return initial_batch_size, "cpu"

        usable_mem = available_mem * self.max_gpu_memory_fraction

        # Binary search for maximum safe batch size
        min_batch = 1
        max_batch = initial_batch_size
        safe_batch = min_batch

        while min_batch <= max_batch:
            mid_batch = (min_batch + max_batch) // 2
            estimated_mem = self._estimate_memory_usage(mid_batch, n_cells, feature_dim)

            if estimated_mem <= usable_mem:
                safe_batch = mid_batch
                min_batch = mid_batch + 1
            else:
                max_batch = mid_batch - 1

        # Check if even smallest batch would exceed memory
        min_mem = self._estimate_memory_usage(1, n_cells, feature_dim)
        if min_mem > usable_mem:
            logger.warning(
                f"Even batch_size=1 would use {min_mem / 1e9:.2f}GB, "
                f"but only {usable_mem / 1e9:.2f}GB available. Falling back to CPU."
            )
            return initial_batch_size, "cpu"

        if safe_batch < initial_batch_size:
            logger.info(
                f"Reduced batch size from {initial_batch_size} to {safe_batch} "
                f"to fit in {usable_mem / 1e9:.2f}GB GPU memory "
                f"(estimated {self._estimate_memory_usage(safe_batch, n_cells, feature_dim) / 1e9:.2f}GB usage)"
            )

        return safe_batch, self.device

    def _compute_distance_similarity(self, distances: torch.Tensor) -> torch.Tensor:
        """Compute distance-based similarity using the specified metric.

        Args:
            distances: Tensor of pairwise distances

        Returns:
            Similarity values in range [0, 1]
        """
        if self.distance_metric == "gaussian":
            # Gaussian kernel: exp(-d^2 / (2*distance_sigma^2))
            return torch.exp(-(distances**2) / (2 * self.distance_sigma**2))

        elif self.distance_metric == "laplacian":
            # Laplacian kernel: exp(-d / distance_sigma)
            return torch.exp(-distances / self.distance_sigma)

        elif self.distance_metric == "inverse":
            # Inverse distance: 1 / (1 + d)
            return 1.0 / (1.0 + distances)

        elif self.distance_metric == "inverse_square":
            # Inverse square distance: 1 / (1 + d^2)
            return 1.0 / (1.0 + distances**2)

        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")

    def _compute_feature_similarity(
        self, batch_features: torch.Tensor, all_features: torch.Tensor
    ) -> torch.Tensor:
        """Compute feature-based similarity using the specified metric.

        Args:
            batch_features: Tensor of shape [batch_size, feature_dim]
            all_features: Tensor of shape [n_cells, feature_dim]

        Returns:
            Similarity matrix of shape [batch_size, n_cells]
        """
        if self.feature_metric == "cosine":
            # Cosine similarity
            batch_norm = torch.nn.functional.normalize(batch_features, p=2, dim=1)
            all_norm = torch.nn.functional.normalize(all_features, p=2, dim=1)
            return torch.mm(batch_norm, all_norm.t())

        elif self.feature_metric == "correlation":
            # Pearson correlation coefficient
            # Center the features
            batch_centered = batch_features - batch_features.mean(dim=1, keepdim=True)
            all_centered = all_features - all_features.mean(dim=1, keepdim=True)

            # Normalize
            batch_norm = torch.nn.functional.normalize(batch_centered, p=2, dim=1)
            all_norm = torch.nn.functional.normalize(all_centered, p=2, dim=1)

            return torch.mm(batch_norm, all_norm.t())

        elif self.feature_metric == "euclidean":
            # Euclidean distance converted to similarity: exp(-distance)
            # Compute pairwise euclidean distances
            batch_expanded = batch_features.unsqueeze(1)  # [batch, 1, features]
            all_expanded = all_features.unsqueeze(0)  # [1, n_cells, features]
            distances = cast(
                torch.Tensor, torch.norm(batch_expanded - all_expanded, dim=2) # type: ignore
            )  

            # Convert to similarity
            return torch.exp(-distances)

        elif self.feature_metric == "gaussian":
            # Gaussian kernel on feature space: exp(-||f_i - f_j||^2 / (2*feature_sigma^2))
            # Compute pairwise euclidean distances
            batch_expanded = batch_features.unsqueeze(1)  # [batch, 1, features]
            all_expanded = all_features.unsqueeze(0)  # [1, n_cells, features]
            distances_sq = cast(
                torch.Tensor, torch.sum((batch_expanded - all_expanded) ** 2, dim=2)
            )  # type: ignore

            # Apply gaussian kernel (using feature_sigma)
            return torch.exp(-distances_sq / (2 * self.feature_sigma**2))

        else:
            raise ValueError(f"Unknown feature metric: {self.feature_metric}")

    def create_edges(
        self, positions: torch.Tensor, cells: List[dict[str, Any]] | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create edges based on distance and feature similarity.

        Args:
            positions: Tensor of shape [N, 2] containing the (x, y) positions of the cells
            cells: List of cell dictionaries with features for similarity computation

        Returns:
            edge_indices: Tensor of shape [2, E] containing source and target node indices
            edge_features: Tensor of shape [E, 3] containing [distance, direction_x, direction_y]
                If similarity_threshold < 1: edges are filtered by weight threshold
                If similarity_threshold >= 1: top-k edges per node are kept (KNN mode)
                Weight = alpha * max(0, cosine_similarity) + (1 - alpha) * exp(-distance^2 / (2*sigma^2))
        """
        logger.info("Creating similarity-based edges...")

        if cells is None:
            logger.warning("No cell data provided for similarity-based edge creation")
            return torch.empty(
                (2, 0), dtype=torch.long, device=self.device
            ), torch.empty((0, 3), dtype=torch.float32, device=self.device)

        n_cells = positions.shape[0]

        if n_cells < 2:
            logger.info("Less than 2 cells, creating empty edge list")
            return torch.empty(
                (2, 0), dtype=torch.long, device=self.device
            ), torch.empty((0, 3), dtype=torch.float32, device=self.device)

        logger.info(f"Creating similarity edges for {n_cells} cells")

        # Extract morphological features from cells
        features_list: list[torch.Tensor] = []
        for cell in cells:
            if "features" not in cell:
                raise ValueError(
                    "Cell missing 'features' key required for similarity computation"
                )
            features_list.append(cell["features"])

        # Convert to tensor and normalize
        features = torch.stack(features_list).to(self.device)  # Shape: [N, F]
        feature_dim = features.shape[1]

        logger.info(f"Extracted {feature_dim} features per cell")

        # Calculate safe batch size and decide device
        safe_batch_size, compute_device = self._calculate_safe_batch_size(
            n_cells, feature_dim, self.batch_size
        )

        # Move data to compute device if needed
        if compute_device != self.device:
            logger.info(
                f"Moving computation to {compute_device} due to memory constraints"
            )
            positions = positions.to(compute_device)
            features = features.to(compute_device)

        # Compute all pairwise distances and similarities
        all_edge_indices: list[torch.Tensor] = []
        all_edge_features: list[torch.Tensor] = []

        # Calculate total number of batches
        total_batches = math.ceil(n_cells / safe_batch_size)

        logger.info(
            f"Processing {n_cells} cells in {total_batches} batches (batch_size={safe_batch_size})"
        )

        for i in tqdm(
            range(0, n_cells, safe_batch_size), desc="Computing similarity edges"
        ):
            end_i = min(i + safe_batch_size, n_cells)
            batch_size_actual = end_i - i
            batch_positions = positions[i:end_i]
            batch_features = features[i:end_i]

            # Compute pairwise distances for this batch against all cells
            # Shape: [batch_size, n_cells]
            diff = batch_positions.unsqueeze(1) - positions.unsqueeze(0)
            distances = cast(torch.Tensor, torch.norm(diff, dim=2))  # type: ignore

            # Compute feature similarity for this batch against all cells
            # Shape: [batch_size, n_cells]
            feature_sim = self._compute_feature_similarity(batch_features, features)

            # Compute distance similarity using specified metric
            distance_sim = self._compute_distance_similarity(distances)

            # Compute edge weights using specified combination method
            feature_sim_positive = torch.clamp(feature_sim, min=0.0)

            if self.combination_method == "additive":
                # Additive combination:
                # w_ij = alpha * feature_sim + (1 - alpha) * distance_sim
                edge_weights = (
                    self.alpha * feature_sim_positive + (1 - self.alpha) * distance_sim
                )
            elif self.combination_method == "multiplicative":
                # Multiplicative combination:
                # w_ij = feature_sim^alpha * distance_sim^(1-alpha)
                # Add small epsilon to avoid log(0)
                epsilon = 1e-8
                edge_weights = torch.pow(
                    feature_sim_positive + epsilon, self.alpha
                ) * torch.pow(distance_sim + epsilon, 1 - self.alpha)
            else:
                raise ValueError(
                    f"Unknown combination method: {self.combination_method}"
                )

            # Mask out self-loops (set diagonal to -inf for this batch's rows)
            batch_indices = torch.arange(i, end_i, device=compute_device)
            edge_weights[
                torch.arange(batch_size_actual, device=compute_device), batch_indices
            ] = -float("inf")

            # ========== VECTORIZED NEIGHBOR SELECTION ==========
            if self.use_knn_mode:
                # KNN mode: select top-k edges by weight for all nodes in batch at once
                actual_k = min(self.k, n_cells - 1)
                if actual_k > 0:
                    # torch.topk along dim=1 gives top-k for each row
                    # Shape: values [batch_size, k], indices [batch_size, k]
                    _, top_k_indices = torch.topk(
                        edge_weights, k=actual_k, dim=1, largest=True
                    )

                    # Create source indices: repeat each batch index k times
                    # Shape: [batch_size * k]
                    batch_src = (
                        batch_indices.unsqueeze(1).repeat(1, actual_k).reshape(-1)
                    )
                    batch_dst = top_k_indices.reshape(-1)

                    batch_edge_indices = torch.stack([batch_src, batch_dst], dim=0)
                else:
                    batch_edge_indices = torch.empty(
                        (2, 0), dtype=torch.long, device=compute_device
                    )
            else:
                # Threshold mode: filter by similarity threshold using boolean mask
                # Shape: mask [batch_size, n_cells]
                valid_mask = edge_weights >= self.similarity_threshold

                # Get row and column indices of True values
                # Shape: rows [num_edges], cols [num_edges]
                rows, cols = torch.nonzero(valid_mask, as_tuple=True)

                # Convert local row indices to global source indices
                batch_src = batch_indices[rows]
                batch_dst = cols

                batch_edge_indices = torch.stack([batch_src, batch_dst], dim=0)

            # Calculate edge features for selected edges
            if batch_edge_indices.shape[1] > 0:
                # Get distances for selected edges
                edge_distances = distances[
                    batch_edge_indices[0] - i, batch_edge_indices[1]
                ].unsqueeze(1)

                # Calculate direction vectors
                direction_vectors = (
                    positions[batch_edge_indices[1]] - positions[batch_edge_indices[0]]
                )
                direction_norms = cast(
                    torch.Tensor,
                    torch.norm(direction_vectors, dim=1, keepdim=True) + 1e-8,  # type: ignore
                )
                direction_unit = direction_vectors / direction_norms

                batch_edge_features = torch.cat([edge_distances, direction_unit], dim=1)

                all_edge_indices.append(batch_edge_indices)
                all_edge_features.append(batch_edge_features)

            # Clear GPU cache if on GPU
            if compute_device.startswith("cuda"):
                torch.cuda.empty_cache()

        # Concatenate all edges
        if all_edge_indices:
            edge_indices = torch.cat(all_edge_indices, dim=1)
            edge_features = torch.cat(all_edge_features, dim=0)

            # Remove duplicate edges (keep edge with higher weight)
            # Create unique edge identifier (ensure i < j for undirected edges)
            src, dst = edge_indices[0], edge_indices[1]
            edge_ids = torch.where(src < dst, src * n_cells + dst, dst * n_cells + src)

            # Sort by edge_id to group duplicates together
            sorted_indices = torch.argsort(edge_ids)
            edge_ids_sorted = edge_ids[sorted_indices]

            # Find first occurrence of each unique edge
            unique_mask = torch.cat(
                [
                    torch.tensor([True], device=compute_device),
                    edge_ids_sorted[1:] != edge_ids_sorted[:-1],
                ]
            )

            # Get indices of unique edges in original order
            unique_indices = sorted_indices[unique_mask]

            # Keep only unique edges
            edge_indices = edge_indices[:, unique_indices]
            edge_features = edge_features[unique_indices]

            # Move back to original device if needed
            if compute_device != self.device:
                edge_indices = edge_indices.to(self.device)
                edge_features = edge_features.to(self.device)

            if self.use_knn_mode:
                logger.info(
                    f"Created {edge_indices.shape[1]} unique similarity-based edges "
                    f"using KNN mode (k={self.k}, distance_sigma={self.distance_sigma}, alpha={self.alpha}, "
                    f"method={self.combination_method}, distance={self.distance_metric}, feature={self.feature_metric})"
                )
            else:
                logger.info(
                    f"Created {edge_indices.shape[1]} unique similarity-based edges "
                    f"using threshold mode (threshold={self.similarity_threshold}, distance_sigma={self.distance_sigma}, alpha={self.alpha}, "
                    f"method={self.combination_method}, distance={self.distance_metric}, feature={self.feature_metric})"
                )
        else:
            logger.info("No edges meet the selection criteria")
            edge_indices = torch.empty((2, 0), dtype=torch.long, device=self.device)
            edge_features = torch.empty((0, 3), dtype=torch.float32, device=self.device)

        return edge_indices, edge_features

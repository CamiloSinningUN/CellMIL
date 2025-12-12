import torch
import networkx as nx
import numpy as np
import cv2
from scipy.spatial import cKDTree, ConvexHull  # type: ignore
from typing import Any, cast, Mapping
from cellmil.interfaces.FeatureExtractorConfig import ExtractorType
from cellmil.utils import logger


class TopologicalExtractor:
    def __init__(self, extractor_name: ExtractorType):
        self.extractor_name = extractor_name

        if self.extractor_name == ExtractorType.connectivity:
            self.extractor = ConnectivityExtractor()
        elif self.extractor_name == ExtractorType.structure:
            self.extractor = StructureExtractor()
        elif self.extractor_name == ExtractorType.geometric:
            self.extractor = GeometricExtractor()
        else:
            raise ValueError(f"Unknown extractor type: {self.extractor_name}")

    def extract_features(
        self,
        cell_id: torch.Tensor,
        graph: dict[str, torch.Tensor],
        cells: list[dict[str, Any]],
    ) -> dict[str, Any]:
        try:
            features = self.extractor.extract_features(cell_id, graph, cells)
            return features
        except Exception as e:
            raise RuntimeError(f"Error extracting features: {e}")


class ConnectivityExtractor:
    """Extract connectivity features from the graph.
    * Degree
    * Weighted degree (by distance)
    * K-core number
    * PageRank
    * Eigenvector centrality (Approx.)
    """

    def __init__(self):
        """Initialize the connectivity extractor."""
        # Cache for computed global metrics
        self._cached_k_core: dict[int, float] | None = None
        self._cached_pagerank: dict[int, float] | None = None
        self._cached_eigenvector: dict[int, float] | None = None
        # Cache for degree metrics
        self._cached_degree: dict[int, float] | None = None
        self._cached_weighted_degree: dict[int, float] | None = None

    def _ensure_global_metrics_computed(
        self, edge_indices: torch.Tensor, edge_features: torch.Tensor, num_nodes: int
    ) -> tuple[
        dict[int, float],
        dict[int, float],
        dict[int, float],
        dict[int, float],
        dict[int, float],
    ]:
        """Compute global metrics (k-core, pagerank, eigenvector, degree, weighted_degree) if not cached."""
        if (
            self._cached_k_core is None
            or self._cached_pagerank is None
            or self._cached_eigenvector is None
            or self._cached_degree is None
            or self._cached_weighted_degree is None
        ):
            # Build NetworkX graph once
            G = nx.Graph()
            G.add_nodes_from(range(num_nodes))  # type: ignore

            edges = cast(np.ndarray[Any, Any], edge_indices.t().cpu().numpy())  # type: ignore

            if edge_features.shape[0] > 0:
                # Weighted graph
                distances = cast(
                    np.ndarray[Any, Any],
                    edge_features[:, 0].cpu().numpy(),  # type: ignore
                )
                weights = 1.0 / (distances + 1e-8)
                weighted_edges = [
                    (int(edges[i, 0]), int(edges[i, 1]), float(weights[i]))
                    for i in range(len(edges))
                ]
                G.add_weighted_edges_from(weighted_edges)  # type: ignore
            else:
                # Unweighted graph
                G.add_edges_from(edges)  # type: ignore

            # Compute all global metrics at once
            try:
                self._cached_k_core = dict(cast(Mapping[int, float], nx.core_number(G)))  # type: ignore
            except Exception:
                self._cached_k_core = {i: 0.0 for i in range(num_nodes)}

            try:
                if edge_features.shape[0] > 0:
                    self._cached_pagerank = dict(
                        cast(
                            Mapping[int, float],
                            nx.pagerank(  # type: ignore
                                G, alpha=0.85, max_iter=100, tol=1e-6, weight="weight"
                            ),
                        )
                    )  # type: ignore
                else:
                    self._cached_pagerank = dict(
                        cast(
                            Mapping[int, float],
                            nx.pagerank(G, alpha=0.85, max_iter=100, tol=1e-6),  # type: ignore
                        )
                    )
            except Exception:
                default_pr = 1.0 / num_nodes if num_nodes > 0 else 0.0
                self._cached_pagerank = {i: default_pr for i in range(num_nodes)}

            try:
                if edge_features.shape[0] > 0:
                    self._cached_eigenvector = dict(
                        cast(
                            Mapping[int, float],
                            nx.eigenvector_centrality(  # type: ignore
                                G, max_iter=100, tol=1e-6, weight="weight"
                            ),
                        )
                    )
                else:
                    self._cached_eigenvector = dict(
                        cast(
                            Mapping[int, float],
                            nx.eigenvector_centrality(G, max_iter=100, tol=1e-6),  # type: ignore
                        )
                    )
            except Exception:
                # Fallback to degree centrality if eigenvector fails
                try:
                    self._cached_eigenvector = dict(
                        cast(Mapping[int, float], nx.degree_centrality(G))  # type: ignore
                    )
                except Exception:
                    self._cached_eigenvector = {i: 0.0 for i in range(num_nodes)}

            # Compute degree metrics efficiently
            self._cached_degree = {}
            self._cached_weighted_degree = {}

            # Use NetworkX degree method for all nodes at once (much faster)
            degree_dict = cast(dict[int, int], dict(G.degree()))  # type: ignore
            for node in range(num_nodes):
                self._cached_degree[node] = float(degree_dict.get(node, 0))

            # Compute weighted degree for all nodes at once
            if edge_features.shape[0] > 0:
                weighted_degree_dict = cast(
                    dict[int, float], dict(G.degree(weight="weight"))  # type: ignore
                )  # type: ignore
                for node in range(num_nodes):
                    self._cached_weighted_degree[node] = float(
                        weighted_degree_dict.get(node, 0.0)
                    )
            else:
                # If no edge features, weighted degree equals regular degree
                self._cached_weighted_degree = self._cached_degree.copy()

        return (
            self._cached_k_core,
            self._cached_pagerank,
            self._cached_eigenvector,
            self._cached_degree,
            self._cached_weighted_degree,
        )

    def clear_cache(self) -> None:
        """Clear the cached global metrics. Useful for testing or memory management."""
        self._cached_k_core = None
        self._cached_pagerank = None
        self._cached_eigenvector = None
        self._cached_degree = None
        self._cached_weighted_degree = None

    def extract_features(
        self,
        cell_id: torch.Tensor,
        graph: dict[str, torch.Tensor],
        cells: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Extract connectivity features for a specific cell from the graph.

        Args:
            cell_id: Tensor containing the target cell ID
            graph: Dictionary containing 'edge_indices', 'edge_features', 'node_features'
            cells: List of cell dictionaries (not used in connectivity extraction)

        Returns:
            Dictionary containing connectivity features
        """
        try:
            edge_indices = graph["edge_indices"]  # [2, num_edges]
            edge_features = graph[
                "edge_features"
            ]  # [num_edges, 3] - [distance, direction_x, direction_y]
            node_features = graph["node_features"]  # [num_nodes, 1] - contains cell_ids

            # Extract the cell_id from the tensor
            # cell_id is the node feature tensor which contains the cell_id value in its first element
            target_cell_id = int(
                cell_id.item() if cell_id.numel() == 1 else cell_id[0].item()
            )

            # Find the node index that corresponds to this cell_id
            # node_features[:, 0] contains the actual cell_ids
            node_index = self._find_index(target_cell_id, node_features)
            if node_index is None:
                # Cell ID not found in the graph
                return {
                    "degree": 0.0,
                    "weighted_degree": 0.0,
                    "k_core": 0.0,
                    "pagerank": 0.0,
                    "eigenvector_centrality": 0.0,
                }

            # Get total number of nodes
            num_nodes = node_features.shape[0]

            # Extract features using the node index
            features: dict[str, Any] = {}

            # Get cached global metrics (computed once for all nodes)
            (
                k_core_scores,
                pagerank_scores,
                eigenvector_scores,
                degree_scores,
                weighted_degree_scores,
            ) = self._ensure_global_metrics_computed(
                edge_indices, edge_features, num_nodes
            )

            # 1. Degree (number of connections) - from cache
            features["degree"] = degree_scores.get(node_index, 0.0)

            # 2. Weighted degree (sum of inverse distances) - from cache
            features["weighted_degree"] = weighted_degree_scores.get(node_index, 0.0)

            features["k_core"] = k_core_scores.get(node_index, 0.0)
            features["pagerank"] = pagerank_scores.get(
                node_index, 1.0 / num_nodes if num_nodes > 0 else 0.0
            )
            features["eigenvector_centrality"] = eigenvector_scores.get(node_index, 0.0)

            return features

        except Exception:
            # Return zero features if computation fails
            return {
                "degree": 0.0,
                "weighted_degree": 0.0,
                "k_core": 0.0,
                "pagerank": 0.0,
                "eigenvector_centrality": 0.0,
            }

    def _find_index(self, cell_id: int, node_features: torch.Tensor) -> int | None:
        """Find the node index that corresponds to the given cell_id.

        Args:
            cell_id: The cell ID to find
            node_features: Tensor with shape [num_nodes, 1] containing cell_ids in first column

        Returns:
            Node index if found, None otherwise
        """
        # node_features[:, 0] contains the cell_ids
        cell_ids = node_features[:, 0]
        matches = (cell_ids == cell_id).nonzero(as_tuple=True)[0]

        if len(matches) > 0:
            return int(matches[0].item())
        return None


class StructureExtractor:
    """Extract structural features from the graph.
    * Weighted clustering coefficient (by distance)
    * Local efficiency
    * Ego-network density
    """

    def __init__(self):
        """Initialize the structure extractor."""
        # Cache for computed global metrics
        self._cached_weighted_clustering: dict[int, float] | None = None

    def _ensure_global_metrics_computed(
        self, edge_indices: torch.Tensor, edge_features: torch.Tensor, num_nodes: int
    ) -> dict[int, float]:
        """Compute global structural metrics if not cached."""
        if self._cached_weighted_clustering is None:
            # Build NetworkX graph
            G = nx.Graph()
            G.add_nodes_from(range(num_nodes))  # type: ignore
            edges = cast(np.ndarray[Any, Any], edge_indices.t().cpu().numpy())  # type: ignore

            if edge_features.shape[0] > 0:
                # Weighted graph
                distances = cast(
                    np.ndarray[Any, Any],
                    edge_features[:, 0].cpu().numpy(),  # type: ignore
                )
                weights = 1.0 / (distances + 1e-8)
                weighted_edges = [
                    (int(edges[i, 0]), int(edges[i, 1]), float(weights[i]))
                    for i in range(len(edges))
                ]
                G.add_weighted_edges_from(weighted_edges)  # type: ignore
            else:
                G.add_edges_from(edges)  # type: ignore

            try:
                if edge_features.shape[0] > 0:
                    self._cached_weighted_clustering = dict(
                        cast(Mapping[int, float], nx.clustering(G, weight="weight"))  # type: ignore
                    )
                else:
                    self._cached_weighted_clustering = dict(
                        cast(Mapping[int, float], nx.clustering(G))  # type: ignore
                    )
            except Exception:
                self._cached_weighted_clustering = {i: 0.0 for i in range(num_nodes)}

        return self._cached_weighted_clustering

    def extract_features(
        self,
        cell_id: torch.Tensor,
        graph: dict[str, torch.Tensor],
        cells: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Extract structural features for a specific cell from the graph."""
        try:
            edge_indices = graph["edge_indices"]
            edge_features = graph["edge_features"]
            node_features = graph["node_features"]

            # Extract the cell_id from the tensor
            target_cell_id = int(
                cell_id.item() if cell_id.numel() == 1 else cell_id[0].item()
            )

            # Find the node index
            node_index = self._find_index(target_cell_id, node_features)
            if node_index is None:
                return {
                    "weighted_clustering_coefficient": 0.0,
                    "local_efficiency": 0.0,
                    "ego_network_density": 0.0,
                }

            num_nodes = node_features.shape[0]
            features: dict[str, Any] = {}

            # Vectorized neighbor finding (compute once, use multiple times)
            neighbors = self._get_neighbors_vectorized(node_index, edge_indices)

            # Get cached global metrics
            weighted_clustering_scores = self._ensure_global_metrics_computed(
                edge_indices, edge_features, num_nodes
            )

            # 1. Weighted clustering coefficient
            features["weighted_clustering_coefficient"] = (
                weighted_clustering_scores.get(node_index, 0.0)
            )

            # 2. Local efficiency (using pre-computed neighbors)
            features["local_efficiency"] = self._calculate_local_efficiency(
                node_index, edge_indices, edge_features, neighbors
            )

            # 3. Ego-network density (using pre-computed neighbors)
            features["ego_network_density"] = self._calculate_ego_network_density(
                node_index, edge_indices, neighbors
            )

            return features

        except Exception:
            return {
                "weighted_clustering_coefficient": 0.0,
                "local_efficiency": 0.0,
                "ego_network_density": 0.0,
            }

    def _find_index(self, cell_id: int, node_features: torch.Tensor) -> int | None:
        """Find the node index that corresponds to the given cell_id."""
        cell_ids = node_features[:, 0]
        matches = (cell_ids == cell_id).nonzero(as_tuple=True)[0]
        if len(matches) > 0:
            return int(matches[0].item())
        return None

    def _get_neighbors_vectorized(
        self, node_idx: int, edge_indices: torch.Tensor
    ) -> torch.Tensor:
        """Vectorized neighbor finding using torch operations."""
        # Find edges where node_idx appears as source or target
        mask_src = edge_indices[0] == node_idx
        mask_tgt = edge_indices[1] == node_idx

        # Get neighbors from both directions
        neighbors_from_src = edge_indices[1][mask_src]  # When node_idx is source
        neighbors_from_tgt = edge_indices[0][mask_tgt]  # When node_idx is target

        # Combine and get unique neighbors
        all_neighbors = torch.cat([neighbors_from_src, neighbors_from_tgt])

        # Get unique values manually to avoid type issues
        if len(all_neighbors) == 0:
            return torch.tensor([], dtype=torch.long)

        neighbors_array = cast(np.ndarray[Any, Any], all_neighbors.cpu().numpy())  # type: ignore
        unique_neighbors_list = list(set(neighbors_array.astype(int)))
        unique_neighbors = torch.tensor(
            unique_neighbors_list, dtype=torch.long, device=all_neighbors.device
        )

        return unique_neighbors

    def _calculate_local_efficiency(
        self,
        node_idx: int,
        edge_indices: torch.Tensor,
        edge_features: torch.Tensor,
        neighbors: torch.Tensor,
    ) -> float:
        """Calculate local efficiency using pre-computed neighbors."""
        if len(neighbors) < 2:
            return 0.0

        # Convert to set for fast lookup
        neighbors_array = cast(np.ndarray[Any, Any], neighbors.cpu().numpy())  # type: ignore
        neighbors_set = set(neighbors_array.astype(int))

        # Build subgraph of neighbors
        G = nx.Graph()
        G.add_nodes_from(neighbors_set)  # type: ignore

        # Vectorized edge filtering between neighbors
        edges = edge_indices.t()  # [num_edges, 2]

        # Create masks for edges between neighbors
        src_in_neighbors = torch.isin(edges[:, 0], neighbors)
        tgt_in_neighbors = torch.isin(edges[:, 1], neighbors)
        neighbor_edge_mask = src_in_neighbors & tgt_in_neighbors

        neighbor_edges_indices = edges[neighbor_edge_mask]

        if edge_features.shape[0] > 0 and neighbor_edge_mask.sum() > 0:
            # Weighted graph - get corresponding distances
            neighbor_distances = edge_features[neighbor_edge_mask, 0]
            weights = 1.0 / (neighbor_distances + 1e-8)

            weighted_edges = [
                (int(edge[0].item()), int(edge[1].item()), float(weight.item()))
                for edge, weight in zip(neighbor_edges_indices, weights)
            ]
            G.add_weighted_edges_from(weighted_edges)  # type: ignore
        else:
            # Unweighted graph
            unweighted_edges = [
                (int(edge[0].item()), int(edge[1].item()))
                for edge in neighbor_edges_indices
            ]
            G.add_edges_from(unweighted_edges)  # type: ignore

        # Calculate efficiency
        try:
            return float(nx.local_efficiency(G))  # type: ignore
        except Exception as e:
            logger.error(f"Error in local_efficiency calculation: {e}", exc_info=True)
            return 0.0

    def _calculate_ego_network_density(
        self, node_idx: int, edge_indices: torch.Tensor, neighbors: torch.Tensor
    ) -> float:
        """Calculate ego-network density using pre-computed neighbors."""
        # Ego network = node + its neighbors
        ego_nodes = torch.cat(
            [torch.tensor([node_idx], device=neighbors.device), neighbors]
        )
        ego_size = len(ego_nodes)

        if ego_size < 2:
            return 0.0

        # Vectorized edge counting within ego network
        edges = edge_indices.t()  # [num_edges, 2]

        # Create masks for edges within ego network
        src_in_ego = torch.isin(edges[:, 0], ego_nodes)
        tgt_in_ego = torch.isin(edges[:, 1], ego_nodes)
        ego_edge_mask = src_in_ego & tgt_in_ego

        # Count unique edges (undirected graph has each edge twice)
        edges_in_ego = ego_edge_mask.sum().item() // 2

        # Maximum possible edges in ego network
        max_edges = ego_size * (ego_size - 1) // 2

        return float(edges_in_ego / max_edges) if max_edges > 0 else 0.0

    def clear_cache(self) -> None:
        """Clear the cached global metrics."""
        self._cached_weighted_clustering = None


class GeometricExtractor:
    """Extract geometric features from the graph.
    * Distance to nearest neighbor
    * Distance to nearest neighbor of each type
    * Mean distance to neighbors
    * Edge length variance
    * Anisotropy → Dominant direction of nearest neighbors
    * Local density (number of nodes in a radius)
    * Spatial entropy of neighbors
    * Shape of local convex hull
    * Area/perimeter ratio of local neighborhood
    * Nucleus size relative to local density
    * Anisotropy of neighborhood
    * Relative orientation of neighbors
    """

    def __init__(self):
        """Initialize the geometric extractor."""
        self.radius_for_density = 200.0  # Default radius for local density calculation
        # Cache mapping from cell_id to cell dict to avoid rebuilding per call
        self._cell_id_to_cell_cache: dict[int, dict[str, Any]] | None = None
        self._cells_obj_id: int | None = None  # track identity of current cells list for caches
        # Spatial index caches for local density
        self._positions_array: np.ndarray[Any, Any] | None = None
        self._cell_ids_array: np.ndarray[Any, Any] | None = None
        self._cell_id_to_pos_index: dict[int, int] | None = None
        self._kdtree: cKDTree[np.ndarray[Any, Any]] | None = None
        # Cache for local density queries: key = (cells_obj_id, int(radius*1000), cell_id)
        self._local_density_cache: dict[tuple[int, int, int], float] = {}
        # Graph-scoped caches (reset when tensors change identity)
        self._neighbor_cache_key: tuple[int, int, int] | None = None
        self._neighbor_cache: dict[int, dict[str, Any]] = {}
        # Node index cache keyed by node_features identity
        self._node_index_cache_key: int | None = None
        self._node_index_cache: dict[int, int] = {}

    def _get_cell_mapping(
        self, cells: list[dict[str, Any]]
    ) -> dict[int, dict[str, Any]]:
        """Return cached mapping cell_id -> cell; rebuild only when cells list identity changes."""
        cells_id = id(cells)
        if self._cell_id_to_cell_cache is None or self._cells_obj_id != cells_id:
            self._cell_id_to_cell_cache = {
                int(cell["cell_id"]): cell for cell in cells if "cell_id" in cell
            }
            self._cells_obj_id = cells_id
        return self._cell_id_to_cell_cache

    def _ensure_spatial_index(self, cells: list[dict[str, Any]]) -> None:
        """Build and cache KDTree and arrays from cells when the list identity changes."""
        if (
            self._kdtree is not None
            and self._positions_array is not None
            and self._cells_obj_id == id(cells)
        ):
            return

        positions: list[list[float]] = []
        cell_ids: list[int] = []
        for cell in cells:
            if "centroid" in cell and cell["centroid"] is not None:
                cx, cy = cell["centroid"][0], cell["centroid"][1]
                positions.append([float(cx), float(cy)])
                cell_ids.append(int(cell.get("cell_id", -1)))

        if not positions:
            # Clear caches if no positions
            self._positions_array = None
            self._cell_ids_array = None
            self._cell_id_to_pos_index = None
            self._kdtree = None
            self._cells_obj_id = id(cells)
            self._local_density_cache.clear()
            return

        self._positions_array = np.asarray(positions, dtype=np.float64)
        self._cell_ids_array = np.asarray(cell_ids, dtype=np.int64)
        # Build mapping from cell_id to the last seen index (stable enough for our use)
        self._cell_id_to_pos_index = {
            int(cid): idx for idx, cid in enumerate(self._cell_ids_array.tolist())
        }
        # Build KDTree once
        self._kdtree = cKDTree(self._positions_array) # type: ignore
        self._cells_obj_id = id(cells)
        self._local_density_cache.clear()

    def _graph_key(
        self,
        edge_indices: torch.Tensor,
        edge_features: torch.Tensor,
        node_features: torch.Tensor,
    ) -> tuple[int, int, int]:
        """Create a lightweight identity key for current graph tensors (no hashing)."""
        return (id(edge_indices), id(edge_features), id(node_features))

    def extract_features(
        self,
        cell_id: torch.Tensor,
        graph: dict[str, torch.Tensor],
        cells: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Extract geometric features for a specific cell from the graph."""
        try:
            edge_indices = graph["edge_indices"]
            edge_features = graph[
                "edge_features"
            ]  # [distance, direction_x, direction_y]
            node_features = graph["node_features"]

            # Extract the cell_id from the tensor
            target_cell_id = int(
                cell_id.item() if cell_id.numel() == 1 else cell_id[0].item()
            )

            # Find the node index
            node_index = self._find_index(target_cell_id, node_features)
            if node_index is None:
                raise ValueError(
                    f"Cell ID {target_cell_id} not found in node features."
                )

            # Get cell information for this node using cached mapping
            cell_id_to_cell = self._get_cell_mapping(cells)
            target_cell = cell_id_to_cell.get(target_cell_id)

            if target_cell is None:
                raise ValueError(f"Cell ID {target_cell_id} not found in cells list.")

            features: dict[str, Any] = {}

            # Get neighbors and their properties
            neighbor_data = self._get_neighbour_data(
                node_index, edge_indices, edge_features, cells, node_features
            )

            if not neighbor_data["neighbors"]:
                raise ValueError(f"Cell ID {target_cell_id} has no neighbors.")

            # 1. Distance to nearest neighbor (using centroid distances)
            features["distance_to_nearest_neighbor"] = (
                min(neighbor_data["distances"]) if neighbor_data["distances"] else 0.0
            )

            # 2. Mean distance to neighbors (using centroid distances)
            features["mean_distance_to_neighbors"] = (
                float(np.mean(neighbor_data["distances"]))  # type: ignore
                if neighbor_data["distances"]
                else 0.0
            )

            # 3. Edge length variance (using centroid distances)
            features["edge_length_variance"] = (
                float(np.var(neighbor_data["distances"]))
                if neighbor_data["distances"]
                else 0.0
            )

            # 4. Anisotropy - dominant direction of nearest neighbors
            features.update(self._calculate_anisotropy(neighbor_data))

            # 5. Local density (number of nodes in a radius)
            features["local_density"] = self._calculate_local_density(
                target_cell, cells, self.radius_for_density
            )

            # 6. Spatial entropy of neighbors
            features["spatial_entropy"] = self._calculate_spatial_entropy(neighbor_data)

            # 7. Convex hull features
            features.update(
                self._calculate_convex_hull_features(neighbor_data, target_cell)
            )

            # 8. Nucleus size relative to local density
            features["nucleus_size_relative_to_density"] = (
                self._calculate_relative_nucleus_size(
                    target_cell, features["local_density"]
                )
            )

            # 9. Relative orientation of neighbors
            features["mean_neighbor_orientation"] = self._calculate_mean_orientation(
                neighbor_data
            )

            return features

        except Exception:
            logger.error("Error extracting features", exc_info=True)
            return {
                "distance_to_nearest_neighbor": 0.0,
                "mean_distance_to_neighbors": 0.0,
                "edge_length_variance": 0.0,
                "anisotropy_ratio": 0.0,
                "dominant_direction_x": 0.0,
                "dominant_direction_y": 0.0,
                "local_density": 0.0,
                "spatial_entropy": 0.0,
                "convex_hull_area": 0.0,
                "convex_hull_perimeter": 0.0,
                "area_perimeter_ratio": 0.0,
                "nucleus_size_relative_to_density": 0.0,
                "mean_neighbor_orientation": 0.0,
            }

    def _find_index(self, cell_id: int, node_features: torch.Tensor) -> int | None:
        """Find the node index that corresponds to the given cell_id."""
        # Rebuild cache if the node_features tensor identity changed
        nf_id = id(node_features)
        if self._node_index_cache_key != nf_id:
            try:
                ids_tensor = node_features[:, 0]
                ids_list = cast(list[int], ids_tensor.detach().cpu().tolist())  # type: ignore[arg-type]
                self._node_index_cache = {}
                for idx, cid in enumerate(ids_list):
                    if cid not in self._node_index_cache:
                        self._node_index_cache[int(cid)] = int(idx)
                self._node_index_cache_key = nf_id
            except Exception:
                cell_ids = node_features[:, 0]
                matches = (cell_ids == cell_id).nonzero(as_tuple=True)[0]
                if len(matches) > 0:
                    return int(matches[0].item())
                return None

        return self._node_index_cache.get(int(cell_id))

    def _get_neighbour_data(
        self,
        node_idx: int,
        edge_indices: torch.Tensor,
        edge_features: torch.Tensor,
        cells: list[dict[str, Any]],
        node_features: torch.Tensor,
    ) -> dict[str, Any]:
        """Get comprehensive neighbor data using vectorized ops (undirected graphs) with simple caching."""
        # Ensure cache corresponds to current graph tensors
        gkey = self._graph_key(edge_indices, edge_features, node_features)
        if self._neighbor_cache_key != gkey:
            self._neighbor_cache_key = gkey
            self._neighbor_cache.clear()

        # Fast path: return cached result
        if node_idx in self._neighbor_cache:
            return self._neighbor_cache[node_idx]
        # Connections mask (node as source or target)
        mask_connected = (edge_indices[0] == node_idx) | (edge_indices[1] == node_idx)

        # Early return if no connections
        if not bool(mask_connected.any().item()):
            result: dict[str, Any] = {
                "neighbors": [],
                "distances": [],
                "directions_x": [],
                "directions_y": [],
                "neighbor_cells": [],
            }
            self._neighbor_cache[node_idx] = result
            return result

        # Indices of all connected edges (preserve order)
        connected_edges = mask_connected.nonzero(as_tuple=True)[0]

        # Source/target tensors for connected edges
        src_nodes = edge_indices[0, connected_edges]
        tgt_nodes = edge_indices[1, connected_edges]

        # Neighbor index for each connected edge (preserves original ordering)
        neighbor_idx_tensor = torch.where(src_nodes == node_idx, tgt_nodes, src_nodes)

        # Convert neighbors to Python list[int]
        neighbors: list[int] = [
            int(x) for x in cast(list[int], neighbor_idx_tensor.tolist())  # type: ignore
        ]

        # Distances and directions (if features exist)
        distances: list[float] = []
        directions_x: list[float] = []
        directions_y: list[float] = []

        if edge_features.shape[0] > 0 and len(connected_edges) > 0:
            # Distances for connected edges
            dist_tensor = edge_features[connected_edges, 0]
            distances = [float(x) for x in cast(list[float], dist_tensor.tolist())]  # type: ignore

            # Direction sign: +1 if node is source, -1 if node is target
            # cast bool to numeric on same device/dtype without creating new tensors
            sign = (src_nodes == node_idx).to(dtype=edge_features.dtype)
            sign = sign.mul_(2).sub_(
                1
            )  # 1 for True -> (2*1-1)=1, 0 for False -> (0*2-1)=-1

            # Apply sign to (dx, dy) for each connected edge
            dir_tensor = edge_features[connected_edges, 1:3] * sign.view(-1, 1)
            directions_x = [
                float(x) for x in cast(list[float], dir_tensor[:, 0].tolist()) # type: ignore
            ]
            directions_y = [
                float(y) for y in cast(list[float], dir_tensor[:, 1].tolist()) # type: ignore
            ]

        # Neighbor cell lookup (keep order; include only found cells)
        neighbor_cells: list[dict[str, Any]] = []
        cell_id_to_cell = self._get_cell_mapping(cells)

        if len(neighbors) > 0:
            neighbor_idx_cpu = neighbor_idx_tensor.cpu()
            valid_mask = (neighbor_idx_cpu >= 0) & (
                neighbor_idx_cpu < node_features.shape[0]
            )
            if bool(valid_mask.any().item()):
                valid_indices = neighbor_idx_cpu[valid_mask]
                neighbor_cell_ids = cast(
                    list[int], node_features[valid_indices, 0].cpu().tolist() # type: ignore
                )  
                # Preserve order; skip missing ids without branching per element
                neighbor_cells = [
                    cell_id_to_cell[cid]
                    for cid in neighbor_cell_ids
                    if cid in cell_id_to_cell
                ]

        result: dict[str, Any] = {
            "neighbors": neighbors,
            "distances": distances,
            "directions_x": directions_x,
            "directions_y": directions_y,
            "neighbor_cells": neighbor_cells,
        }
        self._neighbor_cache[node_idx] = result
        return result

    def clear_cache(self) -> None:
        """Clear all caches maintained by GeometricExtractor."""
        self._cell_id_to_cell_cache = None
        self._cells_obj_id = None
        self._positions_array = None
        self._cell_ids_array = None
        self._cell_id_to_pos_index = None
        self._kdtree = None
        self._neighbor_cache_key = None
        self._neighbor_cache.clear()
        self._node_index_cache_key = None
        self._node_index_cache.clear()

    def _calculate_anisotropy(self, neighbor_data: dict[str, Any]) -> dict[str, float]:
        """Calculate anisotropy and dominant direction."""
        directions_x = np.array(neighbor_data["directions_x"])
        directions_y = np.array(neighbor_data["directions_y"])

        if len(directions_x) == 0:
            return {
                "anisotropy_ratio": 0.0,
                "dominant_direction_x": 0.0,
                "dominant_direction_y": 0.0,
            }

        # Calculate covariance matrix of directions
        directions = np.column_stack([directions_x, directions_y])
        if directions.shape[0] < 2:
            return {
                "anisotropy_ratio": 0.0,
                "dominant_direction_x": 0.0,
                "dominant_direction_y": 0.0,
            }

        try:
            cov_matrix = np.cov(directions.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

            # Sort eigenvalues and eigenvectors
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]

            # Anisotropy ratio (ratio of major to minor axis)
            anisotropy_ratio = eigenvalues[0] / (eigenvalues[1] + 1e-8)

            # Dominant direction (first eigenvector)
            dominant_direction = eigenvectors[:, 0]

            return {
                "anisotropy_ratio": float(anisotropy_ratio),
                "dominant_direction_x": float(dominant_direction[0]),
                "dominant_direction_y": float(dominant_direction[1]),
            }
        except Exception:
            return {
                "anisotropy_ratio": 0.0,
                "dominant_direction_x": 0.0,
                "dominant_direction_y": 0.0,
            }

    def _calculate_local_density(
        self, target_cell: dict[str, Any], cells: list[dict[str, Any]], radius: float
    ) -> float:
        """Calculate number of cells within radius using cached KDTree (no Python loops)."""

        # Early return if no centroid
        if "centroid" not in target_cell or target_cell["centroid"] is None:
            return 0.0

        # Ensure spatial index is ready (built once per cells list identity)
        self._ensure_spatial_index(cells)

        if self._kdtree is None or self._positions_array is None:
            return 0.0

        # Query count of neighbors within radius (prefer return_length for speed)
        # Check cache first
        cells_id = self._cells_obj_id if self._cells_obj_id is not None else id(cells)
        target_id = int(target_cell.get("cell_id", -1))
        radius_key = int(round(float(radius) * 1000))
        cache_key = (cells_id, radius_key, target_id)
        cached = self._local_density_cache.get(cache_key)
        if cached is not None:
            return cached
        target_pos = np.asarray(
            [target_cell["centroid"][0], target_cell["centroid"][1]], dtype=np.float64
        )
        try:
            cnt = self._kdtree.query_ball_point(
                target_pos, radius, p=2.0, return_length=True
            )
            count_val = (
                int(cnt)
                if not isinstance(cnt, (list, tuple, np.ndarray))
                else (int(cnt[0]) if len(cnt) > 0 else 0) # type: ignore
            )  
        except TypeError:
            # Older SciPy: no return_length, fall back to indices length
            indices = self._kdtree.query_ball_point(target_pos, radius, p=2.0)
            count_val = (
                len(indices) # type: ignore
                if isinstance(indices, (list, tuple, np.ndarray)) # type: ignore
                else int(indices)
            )  

        # Exclude the target cell itself only if we know it's indexed in the tree
        if (
            self._cell_id_to_pos_index is not None
            and target_id in self._cell_id_to_pos_index
        ):
            count_val = max(0, count_val - 1)
        result = float(count_val)
        self._local_density_cache[cache_key] = result
        return result

    def _calculate_spatial_entropy(self, neighbor_data: dict[str, Any]) -> float:
        """Calculate spatial entropy of neighbor distribution."""
        if not neighbor_data["directions_x"] or not neighbor_data["directions_y"]:
            return 0.0

        # Divide space into angular bins
        angles = np.arctan2(
            neighbor_data["directions_y"], neighbor_data["directions_x"]
        )
        # Normalize to [0, 2π]
        angles = (angles + 2 * np.pi) % (2 * np.pi)

        # Create bins (8 directions)
        n_bins = 8
        bin_edges = np.linspace(0, 2 * np.pi, n_bins + 1)
        hist, _ = np.histogram(angles, bins=bin_edges)

        # Normalize to probabilities
        prob = hist / np.sum(hist)
        prob = prob[prob > 0]  # Remove zero probabilities

        # Calculate entropy
        if len(prob) > 0:
            entropy = -np.sum(prob * np.log2(prob))
            return float(entropy)
        return 0.0

    def _calculate_convex_hull_features(
        self, neighbor_data: dict[str, Any], target_cell: dict[str, Any]
    ) -> dict[str, float]:
        """Calculate convex hull area and perimeter of neighborhood."""
        try:
            # Get positions of neighbors + target cell
            points: list[list[float]] = []

            # Add target cell position
            if "centroid" in target_cell:
                points.append([target_cell["centroid"][0], target_cell["centroid"][1]])

            # Add neighbor positions
            for cell in neighbor_data["neighbor_cells"]:
                if "centroid" in cell:
                    points.append([cell["centroid"][0], cell["centroid"][1]])

            if len(points) < 3:  # Need at least 3 points for convex hull
                return {
                    "convex_hull_area": 0.0,
                    "convex_hull_perimeter": 0.0,
                    "area_perimeter_ratio": 0.0,
                }

            points_array = np.array(points)
            hull = ConvexHull(points_array)

            area = float(hull.volume)  # In 2D, volume is actually area
            perimeter = 0.0

            # Calculate perimeter
            for simplex in hull.simplices:
                p1 = points_array[simplex[0]]
                p2 = points_array[simplex[1]]
                perimeter += np.linalg.norm(p2 - p1)

            area_perimeter_ratio = (
                (4 * np.pi * area) / (perimeter**2) if perimeter > 0 else 0.0
            )

            return {
                "convex_hull_area": area,
                "convex_hull_perimeter": float(perimeter),
                "area_perimeter_ratio": float(area_perimeter_ratio),
            }

        except Exception:
            return {
                "convex_hull_area": 0.0,
                "convex_hull_perimeter": 0.0,
                "area_perimeter_ratio": 0.0,
            }

    def _calculate_relative_nucleus_size(
        self, target_cell: dict[str, Any], local_density: float
    ) -> float:
        """Calculate nucleus size relative to local density."""
        # Get area from contour if available, otherwise fall back to area or size fields
        nucleus_size = target_cell.get("area", 0.0)

        # Use contour to calculate area if available and area not already set
        if nucleus_size == 0.0 and "contour" in target_cell:
            contour = target_cell["contour"]
            if contour is not None and len(contour) > 0:
                try:
                    # Convert contour points to the format expected by cv2.contourArea
                    # It should be an array of points with shape (n, 1, 2) where n is number of points
                    if isinstance(contour, list):
                        contour_array = np.array(contour, dtype=np.int32)
                        # Reshape if needed
                        if (
                            len(contour_array.shape) == 2
                            and contour_array.shape[1] == 2
                        ):
                            # Already in format [[x1, y1], [x2, y2], ...] - convert to required format
                            contour_array = contour_array.reshape((-1, 1, 2))
                        nucleus_size = float(cv2.contourArea(contour_array))
                    else:
                        # Assuming contour is already a numpy array
                        nucleus_size = float(cv2.contourArea(contour))
                except Exception:
                    # Fall back to area or size if contour processing fails
                    nucleus_size = 0.0

        # Fall back to size if contour and area are not available
        if nucleus_size == 0.0:
            nucleus_size = target_cell.get("size", 1.0)

        if local_density > 0:
            return float(nucleus_size / local_density)
        return float(nucleus_size)

    def _calculate_mean_orientation(self, neighbor_data: dict[str, Any]) -> float:
        """Calculate mean orientation of neighbors."""
        if not neighbor_data["directions_x"] or not neighbor_data["directions_y"]:
            return 0.0

        directions_x_array = np.array(neighbor_data["directions_x"])
        directions_y_array = np.array(neighbor_data["directions_y"])

        # Directly average the vectors
        mean_x = np.mean(directions_x_array)
        mean_y = np.mean(directions_y_array)

        # Convert average vector back to an angle
        mean_angle = np.arctan2(mean_y, mean_x)

        return float(mean_angle)

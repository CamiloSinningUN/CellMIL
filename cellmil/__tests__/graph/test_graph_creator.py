import pytest
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, cast
from matplotlib.collections import LineCollection
from unittest.mock import patch
from cellmil.graph.creator import (
    Creator,
    KNNEdgeCreator,
    RadiusEdgeCreator,
    DelaunayEdgeCreator,
    DilateEdgeCreator,
)
from cellmil.interfaces.GraphCreatorConfig import GraphCreatorType


class TestGraphCreator:
    @pytest.fixture
    def device(self):
        """Test device - use CPU for reproducibility"""
        return "cpu"

    @pytest.fixture
    def sample_cells_10(self):
        """Create 10 sample cells with realistic data"""
        np.random.seed(42)
        cells: list[dict[str, Any]] = []

        # Create 10 cells in a roughly circular pattern for interesting graph structures
        angles = np.linspace(0, 2 * np.pi, 10, endpoint=False)
        radius = 200
        center = np.array([500, 500])

        for i, angle in enumerate(angles):
            # Add some noise to make it more realistic
            noise = np.random.normal(0, 30, 2)
            position = (
                center + radius * np.array([np.cos(angle), np.sin(angle)]) + noise
            )

            # Create a simple square contour around each cell position
            size = 20 + np.random.randint(-5, 6)  # type: ignore
            contour = [
                [int(position[0] - size), int(position[1] - size)],
                [int(position[0] + size), int(position[1] - size)],
                [int(position[0] + size), int(position[1] + size)],
                [int(position[0] - size), int(position[1] + size)],
            ]

            cell: dict[str, Any] = {
                "cell_id": i + 1,
                "centroid": [float(position[0]), float(position[1])],
                "contour": contour,
            }
            cells.append(cell)

        return cells

    @pytest.fixture
    def sample_cells_small(self) -> list[dict[str, Any]]:
        """Create 3 sample cells for edge case testing"""
        return [
            {
                "cell_id": 1,
                "centroid": [10.0, 10.0],
                "contour": [[5, 5], [15, 5], [15, 15], [5, 15]],
            },
            {
                "cell_id": 2,
                "centroid": [50.0, 50.0],
                "contour": [[45, 45], [55, 45], [55, 55], [45, 55]],
            },
            {
                "cell_id": 3,
                "centroid": [100.0, 100.0],
                "contour": [[95, 95], [105, 95], [105, 105], [95, 105]],
            },
        ]

    def _create_plot_html(
        self,
        cells: list[dict[str, Any]],
        edge_indices: torch.Tensor,
        positions: torch.Tensor,
        title: str,
        method_name: str,
    ):
        """Create an HTML plot for the graph visualization"""
        _, ax = plt.subplots(1, 1, figsize=(10, 8))  # type: ignore

        # Plot cells as points
        x_coords = [cell["centroid"][0] for cell in cells]
        y_coords = [cell["centroid"][1] for cell in cells]
        cell_ids = [cell["cell_id"] for cell in cells]

        # Plot cell positions
        ax.scatter(x_coords, y_coords, c="red", s=100, alpha=0.7, zorder=3)  # type: ignore

        # Add cell ID labels
        for i, (x, y, cell_id) in enumerate(zip(x_coords, y_coords, cell_ids)):
            ax.annotate(  # type: ignore
                f"C{cell_id}",
                (x, y),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                alpha=0.8,
            )

        # Plot contours if available
        for cell in cells:
            contour = cell.get("contour", [])
            if contour:
                contour_array = np.array(contour + [contour[0]])  # Close the contour
                ax.plot( # type: ignore
                    contour_array[:, 0],
                    contour_array[:, 1],
                    "b-",
                    alpha=0.3,
                    linewidth=1,
                )  

                # For dilate method, show dilated contours
                if method_name == "dilate":
                    # Show a rough approximation of dilation
                    from matplotlib.patches import Polygon

                    dilated_contour = np.array(contour)
                    # Simple dilation approximation by expanding outward
                    center = np.mean(dilated_contour, axis=0)
                    dilated_contour = center + 1.5 * (dilated_contour - center)
                    poly = Polygon(
                        dilated_contour,
                        alpha=0.2,
                        facecolor="orange",
                        edgecolor="orange",
                    )
                    ax.add_patch(poly)

        # Plot edges
        if edge_indices.shape[1] > 0:
            positions_np = cast(np.ndarray[Any, Any], positions.cpu().numpy())  # type: ignore
            edges: list[list[int]] = []
            for i in range(edge_indices.shape[1]):
                src_idx = edge_indices[0, i].item()
                dst_idx = edge_indices[1, i].item()
                src_pos = positions_np[src_idx]  # type: ignore
                dst_pos = positions_np[dst_idx]  # type: ignore
                edges.append([src_pos, dst_pos])

            if edges:
                line_collection = LineCollection(
                    edges, colors="blue", alpha=0.6, linewidths=1.5
                )
                ax.add_collection(line_collection)

        ax.set_title( # type: ignore
            f"{title}\nNodes: {len(cells)}, Edges: {edge_indices.shape[1]}", fontsize=14
        )  
        ax.set_xlabel("X Position (pixels)")  # type: ignore
        ax.set_ylabel("Y Position (pixels)")  # type: ignore
        ax.grid(True, alpha=0.3)  # type: ignore
        ax.set_aspect("equal")  # type: ignore

        # Auto-adjust the view to show all data
        if x_coords and y_coords:
            margin = 50
            ax.set_xlim(min(x_coords) - margin, max(x_coords) + margin)
            ax.set_ylim(min(y_coords) - margin, max(y_coords) + margin)

        plt.tight_layout()

        # Save plot to file instead of base64 to ensure it shows up
        plot_filename = f"plot_{method_name}_{hash(title)}.png"
        plot_path = f"/home/camilo/Thesis/test_reports/{plot_filename}"
        plt.savefig(plot_path, format="png", dpi=150, bbox_inches="tight")  # type: ignore
        plt.close()

        return plot_path

    def test_creator_initialization_knn(self, device: str):
        """Test Creator initialization with KNN method"""
        creator = Creator(GraphCreatorType.knn, device)
        assert creator.method == "knn"
        assert creator.device == device
        assert isinstance(creator.edge_creator, KNNEdgeCreator)
        assert creator.edge_creator.k == 8

    def test_creator_initialization_radius(self, device: str):
        """Test Creator initialization with radius method"""
        creator = Creator(GraphCreatorType.radius, device)
        assert creator.method == "radius"
        assert isinstance(creator.edge_creator, RadiusEdgeCreator)
        assert creator.edge_creator.radius == 100

    def test_creator_initialization_delaunay_radius(self, device: str):
        """Test Creator initialization with delaunay_radius method"""
        creator = Creator(GraphCreatorType.delaunay_radius, device)
        assert creator.method == "delaunay_radius"
        assert isinstance(creator.edge_creator, DelaunayEdgeCreator)
        assert creator.edge_creator.limit_radius == 4000

    def test_creator_initialization_dilate(self, device: str):
        """Test Creator initialization with dilate method"""
        creator = Creator(GraphCreatorType.dilate, device)
        assert creator.method == "dilate"
        assert isinstance(creator.edge_creator, DilateEdgeCreator)
        assert creator.edge_creator.dilation == 40

    def test_create_empty_graph(self, device: str):
        """Test creating graph with empty cells list"""
        creator = Creator(GraphCreatorType.knn, device)

        # Test with empty list
        node_features, edge_indices, edge_features = creator.create([])
        assert node_features.shape == (0, 1)
        assert edge_indices.shape == (2, 0)
        assert edge_features.shape == (0, 3)

        # Test with None (should handle gracefully)
        with patch("cellmil.graph.creator.logger") as mock_logger:
            node_features, edge_indices, edge_features = creator.create([])
            mock_logger.warning.assert_called_once()

    def test_extract_node_features(
        self, device: str, sample_cells_10: list[dict[str, Any]]
    ):
        """Test node feature extraction"""
        creator = Creator(GraphCreatorType.knn, device)
        node_features, positions = creator._extract_node_features(sample_cells_10)  # type: ignore

        assert node_features.shape == (10, 1)
        assert positions.shape == (10, 2)
        assert node_features.dtype == torch.long
        assert positions.dtype == torch.float32

        # Check that cell IDs are correctly extracted
        expected_ids = torch.tensor(
            [cell["cell_id"] for cell in sample_cells_10], dtype=torch.long
        )
        assert torch.equal(node_features[:, 0], expected_ids)

    def test_knn_graph_creation_with_plots(
        self, device: str, sample_cells_10: list[dict[str, Any]]
    ):
        """Test KNN graph creation and generate visualization"""
        creator = Creator(GraphCreatorType.knn, device)
        node_features, edge_indices, edge_features = creator.create(sample_cells_10)

        # Basic validation
        assert node_features.shape[0] == 10
        assert edge_indices.shape[0] == 2
        assert edge_features.shape[0] == edge_indices.shape[1]
        assert edge_features.shape[1] == 3  # distance, direction_x, direction_y

        # Check that we have reasonable number of edges (each node should connect to k=8 neighbors)
        # With 10 nodes, each connects to 8, but edges are undirected, so approximately 10*8/2 = 40 edges
        assert edge_indices.shape[1] > 0
        assert edge_indices.shape[1] <= 10 * 8  # Upper bound

        # Extract positions for plotting
        positions = torch.zeros((10, 2))
        for i, cell in enumerate(sample_cells_10):
            positions[i] = torch.tensor(cell["centroid"])

        # Generate plot
        html_plot = self._create_plot_html(
            sample_cells_10, edge_indices, positions, "KNN Graph (k=8)", "knn"
        )

        # Add plot to pytest HTML report
        pytest.current_test_html = getattr(pytest, "current_test_html", "") + html_plot  # type: ignore

    def test_radius_graph_creation_with_plots(
        self, device: str, sample_cells_10: list[dict[str, Any]]
    ):
        """Test radius graph creation and generate visualization"""
        creator = Creator(GraphCreatorType.radius, device)
        node_features, edge_indices, edge_features = creator.create(sample_cells_10)

        # Basic validation
        assert node_features.shape[0] == 10
        assert edge_indices.shape[0] == 2
        assert edge_features.shape[0] == edge_indices.shape[1]
        assert edge_features.shape[1] == 3

        # Extract positions for plotting
        positions = torch.zeros((10, 2))
        for i, cell in enumerate(sample_cells_10):
            positions[i] = torch.tensor(cell["centroid"])

        # Generate plot
        html_plot = self._create_plot_html(
            sample_cells_10, edge_indices, positions, "Radius Graph (r=150)", "radius"
        )

        # Add plot to pytest HTML report
        pytest.current_test_html = getattr(pytest, "current_test_html", "") + html_plot  # type: ignore

    def test_delaunay_graph_creation_with_plots(
        self, device: str, sample_cells_10: list[dict[str, Any]]
    ):
        """Test Delaunay triangulation graph creation and generate visualization"""
        creator = Creator(GraphCreatorType.delaunay_radius, device)
        node_features, edge_indices, edge_features = creator.create(sample_cells_10)

        # Basic validation
        assert node_features.shape[0] == 10
        assert edge_indices.shape[0] == 2
        assert edge_features.shape[0] == edge_indices.shape[1]
        assert edge_features.shape[1] == 3

        # Extract positions for plotting
        positions = torch.zeros((10, 2))
        for i, cell in enumerate(sample_cells_10):
            positions[i] = torch.tensor(cell["centroid"])

        # Generate plot
        html_plot = self._create_plot_html(
            sample_cells_10,
            edge_indices,
            positions,
            "Delaunay + Radius Graph (limit=4000)",
            "delaunay_radius",
        )

        # Add plot to pytest HTML report
        pytest.current_test_html = getattr(pytest, "current_test_html", "") + html_plot  # type: ignore

    def test_dilate_graph_creation_with_plots(
        self, device: str, sample_cells_10: list[dict[str, Any]]
    ):
        """Test dilate contour graph creation and generate visualization"""
        creator = Creator(GraphCreatorType.dilate, device)

        # Create cells that are closer together so dilation can create intersections
        close_cells: list[dict[str, Any]] = []
        positions = [(300, 300), (320, 300), (340, 300), (300, 320), (320, 320)]

        for i, (x, y) in enumerate(positions):
            # Create square contours around each position
            size = 15
            contour = [
                [int(x - size), int(y - size)],
                [int(x + size), int(y - size)],
                [int(x + size), int(y + size)],
                [int(x - size), int(y + size)],
            ]

            cell: dict[str, Any] = {
                "cell_id": i + 1,
                "centroid": [float(x), float(y)],
                "contour": contour,
            }
            close_cells.append(cell)

        node_features, edge_indices, edge_features = creator.create(close_cells)

        # Basic validation
        assert node_features.shape[0] == 5
        assert edge_indices.shape[0] == 2
        assert edge_features.shape[0] == edge_indices.shape[1]
        assert edge_features.shape[1] == 3

        # Extract positions for plotting
        positions_tensor = torch.zeros((5, 2))
        for i, cell in enumerate(close_cells):
            positions_tensor[i] = torch.tensor(cell["centroid"])

        # Generate plot
        html_plot = self._create_plot_html(
            close_cells,
            edge_indices,
            positions_tensor,
            "Dilate Contour Graph (dilation=40px)",
            "dilate",
        )

        # Add plot to pytest HTML report
        pytest.current_test_html = getattr(pytest, "current_test_html", "") + html_plot  # type: ignore

    def test_edge_feature_calculation(self, device: str):
        """Test edge feature calculation"""
        creator = Creator(GraphCreatorType.knn, device)

        # Create simple test case
        positions = torch.tensor(
            [[0.0, 0.0], [3.0, 4.0], [1.0, 0.0]], dtype=torch.float32
        )
        edge_indices = torch.tensor([[0, 0], [1, 2]], dtype=torch.long)

        edge_features = creator.edge_creator._calculate_edge_features(  # type: ignore
            edge_indices, positions
        ) 

        assert edge_features.shape == (2, 3)

        # First edge: (0,0) -> (3,4), distance = 5, direction = (0.6, 0.8)
        assert abs(edge_features[0, 0].item() - 5.0) < 1e-5
        assert abs(edge_features[0, 1].item() - 0.6) < 1e-5
        assert abs(edge_features[0, 2].item() - 0.8) < 1e-5

        # Second edge: (0,0) -> (1,0), distance = 1, direction = (1.0, 0.0)
        assert abs(edge_features[1, 0].item() - 1.0) < 1e-5
        assert abs(edge_features[1, 1].item() - 1.0) < 1e-5
        assert abs(edge_features[1, 2].item() - 0.0) < 1e-5

    def test_knn_edge_creator_invalid_k(self, device: str):
        """Test KNN edge creator with invalid k value"""
        edge_creator = KNNEdgeCreator(device, k=None)
        positions = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float32)

        with pytest.raises(ValueError, match="K value is not set"):
            edge_creator.create_edges(positions)

    def test_radius_edge_creator_invalid_radius(self, device: str):
        """Test radius edge creator with invalid radius value"""
        edge_creator = RadiusEdgeCreator(device, radius=None)
        positions = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float32)

        with pytest.raises(ValueError, match="Radius value is not set"):
            edge_creator.create_edges(positions)

    def test_delaunay_edge_creator_invalid_limit_radius(self, device: str):
        """Test Delaunay edge creator with invalid limit radius value"""
        edge_creator = DelaunayEdgeCreator(device, limit_radius=None)
        positions = torch.tensor(
            [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=torch.float32
        )

        with pytest.raises(ValueError, match="Limit radius value is not set"):
            edge_creator.create_edges(positions)

    def test_delaunay_too_few_points(self, device: str):
        """Test Delaunay triangulation with too few points"""
        edge_creator = DelaunayEdgeCreator(device, limit_radius=100)
        positions = torch.tensor(
            [[0.0, 0.0], [1.0, 1.0]], dtype=torch.float32
        )  # Only 2 points

        edge_indices, edge_features = edge_creator.create_edges(positions)

        assert edge_indices.shape == (2, 0)
        assert edge_features.shape == (0, 3)

    def test_knn_single_cell(self, device: str):
        """Test KNN with single cell"""
        edge_creator = KNNEdgeCreator(device, k=5)
        positions = torch.tensor([[0.0, 0.0]], dtype=torch.float32)

        edge_indices, edge_features = edge_creator.create_edges(positions)

        assert edge_indices.shape == (2, 0)
        assert edge_features.shape == (0, 3)

    def test_dilate_no_cells_provided(self, device: str):
        """Test dilate edge creator when no cells are provided"""
        edge_creator = DilateEdgeCreator(device, dilation=40)
        positions = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float32)

        edge_indices, edge_features = edge_creator.create_edges(positions, cells=None)

        assert edge_indices.shape == (2, 0)
        assert edge_features.shape == (0, 3)

    def test_dilate_single_cell(self, device: str):
        """Test dilate edge creator with single cell"""
        edge_creator = DilateEdgeCreator(device, dilation=40)
        positions = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
        cells: list[dict[str, Any]] = [
            {
                "cell_id": 1,
                "centroid": [0.0, 0.0],
                "contour": [[0, 0], [1, 0], [1, 1], [0, 1]],
            }
        ]

        edge_indices, edge_features = edge_creator.create_edges(positions, cells)

        assert edge_indices.shape == (2, 0)
        assert edge_features.shape == (0, 3)

    def test_device_handling(self, sample_cells_small: list[dict[str, Any]]):
        """Test that creator handles different devices correctly"""
        device = "cpu"  # Use CPU for testing
        creator = Creator(GraphCreatorType.knn, device)

        node_features, edge_indices, edge_features = creator.create(sample_cells_small)

        # Results should be on CPU after processing
        assert node_features.device.type == "cpu"
        assert edge_indices.device.type == "cpu"
        assert edge_features.device.type == "cpu"

    def test_logging_output(
        self, device: str, sample_cells_small: list[dict[str, Any]]
    ):
        """Test that appropriate logging messages are generated"""
        with patch("cellmil.graph.creator.logger") as mock_logger:
            creator = Creator(GraphCreatorType.knn, device)
            creator.create(sample_cells_small)

            # Should log creation info
            assert mock_logger.info.call_count >= 3

            # Check for specific log messages
            call_args = [call.args[0] for call in mock_logger.info.call_args_list]
            assert any("Creating graph for" in msg for msg in call_args)
            assert any("Extracting node features" in msg for msg in call_args)
            assert any("Creating edges" in msg for msg in call_args)

    def test_comprehensive_comparison_plot(
        self, device: str, sample_cells_10: list[dict[str, Any]]
    ):
        """Create a comprehensive comparison plot of all graph types"""
        methods: list[GraphCreatorType] = [
            GraphCreatorType.knn,
            GraphCreatorType.radius,
            GraphCreatorType.delaunay_radius,
            GraphCreatorType.dilate,
        ]

        # Create a special set of cells for comparison - closer together for dilate method
        comparison_cells: list[dict[str, Any]] = []
        # Create a 3x3 grid pattern with some variation
        positions: list[tuple[float, float]] = []
        for i in range(3):
            for j in range(3):
                x = 300 + i * 80 + np.random.normal(0, 10)
                y = 300 + j * 80 + np.random.normal(0, 10)
                positions.append((x, y))

        for idx, (x, y) in enumerate(positions):
            size = 20
            contour = [
                [int(x - size), int(y - size)],
                [int(x + size), int(y - size)],
                [int(x + size), int(y + size)],
                [int(x - size), int(y + size)],
            ]

            cell: dict[str, Any] = {
                "cell_id": idx + 1,
                "centroid": [float(x), float(y)],
                "contour": contour,
            }
            comparison_cells.append(cell)

        # Generate plots for all methods
        comparison_html = (
            "<div style='margin: 30px 0;'><h2>Graph Creation Methods Comparison</h2>"
        )
        comparison_html += "<p>Comparison of different graph creation methods on the same set of 9 cells arranged in a 3x3 grid pattern.</p>"

        for method in methods:
            creator = Creator(method, device)
            _, edge_indices, _ = creator.create(comparison_cells)

            # Extract positions for plotting
            positions_tensor = torch.zeros((len(comparison_cells), 2))
            for i, cell in enumerate(comparison_cells):
                positions_tensor[i] = torch.tensor(cell["centroid"])

            # Generate plot for this method
            method_titles = {
                "knn": "K-Nearest Neighbors (k=8)",
                "radius": "Radius-based (r=150px)",
                "delaunay_radius": "Delaunay Triangulation + Radius Filter (limit=4000px)",
                "dilate": "Dilated Contour Intersection (dilation=40px)",
            }

            html_plot = self._create_plot_html(
                comparison_cells,
                edge_indices,
                positions_tensor,
                method_titles[method],
                method,
            )
            comparison_html += html_plot

        comparison_html += "</div>"

        # Add comparison to pytest HTML report
        pytest.current_test_html = ( # type: ignore
            getattr(pytest, "current_test_html", "") + comparison_html
        )  


# Hook to add HTML content to pytest report
def pytest_runtest_setup(item: Any):
    """Setup hook to initialize HTML content for each test"""
    pytest.current_test_html = ""  # type: ignore


def pytest_runtest_teardown(item: Any):
    """Teardown hook to add HTML content to test report"""
    if hasattr(pytest, "current_test_html") and pytest.current_test_html:  # type: ignore
        # Add HTML content to the test item for the HTML reporter
        if hasattr(item, "user_properties"):
            item.user_properties.append(("html", pytest.current_test_html))  # type: ignore

        # Also store in test node for HTML reporter
        if hasattr(item, "_report_sections"):
            item._report_sections = getattr(item, "_report_sections", [])
            item._report_sections.append(("html_content", pytest.current_test_html))  # type: ignore


@pytest.fixture(autouse=True)
def setup_matplotlib():
    """Setup matplotlib for non-interactive plotting"""
    plt.switch_backend("Agg")  # Use non-interactive backend

import torch
import json
import random
import numpy as np
import plotly.graph_objs as go  # type: ignore
from typing import Any, cast
from pathlib import Path
from cellmil.utils import logger
from cellmil.interfaces import GraphCreatorConfig
from cellmil.interfaces.GraphCreatorConfig import GraphCreatorCategory
from cellmil.interfaces.FeatureExtractorConfig import ExtractorType
from cellmil.datamodels.datasets.utils import get_cell_features, validate_features, get_cell_types, cell_types_to_tensor
from cellmil.datamodels.transforms.correlation_filter import CorrelationFilterTransform
from cellmil.datamodels.transforms.normalization import RobustScalerTransform
from .debug_visualizer import GraphDebugVisualizer
from .creator import Creator

class GraphCreator:
    """Main class for graph creation from segmented cells."""
    def __init__(self, config: GraphCreatorConfig) -> None:
        self.config = config
        self.config.patched_slide_path = Path(self.config.patched_slide_path)
        self.device = f"cuda:{self.config.gpu}" if torch.cuda.is_available() else "cpu"         
        # self.device = "cpu"         
        
        # Create the creator with appropriate parameters
        self.creator = Creator(
            self.config.method, 
            self.device,
        )

        # TODO: Make configurable
        self.correlation_threshold = 0.9
        self.feature_extractors = [ExtractorType.morphometrics, ExtractorType.pyradiomics_hed]
        self.cell_type = True
        self.debug_sample_size = 2000  # Number of cells to sample in debug mode
        # TODO: -----
        
        logger.info(f"Graph creator initialized with config: {self.config}, GPU: {self.device}")
        
        self._load_cells()
        
        if self.config.method in GraphCreatorCategory.FeatureDependent:
            logger.info(f"Using feature-dependent graph creation method: {self.config.method}")
            self._load_morphological_features(self.feature_extractors)

    def _load_morphological_features(self, feature_extractors: list[ExtractorType]) -> None:
        """Load morphological features for similarity-based graph creation."""    
        
        valid_features = validate_features(
            folder=self.config.patched_slide_path.parent,
            slide_name=self.config.patched_slide_path.name,
            extractor=feature_extractors,
            segmentation_model=self.config.segmentation_model
        )
        
        if valid_features is False:
            raise ValueError("Morphological features are not valid for graph creation.")
        
        features, cell_indices, _ = get_cell_features(
            folder=self.config.patched_slide_path.parent,
            slide_name=self.config.patched_slide_path.name,
            extractor=feature_extractors,
            segmentation_model=self.config.segmentation_model
        )
        
        if features is None or cell_indices is None:
            raise ValueError("Failed to load morphological features for graph creation.")
        
        # Filter cells to only those with available features
        original_count = len(self.cells)
        self.cells = [cell for cell in self.cells if cell["cell_id"] in cell_indices]
        filtered_count = len(self.cells)
        
        if filtered_count < original_count:
            logger.info(
                f"Filtered cells from {original_count} to {filtered_count} based on available features. "
                f"Excluded {original_count - filtered_count} cells without features."
            )
        
        # Apply correlation filtering to remove redundant features
        logger.info(f"Applying correlation filter to {features.shape[1]} features...")
        corr_filter = CorrelationFilterTransform(
            correlation_threshold=self.correlation_threshold,
            plot_correlation_matrix=False,
        )
        corr_filter.fit(features)
        filtered_features = corr_filter.transform(features)
        logger.info(f"Features reduced to {filtered_features.shape[1]} after correlation filtering")
        
        # Apply robust scaling for normalization
        logger.info("Applying robust scaling to features...")
        robust_scaler = RobustScalerTransform(
            apply_log_transform=True,
            quantile_range=(0.25, 0.75),
            clip_quantiles=(0.005, 0.995),
            constant_threshold=1e-8,
        )
        robust_scaler.fit(filtered_features)
        normalized_features = robust_scaler.transform(filtered_features)
        logger.info("Features normalized using robust scaling")
        
        # Load and concatenate cell types if requested
        if self.cell_type:
            logger.info("Loading cell types for feature concatenation...")
            cell_types_dict = get_cell_types(
                self.config.patched_slide_path.parent,
                self.config.patched_slide_path.name,
                self.config.segmentation_model
            )
            
            if cell_types_dict is None or len(cell_types_dict) == 0:
                logger.warning(f"No cell types found for slide. Skipping cell type concatenation.")
                cell_types_tensor = None
            else:
                # Convert cell types to one-hot tensor
                cell_types_tensor = cell_types_to_tensor(cell_types_dict, cell_indices)
                logger.info(f"Loaded cell types: {cell_types_tensor.shape}")
                
                # Concatenate cell types to features
                normalized_features = torch.cat([normalized_features, cell_types_tensor], dim=1)
                # normalized_features = cell_types_tensor.float()
                logger.info(
                    f"Concatenated {cell_types_tensor.shape[1]} cell type features. "
                    f"New feature dimension: {normalized_features.shape[1]}"
                )
        
        # Add processed features to each cell using cell_indices
        # cell_indices is a dict: {cell_id: row_index_in_features}
        for cell in self.cells:
            cell_id = cell["cell_id"]
            if cell_id in cell_indices:
                feature_idx = cell_indices[cell_id]
                cell["features"] = normalized_features[feature_idx, :]
        
        logger.info(
            f"Added {normalized_features.shape[1]} processed features to {len(self.cells)} cells"
        )

        
        
    def _load_cells(self):
        """Load cell data from the specified path."""
        json_path = (
            self.config.patched_slide_path
            / "cell_detection"
            / self.config.segmentation_model
            / "cells.json"
        )
        if not json_path.exists():
            raise FileNotFoundError(f"Cells path {json_path} does not exist.")

        with open(json_path, "r") as f:
            self.json = json.load(f)
        all_cells: list[dict[str, Any]] = self.json["cells"]
        
        # Sample cells if in debug mode
        if self.config.debug and len(all_cells) > self.debug_sample_size:
            random.seed(42)  # For reproducibility
            self.cells = random.sample(all_cells, self.debug_sample_size)
            logger.info(f"Debug mode: Sampled {len(self.cells)} cells from {len(all_cells)} total cells")
        else:
            self.cells = all_cells
            logger.info(f"Loaded {len(self.cells)} cells from {json_path}")

    def create_graph(self) -> None:
        """Create a graph from the segmented cells."""
        
        # Launch debug visualizer if in debug mode
        if self.config.debug:
            self._launch_debug_visualizer()
            return
        
        logger.info(f"Creating graph using {self.config.method} method...")
        
        # Create graph using the Creator class
        node_features, edge_indices, edge_features = self.creator.create(
            self.cells
        )
        
        # Save graph
        self._save_graph(node_features, edge_indices, edge_features)

        # Plot graph
        if self.config.plot:
            self._plot_graph(node_features, edge_indices)
        
        logger.info("Graph creation completed successfully.")
    
    def _launch_debug_visualizer(self) -> None:
        """Launch the interactive debug visualizer."""
        
        logger.info("Launching debug visualizer...")
        logger.info(f"Using {len(self.cells)} sampled cells for visualization")
        
        visualizer = GraphDebugVisualizer(
            cells=self.cells,
            method=self.config.method,
            device=self.device,
        )
        
        visualizer.run(port=8050, debug=True)

    def _save_graph(
        self, 
        node_features: torch.Tensor, 
        edge_indices: torch.Tensor, 
        edge_features: torch.Tensor
    ):
        """Save the graph to a PyTorch file."""
        # Create output directory
        output_dir = (
            self.config.patched_slide_path
            / "graphs"
            / self.config.method
            / self.config.segmentation_model
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare graph data
        graph_data: dict[str, Any] = {
            "node_features": node_features,
            "edge_indices": edge_indices,
            "edge_features": edge_features,
            "metadata": {
                "n_nodes": node_features.shape[0],
                "n_edges": edge_indices.shape[1],
                "node_feature_dim": node_features.shape[1],
                "edge_feature_dim": edge_features.shape[1],
                "graph_method": self.config.method,
                "segmentation_model": self.config.segmentation_model,
                "name_node_features": "cell_id",
                "name_edge_features": ["distance", "direction_x", "direction_y"]
            }
        }
        
        # Save graph
        output_path = output_dir / "graph.pt"
        torch.save(graph_data, output_path)
        
        logger.info(f"Graph saved to {output_path}")
        logger.info(f"Graph statistics: {graph_data['metadata']['n_nodes']} nodes, {graph_data['metadata']['n_edges']} edges")

    def _plot_graph(self, node_features: torch.Tensor, edge_indices: torch.Tensor) -> None:
        """Create and save a Plotly visualization of the graph."""
        try:
            # Create output directory for images
            output_dir = (
                self.config.patched_slide_path
                / "graphs"
                / self.config.method
                / self.config.segmentation_model
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            
            title = f'Cell Graph Visualization ({self.config.method})'
            
            positions = self._get_cells_position(node_features)
            
            # Convert positions to numpy for plotting
            pos_np = cast(
                np.ndarray[Any, Any],
                positions.cpu().numpy() if positions.is_cuda else positions.numpy(),  # type: ignore
            )
            edges_np = cast(
                np.ndarray[Any, Any],
                (
                    edge_indices.cpu().numpy()  # type: ignore
                    if edge_indices.is_cuda
                    else edge_indices.numpy()  # type: ignore
                ),
            )
            
            max_nodes_to_plot = 100000
            
            if len(pos_np) > max_nodes_to_plot:
                # Sample nodes first
                node_sample_indices = np.random.choice(
                    len(pos_np), max_nodes_to_plot, replace=False
                )
                node_set = set(node_sample_indices)
                pos_sample = pos_np[node_sample_indices]

                # Create mapping from old indices to new indices
                old_to_new = {
                    old_idx: new_idx for new_idx, old_idx in enumerate(node_sample_indices)
                }

                logger.info(
                    f"Sampling {max_nodes_to_plot} nodes for visualization from {len(pos_np)} total nodes"
                )

                # Filter edges to only include those between sampled nodes
                valid_edges: list[list[int]] = []
                if edges_np.shape[1] > 0:
                    for i in range(edges_np.shape[1]):
                        src_idx, dst_idx = edges_np[0, i], edges_np[1, i]
                        if src_idx in node_set and dst_idx in node_set:
                            # Remap indices to the sampled node indices
                            new_src = old_to_new[src_idx]
                            new_dst = old_to_new[dst_idx]
                            valid_edges.append([new_src, new_dst])

                if valid_edges:
                    edges_to_plot = np.array(valid_edges).T
                    logger.info(
                        f"Plotting {len(valid_edges)} edges between sampled nodes from {edges_np.shape[1]} total edges"
                    )
                else:
                    edges_to_plot = np.empty((2, 0))
            else:
                pos_sample = pos_np
                edges_to_plot = edges_np
            
            # Create edge traces for Plotly
            edge_x: list[float | None] = []
            edge_y: list[float | None] = []
            
            if edges_to_plot.shape[1] > 0:
                for i in range(edges_to_plot.shape[1]):
                    start_idx, end_idx = int(edges_to_plot[0, i]), int(edges_to_plot[1, i])
                    start_pos = pos_sample[start_idx]
                    end_pos = pos_sample[end_idx]
                    
                    edge_x.extend([start_pos[0], end_pos[0], None])
                    edge_y.extend([start_pos[1], end_pos[1], None])
            
            edge_trace = go.Scatter(
                x=edge_x,
                y=edge_y,
                line=dict(width=1.5, color='rgba(70, 130, 180, 0.6)'),
                hoverinfo='none',
                mode='lines',
                showlegend=False
            )
            
            # Create node trace
            node_x = pos_sample[:, 0].tolist()
            node_y = pos_sample[:, 1].tolist()
            
            node_trace = go.Scatter(
                x=node_x,
                y=node_y,
                mode='markers',
                hoverinfo='text',
                text=[f'Cell {i}' for i in range(len(pos_sample))],
                marker=dict(
                    size=4,
                    color='rgba(255, 100, 100, 0.7)',
                    line=dict(width=0.5, color='rgba(255, 255, 255, 0.5)'),
                    opacity=0.8
                ),
                showlegend=False
            )
            
            # Create figure with elegant styling
            fig = go.Figure(
                data=[edge_trace, node_trace],
                layout=go.Layout(
                    title=dict(
                        text=title,
                        font=dict(size=20, color='black', family='Arial Black'),
                        x=0.5
                    ),
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=30, l=20, r=20, t=60),
                    annotations=[
                        dict(
                            text=f"Nodes: {len(pos_sample):,} | Edges: {edges_to_plot.shape[1]:,}",
                            showarrow=False,
                            xref="paper",
                            yref="paper",
                            x=0.02,
                            y=0.98,
                            xanchor='left',
                            yanchor='top',
                            font=dict(
                                color='rgba(0, 0, 0, 0.6)',
                                size=12,
                                family='Arial'
                            )
                        )
                    ],
                    xaxis=dict(
                        showgrid=False,
                        zeroline=False,
                        showticklabels=True,
                        title='X coordinate',
                        showline=True,
                        linecolor='rgba(0, 0, 0, 0.3)'
                    ),
                    yaxis=dict(
                        showgrid=False,
                        zeroline=False,
                        showticklabels=True,
                        title='Y coordinate',
                        autorange='reversed',
                        showline=True,
                        linecolor='rgba(0, 0, 0, 0.3)',
                        scaleanchor='x',
                        scaleratio=1
                    ),
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font=dict(color='black')
                )
            )
            
            # Save interactive HTML plot
            html_path = output_dir / "graph_interactive.html"
            fig.write_html(str(html_path))  # type: ignore
            
            # Save static PNG image
            png_path = output_dir / "graph_visualization.png"
            fig.write_image(str(png_path), width=1920, height=1080, scale=2)  # type: ignore
            
            logger.info(f"Static graph visualization saved to {png_path}")
            logger.info(f"Interactive graph visualization saved to {html_path}")
            
        except Exception as e:
            logger.warning(f"Failed to create graph visualization: {e}")
            import traceback
            traceback.print_exc()

    def _get_cells_position(self, node_features: torch.Tensor) -> torch.Tensor:
        """Get the positions of the cell centroid for visualization."""
        # Extract centroid positions from cells
        centroids = np.array([cell["centroid"] for cell in self.cells])
        
        # Convert to tensor
        positions = torch.from_numpy(centroids).float() # type: ignore

        return positions

"""
Graph visualization for GraphMIL attention weights.

This module creates interactive graph plots showing attention weights on edges
and nodes, specifically designed for GraphMIL models.
"""

import torch
import numpy as np
import networkx as nx
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Any, cast

from cellmil.interfaces.ExplainerCreatorConfig import ExplainerCreatorConfig
from cellmil.explainability.core.attention_extractor import AttentionResult
from cellmil.utils import logger

from plotly.graph_objs import Scatter  # type: ignore
import plotly.graph_objects as go  # type: ignore
from plotly.subplots import make_subplots  # type: ignore
from torch_geometric.data import Data  # type: ignore


class AttentionGraphVisualizer:
    """Creates graph visualizations for GraphMIL attention weights."""

    def __init__(self, config: ExplainerCreatorConfig):
        self.config = config

    def create_visualization(
        self,
        attention_result: AttentionResult,
        graph_data: Data,
        output_path: Path,
        cell_coordinates: Optional[Dict[int, Tuple[float, float]]] = None,
    ) -> List[Path]:
        """
        Create graph visualizations for GraphMIL attention weights.

        Args:
            attention_result: AttentionResult from GraphMIL model
            graph_data: PyTorch Geometric Data object with graph structure
            output_path: Directory to save visualization files
            cell_coordinates: Optional mapping from node_id to (x, y) coordinates

        Returns:
            List of paths to created visualization files
        """
        logger.info("Creating graph attention visualizations")

        output_path.mkdir(parents=True, exist_ok=True)
        created_files: list[Path] = []

        # Extract graph structure
        edge_index = cast(
            np.ndarray[Any, Any],
            graph_data.edge_index.detach().cpu().numpy(),  # type: ignore
        )

        if hasattr(graph_data, "x") and graph_data.x is not None:
            num_nodes = graph_data.x.shape[0]
        else:
            raise ValueError(
                "Graph data must have node features 'x' to determine number of nodes"
            )

        # Create NetworkX graph
        G = self._create_networkx_graph(edge_index, num_nodes)

        # Get node positions
        pos = self._get_node_positions(G, cell_coordinates)

        # Create visualizations for different attention types
        # More flexible filtering to handle different naming conventions

        gnn_attentions = {
            k: v
            for k, v in attention_result.attention_weights.items()
            if any(
                gnn_pattern in k
                for gnn_pattern in ["gnn_attention", "selected_gnn_layer"]
            )
        }
        pooling_attentions = {
            k: v
            for k, v in attention_result.attention_weights.items()
            if any(
                pool_pattern in k
                for pool_pattern in [
                    "pooling_attention",
                    "random_walk",
                    "mean_attention",
                    "head_",  # Include individual Head4Type heads
                    "class_",  # Include individual CLAM classes
                    # "initial_state",
                ]
            )
        }

        # GNN attention visualizations (edge-based)
        if gnn_attentions:
            gnn_files = self._create_gnn_attention_plots(
                G, pos, gnn_attentions, edge_index, output_path
            )
            created_files.extend(gnn_files)

        # Pooling attention visualizations (node-based)
        if pooling_attentions:
            pooling_files = self._create_pooling_attention_plots(
                G, pos, pooling_attentions, output_path
            )
            created_files.extend(pooling_files)

        # Combined visualization
        # if gnn_attentions and pooling_attentions:
        #     combined_files = self._create_combined_plots(
        #         G, pos, gnn_attentions, pooling_attentions, edge_index, output_path
        #     )
        #     created_files.extend(combined_files)

        logger.info(
            f"Created {len(created_files)} visualization files in {output_path}"
        )

        return created_files

    def _create_networkx_graph(
        self, edge_index: np.ndarray[Any, Any], num_nodes: int
    ) -> nx.Graph:
        """Create NetworkX graph from edge index."""
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))  # type: ignore

        # Add edges
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i], edge_index[1, i]
            G.add_edge(src, dst)  # type: ignore

        return G

    def _get_node_positions(
        self,
        G: nx.Graph,
        cell_coordinates: Optional[Dict[int, Tuple[float, float]]],
    ) -> Dict[int, Tuple[float, float]]:
        """Get node positions for graph layout."""

        if cell_coordinates:
            # Use provided spatial coordinates
            return cell_coordinates
        else:
            raise ValueError("Cell coordinates must be provided for node positioning")

    def _create_gnn_attention_plots(
        self,
        G: nx.Graph,
        pos: dict[int, tuple[float, float]],
        gnn_attentions: dict[str, torch.Tensor],
        edge_index: np.ndarray[Any, Any],
        output_path: Path,
    ) -> list[Path]:
        """Create GNN attention visualizations (edge-based)."""

        created_files: list[Path] = []

        for attention_key, attention_weights in gnn_attentions.items():
            plotly_file = self._create_plotly_edge_plot(
                G, pos, attention_weights, edge_index, attention_key, output_path
            )
            if plotly_file:
                created_files.append(plotly_file)

        return created_files

    def _create_pooling_attention_plots(
        self,
        G: nx.Graph,
        pos: Dict[int, Tuple[float, float]],
        pooling_attentions: Dict[str, torch.Tensor],
        output_path: Path,
    ) -> List[Path]:
        """Create pooling attention visualizations (node-based)."""

        created_files: list[Path] = []

        for attention_key, attention_weights in pooling_attentions.items():
            plotly_file_robust = self._create_plotly_node_plot(
                G, pos, attention_weights, attention_key, output_path, robust_color=True
            )
            if plotly_file_robust:
                created_files.append(plotly_file_robust)

            plotly_file_standard = self._create_plotly_node_plot(
                G,
                pos,
                attention_weights,
                attention_key,
                output_path,
                robust_color=False,
            )
            if plotly_file_standard:
                created_files.append(plotly_file_standard)

        logger.info(
            f"Created {len(created_files)} pooling attention visualization files in {output_path}"
        )

        return created_files

    def _create_plotly_edge_plot(
        self,
        G: nx.Graph,
        pos: Dict[int, Tuple[float, float]],
        attention_weights: torch.Tensor,
        edge_index: np.ndarray[Any, Any],
        attention_key: str,
        output_path: Path,
    ) -> Optional[Path]:
        """Create interactive Plotly plot for edge-based attention."""
        try:
            # Process attention weights
            if len(attention_weights.shape) == 2:
                edge_attention = cast(
                    np.ndarray[Any, Any],
                    attention_weights.mean(dim=1).detach().cpu().numpy(),  # type: ignore
                )
            else:
                edge_attention = cast(
                    np.ndarray[Any, Any],
                    attention_weights.detach().cpu().numpy(),  # type: ignore
                )

            # Create edge traces with attention-based coloring
            # Normalize attention values for color mapping
            if len(edge_attention) > 0:
                edge_attention_norm = (edge_attention - edge_attention.min()) / (
                    edge_attention.max() - edge_attention.min() + 1e-8
                )
            else:
                edge_attention_norm = edge_attention

            # Create individual edge traces for better coloring
            edge_traces: list[Scatter] = []
            for i, (src, dst) in enumerate(edge_index.T):
                if i < len(edge_attention):
                    x0, y0 = pos[src]
                    x1, y1 = pos[dst]

                    attention_val = edge_attention_norm[i]
                    # Map attention to color intensity with better contrast
                    color_intensity = attention_val
                    line_width = 1 + attention_val * 4  # Width from 1 to 5

                    # Create gorgeous color gradient optimized for black background
                    # Low attention: soft cyan/teal, High attention: vibrant magenta/pink
                    if color_intensity < 0.33:
                        # Soft cyan to bright cyan for low attention
                        intensity = color_intensity * 3  # 0 to 1
                        r = int(50 + 100 * intensity)  # 50 to 150
                        g = int(200 + 55 * intensity)  # 200 to 255
                        b = int(200 + 55 * intensity)  # 200 to 255
                        edge_color = f"rgb({r}, {g}, {b})"
                    elif color_intensity < 0.66:
                        # Cyan to purple transition for medium attention
                        intensity = (color_intensity - 0.33) * 3  # 0 to 1
                        r = int(150 + 105 * intensity)  # 150 to 255
                        g = int(255 - 155 * intensity)  # 255 to 100
                        b = int(255 - 55 * intensity)  # 255 to 200
                        edge_color = f"rgb({r}, {g}, {b})"
                    else:
                        # Purple to vibrant magenta for high attention
                        intensity = (color_intensity - 0.66) * 3  # 0 to 1
                        r = int(255)  # Constant bright red
                        g = int(100 - 50 * intensity)  # 100 to 50
                        b = int(200 + 55 * intensity)  # 200 to 255
                        edge_color = f"rgb({r}, {g}, {b})"

                    edge_trace = go.Scatter(
                        x=[x0, x1],
                        y=[y0, y1],
                        mode="lines",
                        line=dict(width=line_width, color=edge_color),
                        hoverinfo="text",
                        hovertext=f"Edge {src}-{dst}<br>Attention: {edge_attention[i]:.4f}",
                        showlegend=False,
                    )
                    edge_traces.append(edge_trace)

            # Node trace - ultra-small and elegant for black background
            node_x = [pos[node][0] for node in G.nodes()]  # type: ignore
            node_y = [pos[node][1] for node in G.nodes()]  # type: ignore

            node_trace = go.Scatter(
                x=node_x,
                y=node_y,
                mode="markers",
                hoverinfo="text",
                text=[f"Node {i}" for i in G.nodes()],  # type: ignore
                marker=dict(
                    size=2,  # Ultra-small size for elegance
                    color="rgba(220, 220, 220, 0.4)",  # Soft white with transparency
                    line=dict(width=0.3, color="rgba(255, 255, 255, 0.6)"),
                    opacity=0.4,  # Subtle presence
                ),
                showlegend=False,
            )

            # Create figure with gorgeous black theme
            all_traces = edge_traces + [node_trace]
            fig = go.Figure(
                data=all_traces,
                layout=go.Layout(
                    title=dict(
                        text=f"GNN Attention: {attention_key.replace('_', ' ').title()}",
                        font=dict(size=20, color="white", family="Arial Black"),
                        x=0.5,  # Center the title
                    ),
                    showlegend=False,
                    hovermode="closest",
                    margin=dict(b=30, l=20, r=20, t=60),
                    annotations=[
                        dict(
                            text="Interactive Graph Visualization",
                            showarrow=False,
                            xref="paper",
                            yref="paper",
                            x=0.02,
                            y=0.98,
                            xanchor="left",
                            yanchor="top",
                            font=dict(
                                color="rgba(255, 255, 255, 0.6)",
                                size=12,
                                family="Arial",
                            ),
                        )
                    ],
                    xaxis=dict(
                        showgrid=False,
                        zeroline=False,
                        showticklabels=False,
                        showline=False,
                    ),
                    yaxis=dict(
                        showgrid=False,
                        zeroline=False,
                        showticklabels=False,
                        autorange="reversed",
                        showline=False,
                    ),
                    plot_bgcolor="black",  # Beautiful black background
                    paper_bgcolor="black",  # Black paper background too
                    font=dict(color="white"),  # White text throughout
                ),
            )

            # Save plot
            plot_path = output_path / f"{attention_key}_plotly.html"
            fig.write_html(str(plot_path))  # type: ignore

            return plot_path

        except Exception as e:
            logger.error(f"Error creating Plotly edge plot: {e}")
            return None

    def _create_plotly_node_plot(
        self,
        G: nx.Graph,
        pos: Dict[int, Tuple[float, float]],
        attention_weights: torch.Tensor,
        attention_key: str,
        output_path: Path,
        robust_color: bool = False,
    ) -> Optional[Path]:
        """Create interactive Plotly plot for node-based attention."""
        try:
            # Process attention weights
            logger.info(
                f"Plotly - Raw attention weights shape: {attention_weights.shape}"
            )

            # Handle multi-head attention by taking the mean across heads
            if len(attention_weights.shape) == 2:
                if attention_weights.shape[0] == 1:
                    # Single head case: [1, num_nodes] -> [num_nodes]
                    node_attention = cast(
                        np.ndarray[Any, Any],
                        attention_weights.squeeze(0).detach().cpu().numpy(),  # type: ignore
                    )
                    logger.info(
                        "Plotly - Applied squeeze(0) to single-head attention weights"
                    )
                else:
                    # Multi-head case: [num_heads, num_nodes] -> [num_nodes] (mean across heads)
                    node_attention = cast(
                        np.ndarray[Any, Any],
                        attention_weights.mean(dim=0).detach().cpu().numpy(),  # type: ignore
                    )
                    logger.info(
                        f"Plotly - Averaged across {attention_weights.shape[0]} attention heads"
                    )
            else:
                # 1D case: [num_nodes]
                node_attention = cast(
                    np.ndarray[Any, Any],
                    attention_weights.detach().cpu().numpy(),  # type: ignore
                )
                logger.info("Plotly - Used 1D attention weights directly")

            # Create edge traces (light gray)
            edge_x: list[float | None] = []
            edge_y: list[float | None] = []
            for edge in G.edges():  # type: ignore
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])

            edge_trace = go.Scatter(
                x=edge_x,
                y=edge_y,
                line=dict(
                    width=0.8, color="rgba(100, 100, 100, 0.3)"
                ),  # Subtle dark gray edges
                hoverinfo="none",
                mode="lines",
            )

            # Node trace with attention coloring
            node_x = [pos[node][0] for node in G.nodes()]  # type: ignore
            node_y = [pos[node][1] for node in G.nodes()]  # type: ignore

            logger.info(
                f"Node attention shape: {node_attention.shape}, Graph nodes: {len(G.nodes())}"  # type: ignore
            )
            logger.info(
                f"Node attention values range: [{node_attention.min():.4f}, {node_attention.max():.4f}]"
            )

            if len(node_attention) == len(G.nodes()):  # type: ignore
                node_color = node_attention
                hover_text = [
                    f"Node {i}<br>Attention: {node_attention[i]:.4f}"
                    for i in range(len(G.nodes()))  # type: ignore
                ]
                logger.info(
                    "Successfully applied attention-based node coloring for Plotly"
                )
            else:
                node_color = "red"
                hover_text = [f"Node {i}" for i in G.nodes()]  # type: ignore
                logger.warning(
                    f"Plotly attention shape mismatch: {len(node_attention)} vs {len(G.nodes())} nodes"  # type: ignore
                )

            # Calculate smaller, more elegant node sizes based on attention
            if isinstance(node_color, np.ndarray):
                # Normalize attention values to smaller, more elegant size range
                min_attention = node_attention.min()
                max_attention = node_attention.max()
                if max_attention > min_attention:
                    normalized_attention = (node_attention - min_attention) / (
                        max_attention - min_attention
                    )
                    node_sizes = (
                        1 + 7 * normalized_attention
                    )  # Sizes from 4 to 12 (much smaller)
                    # node_sizes = 6
                else:
                    node_sizes = 6  # Smaller default size
            else:
                node_sizes = 6  # Smaller default size for fallback case

            # Apply robust color scaling to handle outliers better
            if isinstance(node_color, np.ndarray) and robust_color:
                # Use percentile-based robust scaling for color
                p1, p99 = np.percentile(node_attention, [1, 99])

                # Create robust color values that compress outliers
                robust_colors = np.where(
                    node_attention <= p1,
                    0,  # Bottom 1% get minimum color
                    np.where(
                        node_attention >= p99,
                        1,  # Top 1% get maximum color
                        (node_attention - p1)
                        / (p99 - p1),  # Middle 98% get proportional scaling
                    ),
                )

                # Set custom color range to make the visualization more interpretable
                color_range = [p1, p99]
            else:
                robust_colors = node_color
                color_range = None

            node_trace = go.Scatter(
                x=node_x,
                y=node_y,
                mode="markers",
                hoverinfo="text",
                text=hover_text,
                marker=dict(
                    size=node_sizes,
                    color=robust_colors,
                    colorscale="Turbo",  # Beautiful plasma colorscale for black background
                    showscale=True,
                    cmin=0
                    if isinstance(node_color, np.ndarray) and robust_color
                    else None,
                    cmax=1
                    if isinstance(node_color, np.ndarray) and robust_color
                    else None,
                    colorbar=dict(
                        title=dict(
                            text="Attention Weight", font=dict(size=14, color="black")
                        ),
                        tickfont=dict(color="black"),
                        bgcolor="rgba(255, 255, 255, 0.0)",
                        # Add custom tick labels to show the actual range
                        tickvals=[0, 0.5, 1]
                        if isinstance(node_color, np.ndarray) and robust_color
                        else None,
                        ticktext=[
                            f"{color_range[0]:.3f}" if color_range else "0",
                            f"{(color_range[0] + color_range[1]) / 2:.3f}"
                            if color_range
                            else "0.5",
                            f"{color_range[1]:.3f}" if color_range else "1",
                        ]
                        if isinstance(node_color, np.ndarray) and color_range
                        else None,
                    ),
                    line=dict(
                        width=0.5, color="rgba(255, 255, 255, 0.4)"
                    ),  # Subtle white border
                    opacity=0.9,  # Slightly transparent for elegance
                ),
            )

            # Create figure with stunning black theme
            fig = go.Figure(
                data=[edge_trace, node_trace],
                layout=go.Layout(
                    title=dict(
                        text=f"Pooling Attention: {attention_key.replace('_', ' ').title()}",
                        font=dict(size=20, color="black", family="Arial Black"),
                        x=0.5,  # Center the title
                    ),
                    showlegend=False,
                    hovermode="closest",
                    margin=dict(
                        b=30, l=20, r=80, t=60
                    ),  # Extra right margin for colorbar
                    annotations=[
                        dict(
                            text="Node-based Attention Visualization",
                            showarrow=False,
                            xref="paper",
                            yref="paper",
                            x=0.02,
                            y=0.98,
                            xanchor="left",
                            yanchor="top",
                            font=dict(
                                color="rgba(255, 255, 255, 0.6)",
                                size=12,
                                family="Arial",
                            ),
                        )
                    ],
                    xaxis=dict(
                        showgrid=False,
                        zeroline=False,
                        showticklabels=False,
                        showline=False,
                    ),
                    yaxis=dict(
                        showgrid=False,
                        zeroline=False,
                        showticklabels=False,
                        autorange="reversed",
                        showline=False,
                    ),
                    plot_bgcolor="white",  # Beautiful white background
                    paper_bgcolor="white",  # White paper background too
                    font=dict(color="black"),  # Black text throughout
                ),
            )

            # Save plot
            plot_path = (
                output_path
                / f"{attention_key}_{'robust' if robust_color else 'standard'}_plotly.html"
            )
            fig.write_html(str(plot_path))  # type: ignore

            logger.info(f"Saved Plotly node plot to {plot_path}")

            return plot_path

        except Exception as e:
            logger.error(f"Error creating Plotly node plot: {e}")
            return None

    def _create_combined_plots(
        self,
        G: nx.Graph,
        pos: Dict[int, Tuple[float, float]],
        gnn_attentions: Dict[str, torch.Tensor],
        pooling_attentions: Dict[str, torch.Tensor],
        edge_index: np.ndarray[Any, Any],
        output_path: Path,
    ) -> List[Path]:
        """Create combined visualization showing both edge and node attention."""

        created_files: list[Path] = []

        # Take first of each type for combined visualization
        _, gnn_attention = next(iter(gnn_attentions.items()))
        _, pooling_attention = next(iter(pooling_attentions.items()))

        plotly_file = self._create_plotly_combined_plot(
            G, pos, gnn_attention, pooling_attention, edge_index, output_path
        )
        if plotly_file:
            created_files.append(plotly_file)

        return created_files

    def _create_plotly_combined_plot(
        self,
        G: nx.Graph,
        pos: Dict[int, Tuple[float, float]],
        gnn_attention: torch.Tensor,
        pooling_attention: torch.Tensor,
        edge_index: np.ndarray[Any, Any],
        output_path: Path,
    ) -> Optional[Path]:
        """Create Plotly combined subplot with proper GNN attention visualization."""
        try:
            # Create subplots
            fig = make_subplots(
                rows=1,
                cols=2,
                subplot_titles=("GNN Attention (Edges)", "Pooling Attention (Nodes)"),
                specs=[[{"type": "scatter"}, {"type": "scatter"}]],
            )

            # Process GNN attention for edge visualization
            if len(gnn_attention.shape) == 2:
                # Take mean across attention heads for edge colors
                edge_attention_values = cast(
                    np.ndarray[Any, Any],
                    gnn_attention.detach().mean(dim=1).cpu().numpy(),  # type: ignore
                )
            else:
                edge_attention_values = cast(
                    np.ndarray[Any, Any],
                    gnn_attention.detach().cpu().numpy(),  # type: ignore
                )

            # Normalize edge attention to [0, 1] for color mapping
            if len(edge_attention_values) > 0:
                min_edge_att = edge_attention_values.min()
                max_edge_att = edge_attention_values.max()
                if max_edge_att > min_edge_att:
                    norm_edge_att = (edge_attention_values - min_edge_att) / (
                        max_edge_att - min_edge_att
                    )
                else:
                    norm_edge_att = np.ones_like(edge_attention_values) * 0.5
            else:
                norm_edge_att = np.array([])

            # Left subplot: GNN attention with colored edges
            for i, (src, dst) in enumerate(edge_index.T):
                x0, y0 = pos[src]
                x1, y1 = pos[dst]

                # Get color for this edge based on attention
                if i < len(norm_edge_att):
                    attention_val = norm_edge_att[i]
                    width = 0.5 + 3.0 * attention_val  # Width based on attention

                    # Use the same gorgeous color scheme optimized for black background
                    if attention_val < 0.33:
                        # Soft cyan to bright cyan for low attention
                        intensity = attention_val * 3  # 0 to 1
                        r = int(50 + 100 * intensity)  # 50 to 150
                        g = int(200 + 55 * intensity)  # 200 to 255
                        b = int(200 + 55 * intensity)  # 200 to 255
                        color = f"rgb({r}, {g}, {b})"
                    elif attention_val < 0.66:
                        # Cyan to purple transition for medium attention
                        intensity = (attention_val - 0.33) * 3  # 0 to 1
                        r = int(150 + 105 * intensity)  # 150 to 255
                        g = int(255 - 155 * intensity)  # 255 to 100
                        b = int(255 - 55 * intensity)  # 255 to 200
                        color = f"rgb({r}, {g}, {b})"
                    else:
                        # Purple to vibrant magenta for high attention
                        intensity = (attention_val - 0.66) * 3  # 0 to 1
                        r = int(255)  # Constant bright red
                        g = int(100 - 50 * intensity)  # 100 to 50
                        b = int(200 + 55 * intensity)  # 200 to 255
                        color = f"rgb({r}, {g}, {b})"
                else:
                    color = "lightblue"
                    width = 1.0

                fig.add_trace(  # type: ignore
                    go.Scatter(
                        x=[x0, x1],
                        y=[y0, y1],
                        line=dict(width=width, color=color),
                        hoverinfo="text",
                        hovertext=f"Edge {src}-{dst}<br>Attention: {edge_attention_values[i]:.4f}"
                        if i < len(edge_attention_values)
                        else f"Edge {src}-{dst}",
                        mode="lines",
                        showlegend=False,
                    ),
                    row=1,
                    col=1,
                )

            # Node traces for left plot
            node_x = [pos[node][0] for node in G.nodes()]  # type: ignore
            node_y = [pos[node][1] for node in G.nodes()]  # type: ignore

            fig.add_trace(  # type: ignore
                go.Scatter(
                    x=node_x,
                    y=node_y,
                    mode="markers",
                    marker=dict(
                        size=2,  # Ultra-small for elegance
                        color="rgba(220, 220, 220, 0.4)",
                        line=dict(width=0.3, color="rgba(255, 255, 255, 0.6)"),
                        opacity=0.4,
                    ),
                    hoverinfo="text",
                    text=[f"Node {i}" for i in G.nodes()],  # type: ignore
                    showlegend=False,
                ),
                row=1,
                col=1,
            )

            # Right subplot: Pooling attention
            # Add edges in light gray
            edge_x: list[float | None] = []
            edge_y: list[float | None] = []
            for _, (src, dst) in enumerate(edge_index.T):
                x0, y0 = pos[src]
                x1, y1 = pos[dst]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])

            fig.add_trace(  # type: ignore
                go.Scatter(
                    x=edge_x,
                    y=edge_y,
                    line=dict(
                        width=0.8, color="rgba(100, 100, 100, 0.3)"
                    ),  # Subtle edges for black background
                    hoverinfo="none",
                    mode="lines",
                    showlegend=False,
                ),
                row=1,
                col=2,
            )

            # Process pooling attention for node visualization
            # Handle multi-head attention by taking the mean across heads
            if len(pooling_attention.shape) == 2:
                if pooling_attention.shape[0] == 1:
                    # Single head case: [1, num_nodes] -> [num_nodes]
                    node_attention = cast(
                        np.ndarray[Any, Any],
                        pooling_attention.detach().squeeze(0).cpu().numpy(),  # type: ignore
                    )
                else:
                    # Multi-head case: [num_heads, num_nodes] -> [num_nodes] (mean across heads)
                    node_attention = cast(
                        np.ndarray[Any, Any],
                        pooling_attention.mean(dim=0).detach().cpu().numpy(),  # type: ignore
                    )
            else:
                # 1D case: [num_nodes]
                node_attention = cast(
                    np.ndarray[Any, Any],
                    pooling_attention.detach().cpu().numpy(),  # type: ignore
                )

            if len(node_attention) == len(G.nodes()):  # type: ignore
                # Calculate node sizes based on attention
                min_attention = node_attention.min()
                max_attention = node_attention.max()
                if max_attention > min_attention:
                    normalized_attention = (node_attention - min_attention) / (
                        max_attention - min_attention
                    )
                    node_sizes = (
                        1 + 10 * normalized_attention
                    )  # Sizes from 3 to 9 (much smaller and elegant)
                else:
                    node_sizes = (
                        5  # Smaller default size if attention values are identical
                    )

                fig.add_trace(  # type: ignore
                    go.Scatter(
                        x=node_x,
                        y=node_y,
                        mode="markers",
                        marker=dict(
                            size=node_sizes,
                            color=node_attention,
                            colorscale="Plasma",  # Beautiful plasma colorscale for black background
                            showscale=True,
                            colorbar=dict(
                                title=dict(
                                    text="Attention Weight",
                                    font=dict(size=12, color="white"),
                                ),
                                tickfont=dict(color="white", size=10),
                                bgcolor="rgba(0, 0, 0, 0.3)",
                                bordercolor="rgba(255, 255, 255, 0.3)",
                                borderwidth=1,
                            ),
                            line=dict(width=0.5, color="rgba(255, 255, 255, 0.4)"),
                            opacity=0.9,
                        ),
                        hoverinfo="text",
                        text=[
                            f"Node {i}<br>Attention: {node_attention[i]:.4f}"
                            for i in range(len(G.nodes()))  # type: ignore
                        ],
                        showlegend=False,
                    ),
                    row=1,
                    col=2,
                )
            else:
                fig.add_trace(  # type: ignore
                    go.Scatter(
                        x=node_x,
                        y=node_y,
                        mode="markers",
                        marker=dict(
                            size=4,
                            color="rgba(255, 100, 100, 0.7)",  # Soft red with transparency
                            line=dict(width=0.5, color="rgba(255, 255, 255, 0.4)"),
                        ),
                        showlegend=False,
                    ),
                    row=1,
                    col=2,
                )

            # Update layout with gorgeous black theme
            fig.update_layout(  # type: ignore
                title=dict(
                    text="Combined GraphMIL Attention Visualization",
                    font=dict(size=22, color="white", family="Arial Black"),
                    x=0.5,  # Center the title
                ),
                showlegend=False,
                height=700,  # Slightly taller for better presentation
                plot_bgcolor="black",
                paper_bgcolor="black",
                font=dict(color="white"),
                margin=dict(b=40, l=30, r=100, t=80),  # Better margins
                annotations=[
                    dict(
                        text="Edge Attention",
                        xref="paper",
                        yref="paper",
                        x=0.25,
                        y=0.95,
                        showarrow=False,
                        font=dict(color="rgba(255, 255, 255, 0.8)", size=14),
                    ),
                    dict(
                        text="Node Attention",
                        xref="paper",
                        yref="paper",
                        x=0.75,
                        y=0.95,
                        showarrow=False,
                        font=dict(color="rgba(255, 255, 255, 0.8)", size=14),
                    ),
                ],
            )

            # Update subplot axes with elegant styling
            for row in [1]:
                for col in [1, 2]:
                    fig.update_xaxes(  # type: ignore
                        showgrid=False,
                        zeroline=False,
                        showticklabels=False,
                        showline=False,
                        row=row,
                        col=col,
                    )
                    fig.update_yaxes(  # type: ignore
                        showgrid=False,
                        zeroline=False,
                        showticklabels=False,
                        autorange="reversed",
                        showline=False,
                        row=row,
                        col=col,
                    )

            # Save plot
            plot_path = output_path / "combined_attention_plotly.html"
            fig.write_html(str(plot_path))  # type: ignore

            return plot_path

        except Exception as e:
            logger.error(f"Error creating Plotly combined plot: {e}")
            return None

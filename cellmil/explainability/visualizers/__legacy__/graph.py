"""
Graph visualization for GraphMIL attention weights.

This module creates interactive graph plots showing attention weights on edges
and nodes, specifically designed for GraphMIL models.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Any, cast
import json

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
        
        if hasattr(graph_data, 'x') and graph_data.x is not None:
            num_nodes = graph_data.x.shape[0]
        else:
            raise ValueError("Graph data must have node features 'x' to determine number of nodes")

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
                for gnn_pattern in [
                    "gnn_attention",
                    "selected_gnn_layer"
                ]
            )
        }
        pooling_attentions = {
            k: v
            for k, v in attention_result.attention_weights.items()
            if any(
                pool_pattern in k
                for pool_pattern in [
                    "pooling_attention",
                    "random_walk"
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
        if gnn_attentions and pooling_attentions:
            combined_files = self._create_combined_plots(
                G, pos, gnn_attentions, pooling_attentions, edge_index, output_path
            )
            created_files.extend(combined_files)

        # Create summary visualization
        summary_file = self._create_summary_plot(attention_result, output_path)
        if summary_file:
            created_files.append(summary_file)

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
            # Matplotlib version
            matplotlib_file = self._create_matplotlib_edge_plot(
                G, pos, attention_weights, edge_index, attention_key, output_path
            )
            if matplotlib_file:
                created_files.append(matplotlib_file)

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
            # Matplotlib version
            matplotlib_file = self._create_matplotlib_node_plot(
                G, pos, attention_weights, attention_key, output_path
            )
            if matplotlib_file:
                created_files.append(matplotlib_file)

            plotly_file = self._create_plotly_node_plot(
                G, pos, attention_weights, attention_key, output_path
            )
            if plotly_file:
                created_files.append(plotly_file)

        return created_files

    def _create_matplotlib_edge_plot(
        self,
        G: nx.Graph,
        pos: Dict[int, Tuple[float, float]],
        attention_weights: torch.Tensor,
        edge_index: np.ndarray[Any, Any],
        attention_key: str,
        output_path: Path,
    ) -> Optional[Path]:
        """Create matplotlib plot for edge-based attention."""

        try:
            # Process attention weights
            if len(attention_weights.shape) == 2:
                # Multiple heads - take mean
                edge_attention = cast(
                    np.ndarray[Any, Any],
                    attention_weights.detach().mean(dim=1).cpu().numpy(),  # type: ignore
                )
            else:
                edge_attention = cast(
                    np.ndarray[Any, Any],
                    attention_weights.detach().cpu().numpy(),  # type: ignore
                )

            # Normalize attention for alpha values
            if len(edge_attention) > 0:
                edge_attention_norm = (edge_attention - edge_attention.min()) / (
                    edge_attention.max() - edge_attention.min() + 1e-8
                )
            else:
                edge_attention_norm = edge_attention

            # Create plot
            plt.figure(figsize=(12, 10))  # type: ignore

            # Draw edges with attention-based coloring and width
            for i, (src, dst) in enumerate(edge_index.T):
                if i < len(edge_attention_norm):
                    attention_val = edge_attention_norm[i]

                    # Use attention for both color intensity and line width
                    color_intensity = attention_val
                    line_width = 0.5 + attention_val * 3.0  # Width from 0.5 to 3.5
                    alpha = 0.4 + attention_val * 0.6  # Alpha from 0.4 to 1.0

                    x_coords = [pos[src][0], pos[dst][0]]
                    y_coords = [pos[src][1], pos[dst][1]]

                    plt.plot(  # type: ignore
                        x_coords,
                        y_coords,
                        color=plt.cm.Blues(color_intensity),  # type: ignore
                        alpha=alpha,
                        linewidth=line_width,
                    )

            # Draw nodes very small and transparent to emphasize edges
            node_x = [pos[node][0] for node in G.nodes()]  # type: ignore
            node_y = [pos[node][1] for node in G.nodes()]  # type: ignore
            plt.scatter(  # type: ignore
                node_x,
                node_y,
                c="gray",
                s=8,  # Much smaller size
                alpha=0.3,  # Much more transparent
                zorder=1,  # Lower z-order so edges are on top
                edgecolors="white",
                linewidth=0.2,
            )

            plt.title(f"GNN Attention: {attention_key.replace('_', ' ').title()}")  # type: ignore
            plt.axis("equal")  # type: ignore
            plt.axis("off")  # type: ignore
            plt.gca().invert_yaxis()  # type: ignore  # Invert y-axis to match image coordinates

            # Add colorbar - fix the error by providing explicit axis
            if len(edge_attention) > 0:
                sm = plt.cm.ScalarMappable(
                    cmap=plt.cm.Blues,  # type: ignore
                    norm=plt.Normalize(  # type: ignore
                        vmin=edge_attention.min(), vmax=edge_attention.max()
                    ),
                )
                sm.set_array(edge_attention)
                plt.colorbar(  # type: ignore
                    sm, ax=plt.gca(), label="Edge Attention Weight", shrink=0.8
                )

            # Save plot
            plot_path = output_path / f"{attention_key}_matplotlib.png"
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")  # type: ignore
            plt.close()

            return plot_path

        except Exception as e:
            logger.error(f"Error creating matplotlib edge plot: {e}")
            return None

    def _create_matplotlib_node_plot(
        self,
        G: nx.Graph,
        pos: Dict[int, Tuple[float, float]],
        attention_weights: torch.Tensor,
        attention_key: str,
        output_path: Path,
    ) -> Optional[Path]:
        """Create matplotlib plot for node-based attention."""

        try:
            # Process attention weights
            logger.info(
                f"Matplotlib - Raw attention weights shape: {attention_weights.shape}"
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
                        "Matplotlib - Applied squeeze(0) to single-head attention weights"
                    )
                else:
                    # Multi-head case: [num_heads, num_nodes] -> [num_nodes] (mean across heads)
                    node_attention = cast(
                        np.ndarray[Any, Any],
                        attention_weights.mean(dim=0).detach().cpu().numpy(),  # type: ignore
                    )
                    logger.info(
                        f"Matplotlib - Averaged across {attention_weights.shape[0]} attention heads"
                    )
            else:
                # 1D case: [num_nodes]
                node_attention = cast(
                    np.ndarray[Any, Any],
                    attention_weights.detach().cpu().numpy(),  # type: ignore
                )
                logger.info("Matplotlib - Used 1D attention weights directly")

            # Create plot
            plt.figure(figsize=(12, 10))  # type: ignore

            # Draw edges (light gray)
            nx.draw_networkx_edges(G, pos, alpha=0.2, width=1, edge_color="gray")  # type: ignore

            # Draw nodes with attention-based coloring
            node_x = [pos[node][0] for node in G.nodes()]  # type: ignore
            node_y = [pos[node][1] for node in G.nodes()]  # type: ignore

            # Ensure we have attention values for all nodes
            logger.info(
                f"Node attention shape: {node_attention.shape}, Graph nodes: {len(G.nodes())}"  # type: ignore
            )
            logger.info(
                f"Node attention values range: [{node_attention.min():.4f}, {node_attention.max():.4f}]"
            )

            if len(node_attention) == len(G.nodes()):  # type: ignore
                scatter = plt.scatter(  # type: ignore
                    node_x,
                    node_y,
                    c=node_attention,
                    s=100,
                    cmap="Reds",
                    alpha=0.8,
                    zorder=5,
                )
                plt.colorbar(scatter, label="Attention Weight", shrink=0.8)  # type: ignore
                logger.info("Successfully applied attention-based node coloring")
            else:
                plt.scatter(node_x, node_y, c="red", s=50, alpha=0.7, zorder=5)  # type: ignore
                logger.warning(
                    f"Attention shape mismatch: {len(node_attention)} vs {len(G.nodes())} nodes"  # type: ignore
                )

            plt.title(f"Pooling Attention: {attention_key.replace('_', ' ').title()}")  # type: ignore
            plt.axis("equal")  # type: ignore
            plt.axis("off")  # type: ignore
            plt.gca().invert_yaxis()  # type: ignore  # Invert y-axis to match image coordinates

            # Save plot
            plot_path = output_path / f"{attention_key}_matplotlib.png"
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")  # type: ignore
            plt.close()

            return plot_path

        except Exception as e:
            logger.error(f"Error creating matplotlib node plot: {e}")
            return None

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

                    # Create color using a blue-to-orange palette for better visibility
                    # Low attention: light blue, High attention: orange/red
                    if color_intensity < 0.5:
                        # Light blue to medium blue for low attention
                        blue_val = int(150 + 105 * (color_intensity * 2))  # 150 to 255
                        edge_color = f"rgb(100, 150, {blue_val})"
                    else:
                        # Blue to orange/red for high attention
                        adjusted_intensity = (color_intensity - 0.5) * 2  # 0 to 1
                        red_val = int(100 + 155 * adjusted_intensity)  # 100 to 255
                        blue_val = int(255 - 155 * adjusted_intensity)  # 255 to 100
                        edge_color = f"rgb({red_val}, 100, {blue_val})"

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

            # Node trace - very small and transparent to emphasize edges
            node_x = [pos[node][0] for node in G.nodes()]  # type: ignore
            node_y = [pos[node][1] for node in G.nodes()]  # type: ignore

            node_trace = go.Scatter(
                x=node_x,
                y=node_y,
                mode="markers",
                hoverinfo="text",
                text=[f"Node {i}" for i in G.nodes()],  # type: ignore
                marker=dict(
                    size=3,  # Much smaller size
                    color="gray",  # Neutral color
                    line=dict(width=0.5, color="white"),
                    opacity=0.2,  # Much more transparent
                ),
                showlegend=False,
            )

            # Create figure
            all_traces = edge_traces + [node_trace]
            fig = go.Figure(
                data=all_traces,
                layout=go.Layout(
                    title=dict(
                        text=f"GNN Attention: {attention_key.replace('_', ' ').title()}",
                        font=dict(size=16),
                    ),
                    showlegend=False,
                    hovermode="closest",
                    margin=dict(b=20, l=5, r=5, t=40),
                    annotations=[
                        dict(
                            text="Interactive Graph Visualization",
                            showarrow=False,
                            xref="paper",
                            yref="paper",
                            x=0.005,
                            y=-0.002,
                            xanchor="left",
                            yanchor="bottom",
                            font=dict(color="gray", size=12),
                        )
                    ],
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(
                        showgrid=False,
                        zeroline=False,
                        showticklabels=False,
                        autorange="reversed",
                    ),
                    plot_bgcolor="white",
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
                line=dict(width=1, color="lightgray"),
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

            # Calculate node sizes based on attention (if we have attention values)
            if isinstance(node_color, np.ndarray):
                # Normalize attention values to reasonable size range (5-25 pixels)
                min_attention = node_attention.min()
                max_attention = node_attention.max()
                if max_attention > min_attention:
                    normalized_attention = (node_attention - min_attention) / (
                        max_attention - min_attention
                    )
                    node_sizes = 8 + 17 * normalized_attention  # Sizes from 8 to 25
                else:
                    node_sizes = 15  # All same size if attention values are identical
            else:
                node_sizes = 15  # Default size for fallback case

            node_trace = go.Scatter(
                x=node_x,
                y=node_y,
                mode="markers",
                hoverinfo="text",
                text=hover_text,
                marker=dict(
                    size=node_sizes,
                    color=node_color,
                    colorscale="Reds",
                    showscale=True,
                    colorbar=dict(title="Attention Weight"),
                    line=dict(width=0),  # Remove border for better color visibility
                ),
            )

            # Create figure
            fig = go.Figure(
                data=[edge_trace, node_trace],
                layout=go.Layout(
                    title=dict(
                        text=f"Pooling Attention: {attention_key.replace('_', ' ').title()}",
                        font=dict(size=16),
                    ),
                    showlegend=False,
                    hovermode="closest",
                    margin=dict(b=20, l=5, r=5, t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(
                        showgrid=False,
                        zeroline=False,
                        showticklabels=False,
                        autorange="reversed",
                    ),
                    plot_bgcolor="white",
                ),
            )

            # Save plot
            plot_path = output_path / f"{attention_key}_plotly.html"
            fig.write_html(str(plot_path))  # type: ignore

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

        # Matplotlib combined plot
        matplotlib_file = self._create_matplotlib_combined_plot(
            G, pos, gnn_attention, pooling_attention, edge_index, output_path
        )
        if matplotlib_file:
            created_files.append(matplotlib_file)

        plotly_file = self._create_plotly_combined_plot(
            G, pos, gnn_attention, pooling_attention, edge_index, output_path
        )
        if plotly_file:
            created_files.append(plotly_file)

        return created_files

    def _create_matplotlib_combined_plot(
        self,
        G: nx.Graph,
        pos: Dict[int, Tuple[float, float]],
        gnn_attention: torch.Tensor,
        pooling_attention: torch.Tensor,
        edge_index: np.ndarray[Any, Any],
        output_path: Path,
    ) -> Optional[Path]:
        """Create matplotlib combined plot."""

        try:
            _, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))  # type: ignore

            # Left plot: GNN attention (edges)
            ax1.set_title("GNN Attention (Edges)")

            if len(gnn_attention.shape) == 2:
                edge_attention = cast(
                    np.ndarray[Any, Any],
                    gnn_attention.mean(dim=1).detach().cpu().numpy(),  # type: ignore
                )
            else:
                edge_attention = cast(
                    np.ndarray[Any, Any],
                    gnn_attention.detach().cpu().numpy(),  # type: ignore
                )

            if len(edge_attention) > 0:
                edge_attention_norm = (edge_attention - edge_attention.min()) / (
                    edge_attention.max() - edge_attention.min() + 1e-8
                )

                for i, (src, dst) in enumerate(edge_index.T):
                    if i < len(edge_attention_norm):
                        alpha = (
                            self.config.edge_alpha_min
                            + (self.config.edge_alpha_max - self.config.edge_alpha_min)
                            * edge_attention_norm[i]
                        )

                        x_coords = [pos[src][0], pos[dst][0]]
                        y_coords = [pos[src][1], pos[dst][1]]

                        ax1.plot(x_coords, y_coords, "b-", alpha=alpha, linewidth=2)

            node_x = [pos[node][0] for node in G.nodes()]  # type: ignore
            node_y = [pos[node][1] for node in G.nodes()]  # type: ignore
            ax1.scatter(node_x, node_y, c="gray", s=8, alpha=0.3, zorder=1)
            ax1.set_aspect("equal")
            ax1.axis("off")
            ax1.invert_yaxis()  # Invert y-axis to match image coordinates

            # Right plot: Pooling attention (nodes)
            ax2.set_title("Pooling Attention (Nodes)")

            # Draw edges lightly
            for edge in G.edges():  # type: ignore
                x_coords = [pos[edge[0]][0], pos[edge[1]][0]]
                y_coords = [pos[edge[0]][1], pos[edge[1]][1]]
                ax2.plot(x_coords, y_coords, "gray", alpha=0.2, linewidth=1)

            # Draw nodes with attention coloring
            # Handle multi-head attention by taking the mean across heads
            if len(pooling_attention.shape) == 2:
                if pooling_attention.shape[0] == 1:
                    # Single head case: [1, num_nodes] -> [num_nodes]
                    node_attention = cast(
                        np.ndarray[Any, Any],
                        pooling_attention.squeeze(0).detach().cpu().numpy(),  # type: ignore
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
                scatter = ax2.scatter(
                    node_x,
                    node_y,
                    c=node_attention,
                    s=60,
                    cmap="Reds",
                    alpha=0.8,
                    zorder=5,
                )
                plt.colorbar(scatter, ax=ax2, label="Attention Weight", shrink=0.8)  # type: ignore
            else:
                ax2.scatter(node_x, node_y, c="red", s=30, alpha=0.7, zorder=5)

            ax2.set_aspect("equal")
            ax2.axis("off")
            ax2.invert_yaxis()  # Invert y-axis to match image coordinates

            plt.tight_layout()

            # Save plot
            plot_path = output_path / "combined_attention_matplotlib.png"
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")  # type: ignore
            plt.close()

            return plot_path

        except Exception as e:
            logger.error(f"Error creating matplotlib combined plot: {e}")
            return None

    # TODO: This is not using the GNN attention
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

                    # Use the same improved color scheme as single plot
                    if attention_val < 0.5:
                        # Light blue to medium blue for low attention
                        blue_val = int(150 + 105 * (attention_val * 2))
                        color = f"rgb(100, 150, {blue_val})"
                    else:
                        # Blue to orange/red for high attention
                        adjusted_intensity = (attention_val - 0.5) * 2
                        red_val = int(100 + 155 * adjusted_intensity)
                        blue_val = int(255 - 155 * adjusted_intensity)
                        color = f"rgb({red_val}, 100, {blue_val})"
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
                        size=3,
                        color="gray",
                        line=dict(width=0.5, color="white"),
                        opacity=0.2,
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
                    line=dict(width=1, color="lightgray"),
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
                        6 + 14 * normalized_attention
                    )  # Sizes from 6 to 20 (smaller for combined plot)
                else:
                    node_sizes = 12  # All same size if attention values are identical

                fig.add_trace(  # type: ignore
                    go.Scatter(
                        x=node_x,
                        y=node_y,
                        mode="markers",
                        marker=dict(
                            size=node_sizes,
                            color=node_attention,
                            colorscale="Reds",
                            showscale=True,
                            colorbar=dict(title="Attention Weight"),
                            line=dict(
                                width=0
                            ),  # Remove border for better color visibility
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
                        marker=dict(size=8, color="red"),
                        showlegend=False,
                    ),
                    row=1,
                    col=2,
                )

            # Update layout
            fig.update_layout(  # type: ignore
                title=dict(
                    text="Combined GraphMIL Attention Visualization", font=dict(size=16)
                ),
                showlegend=False,
                height=600,
            )

            # Update subplot axes
            for row in [1]:
                for col in [1, 2]:
                    fig.update_xaxes(  # type: ignore
                        showgrid=False,
                        zeroline=False,
                        showticklabels=False,
                        row=row,
                        col=col,
                    )
                    fig.update_yaxes(  # type: ignore
                        showgrid=False,
                        zeroline=False,
                        showticklabels=False,
                        autorange="reversed",
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

    def _create_summary_plot(
        self, attention_result: AttentionResult, output_path: Path
    ) -> Optional[Path]:
        """Create a summary visualization showing attention statistics."""

        try:
            # Collect attention statistics
            stats: dict[str, dict[str, Any]] = {}
            for key, attention in attention_result.attention_weights.items():
                attention_np = cast(
                    np.ndarray[Any, Any],
                    attention.detach().cpu().numpy().flatten(),  # type: ignore
                )
                stats[key] = {
                    "mean": np.mean(attention_np),
                    "std": np.std(attention_np),
                    "min": np.min(attention_np),
                    "max": np.max(attention_np),
                    "entropy": -1 * np.sum(attention_np * np.log(attention_np + 1e-8)),
                }

            # Create summary plot
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))  # type: ignore

            keys = list(stats.keys())

            # Mean attention
            means = [stats[key]["mean"] for key in keys]
            ax1.bar(range(len(keys)), means)
            ax1.set_title("Mean Attention Weights")
            ax1.set_xticks(range(len(keys)))
            ax1.set_xticklabels([k.replace("_", "\n") for k in keys], rotation=45)

            # Standard deviation
            stds = [stats[key]["std"] for key in keys]
            ax2.bar(range(len(keys)), stds)
            ax2.set_title("Attention Weight Standard Deviation")
            ax2.set_xticks(range(len(keys)))
            ax2.set_xticklabels([k.replace("_", "\n") for k in keys], rotation=45)

            # Min/Max ranges
            mins = [stats[key]["min"] for key in keys]
            maxs = [stats[key]["max"] for key in keys]
            x_pos = range(len(keys))
            ax3.bar(x_pos, maxs, label="Max", alpha=0.7)
            ax3.bar(x_pos, mins, label="Min", alpha=0.7)
            ax3.set_title("Attention Weight Ranges")
            ax3.set_xticks(range(len(keys)))
            ax3.set_xticklabels([k.replace("_", "\n") for k in keys], rotation=45)
            ax3.legend()

            # Entropy
            entropies = [stats[key]["entropy"] for key in keys]
            ax4.bar(range(len(keys)), entropies)
            ax4.set_title("Attention Entropy")
            ax4.set_xticks(range(len(keys)))
            ax4.set_xticklabels([k.replace("_", "\n") for k in keys], rotation=45)

            plt.tight_layout()

            # Save plot
            plot_path = output_path / "attention_summary.png"
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")  # type: ignore
            plt.close()

            # Also save statistics as JSON
            stats_path = output_path / "attention_statistics.json"
            with open(stats_path, "w") as f:
                json.dump(stats, f, indent=4, default=str)

            return plot_path

        except Exception as e:
            logger.error(f"Error creating summary plot: {e}")
            return None

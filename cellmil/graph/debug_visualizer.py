"""
Interactive debug visualizer for graph creation hyperparameter tuning.

This module provides an interactive Dash application for visualizing and tuning
graph creation hyperparameters in real-time with a sampled subset of cells.
"""

import torch
import numpy as np
from typing import Any, Dict, List, cast
import plotly.graph_objs as go # type: ignore
from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc # type: ignore
from cellmil.utils import logger
from cellmil.interfaces.GraphCreatorConfig import GraphCreatorType
from .creator import (
    KNNEdgeCreator,
    RadiusEdgeCreator,
    DelaunayEdgeCreator,
    DilateEdgeCreator,
    SimilarityEdgeCreator,
)


class GraphDebugVisualizer:
    """Interactive visualizer for debugging graph creation with different hyperparameters."""

    def __init__(
        self,
        cells: List[Dict[str, Any]],
        method: GraphCreatorType,
        device: str = "cpu",
    ):
        """
        Initialize the debug visualizer.

        Args:
            cells: List of cell dictionaries with centroid and other features
            method: Graph creation method to use
            device: Computing device ('cpu' or 'cuda:X')
        """
        self.cells = cells
        self.method = method
        self.device = device
        self.app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

        # Extract positions
        centroids = np.array([cell["centroid"] for cell in cells])
        self.positions = cast(torch.Tensor, torch.from_numpy(centroids).float().to(device)) # type: ignore

        logger.info(
            f"Debug visualizer initialized with {len(cells)} cells using {method} method"
        )

        self._setup_layout()
        self._setup_callbacks()

    def _setup_layout(self):
        """Setup the Dash application layout with controls and visualization."""

        # Create control panel based on method
        controls = self._create_controls()

        self.app.layout = dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            html.H1(
                                f"Graph Creator Debug Mode - {self.method.upper()}",
                                className="text-center mb-4",
                            )
                        )
                    ]
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.H4("Hyperparameters", className="mb-3"),
                                html.Div(controls),
                                html.Hr(),
                                html.Div(id="graph-stats", className="mt-3"),
                            ],
                            width=3,
                        ),
                        dbc.Col(
                            [
                                dcc.Loading(
                                    id="loading",
                                    type="default",
                                    children=dcc.Graph(
                                        id="graph-plot",
                                        style={"height": "85vh"},
                                        config={"displayModeBar": True},
                                    ),
                                )
                            ],
                            width=9,
                        ),
                    ]
                ),
            ],
            fluid=True,
            style={"padding": "20px"},
        )

    def _create_controls(self) -> List[html.Div]:
        """Create control widgets based on the graph creation method."""
        controls: list[Any] = []

        if self.method == GraphCreatorType.knn:
            controls.extend(
                [
                    html.Label("K (Number of Neighbors):", className="fw-bold"),
                    dbc.Row(
                        [
                            dbc.Col(
                                dcc.Slider(
                                    id="k-slider",
                                    min=1,
                                    max=50,
                                    step=1,
                                    value=5,
                                    marks={i: str(i) for i in [1, 5, 10, 20, 30, 40, 50]},
                                    tooltip={"placement": "bottom", "always_visible": True},
                                ),
                                width=8,
                            ),
                            dbc.Col(
                                dcc.Input(
                                    id="k-input",
                                    type="number",
                                    min=1,
                                    max=50,
                                    step=1,
                                    value=5,
                                    style={"width": "100%"},
                                ),
                                width=4,
                            ),
                        ],
                        className="mb-3",
                    ),
                ]
            )

        elif self.method == GraphCreatorType.radius:
            controls.extend(
                [
                    html.Label("Radius (pixels):", className="fw-bold"),
                    dbc.Row(
                        [
                            dbc.Col(
                                dcc.Slider(
                                    id="radius-slider",
                                    min=10,
                                    max=500,
                                    step=10,
                                    value=100,
                                    marks={i: str(i) for i in [10, 100, 200, 300, 400, 500]},
                                    tooltip={"placement": "bottom", "always_visible": True},
                                ),
                                width=8,
                            ),
                            dbc.Col(
                                dcc.Input(
                                    id="radius-input",
                                    type="number",
                                    min=10,
                                    max=500,
                                    step=10,
                                    value=100,
                                    style={"width": "100%"},
                                ),
                                width=4,
                            ),
                        ],
                        className="mb-3",
                    ),
                ]
            )

        elif self.method == GraphCreatorType.delaunay_radius:
            controls.extend(
                [
                    html.Label("Limit Radius (pixels):", className="fw-bold"),
                    dbc.Row(
                        [
                            dbc.Col(
                                dcc.Slider(
                                    id="limit-radius-slider",
                                    min=10,
                                    max=500,
                                    step=10,
                                    value=150,
                                    marks={i: str(i) for i in [10, 100, 200, 300, 400, 500]},
                                    tooltip={"placement": "bottom", "always_visible": True},
                                ),
                                width=8,
                            ),
                            dbc.Col(
                                dcc.Input(
                                    id="limit-radius-input",
                                    type="number",
                                    min=10,
                                    max=500,
                                    step=10,
                                    value=150,
                                    style={"width": "100%"},
                                ),
                                width=4,
                            ),
                        ],
                        className="mb-3",
                    ),
                ]
            )

        elif self.method == GraphCreatorType.dilate:
            controls.extend(
                [
                    html.Label("Dilation (pixels):", className="fw-bold"),
                    dbc.Row(
                        [
                            dbc.Col(
                                dcc.Slider(
                                    id="dilation-slider",
                                    min=1,
                                    max=50,
                                    step=1,
                                    value=10,
                                    marks={i: str(i) for i in [1, 5, 10, 20, 30, 40, 50]},
                                    tooltip={"placement": "bottom", "always_visible": True},
                                ),
                                width=8,
                            ),
                            dbc.Col(
                                dcc.Input(
                                    id="dilation-input",
                                    type="number",
                                    min=1,
                                    max=50,
                                    step=1,
                                    value=10,
                                    style={"width": "100%"},
                                ),
                                width=4,
                            ),
                        ],
                        className="mb-3",
                    ),
                ]
            )

        elif self.method == GraphCreatorType.similarity:
            # Check if cells have features
            has_features = "features" in self.cells[0] if self.cells else False

            if not has_features:
                controls.append(
                    dbc.Alert(
                        "Warning: Cells don't have morphological features loaded. "
                        "Similarity method requires features.",
                        color="warning",
                    )
                )

            controls.extend(
                [
                    html.Label("Similarity Threshold / K:", className="fw-bold"),
                    dbc.Row(
                        [
                            dbc.Col(
                                dcc.Slider(
                                    id="similarity-threshold-slider",
                                    min=0.1,
                                    max=20,
                                    step=0.01,
                                    value=0.5,
                                    marks={
                                        0.1: "0.1",
                                        0.5: "0.5",
                                        1.0: "1",
                                        5: "5",
                                        10: "10",
                                        15: "15",
                                        20: "20",
                                    },
                                    tooltip={"placement": "bottom", "always_visible": True},
                                ),
                                width=8,
                            ),
                            dbc.Col(
                                dcc.Input(
                                    id="similarity-threshold-input",
                                    type="number",
                                    min=0.1,
                                    max=20,
                                    step=0.01,
                                    value=0.5,
                                    style={"width": "100%"},
                                ),
                                width=4,
                            ),
                        ],
                        className="mb-2",
                    ),
                    html.Small(
                        "Values < 1: threshold mode, Values ≥ 1: KNN mode",
                        className="text-muted",
                    ),
                    html.Br(),
                    html.Br(),
                    html.Label("Distance Sigma (for spatial distance):", className="fw-bold"),
                    dbc.Row(
                        [
                            dbc.Col(
                                dcc.Slider(
                                    id="distance-sigma-slider",
                                    min=10,
                                    max=2000,
                                    step=1,
                                    value=200,
                                    marks={i: str(i) for i in [20, 100, 200, 300, 400, 500, 1000, 1500, 2000]},
                                    tooltip={"placement": "bottom", "always_visible": True},
                                ),
                                width=8,
                            ),
                            dbc.Col(
                                dcc.Input(
                                    id="distance-sigma-input",
                                    type="number",
                                    min=10,
                                    max=2000,
                                    step=1,
                                    value=200,
                                    style={"width": "100%"},
                                ),
                                width=4,
                            ),
                        ],
                        className="mb-3",
                    ),
                    html.Label("Alpha (similarity vs distance):", className="fw-bold"),
                    dbc.Row(
                        [
                            dbc.Col(
                                dcc.Slider(
                                    id="alpha-slider",
                                    min=0.0,
                                    max=1.0,
                                    step=0.01,
                                    value=0.5,
                                    marks={i / 10: str(i / 10) for i in range(0, 11, 2)},
                                    tooltip={"placement": "bottom", "always_visible": True},
                                ),
                                width=8,
                            ),
                            dbc.Col(
                                dcc.Input(
                                    id="alpha-input",
                                    type="number",
                                    min=0.0,
                                    max=1.0,
                                    step=0.01,
                                    value=0.5,
                                    style={"width": "100%"},
                                ),
                                width=4,
                            ),
                        ],
                        className="mb-2",
                    ),
                    html.Small(
                        "0: distance only, 1: similarity only", className="text-muted"
                    ),
                    html.Br(),
                    html.Br(),
                    html.Label("Combination Method:", className="fw-bold"),
                    dcc.RadioItems(
                        id="combination-method-radio",
                        options=[
                            {"label": " Additive", "value": "additive"},
                            {"label": " Multiplicative", "value": "multiplicative"},
                        ],
                        value="additive",
                        className="mb-2",
                        labelStyle={"display": "block", "margin": "5px 0"},
                    ),
                    html.Small(
                        "Additive: α·sim + (1-α)·dist | Multiplicative: sim^α · dist^(1-α)",
                        className="text-muted",
                    ),
                    html.Br(),
                    html.Br(),
                    html.Label("Distance Similarity Metric:", className="fw-bold"),
                    dcc.Dropdown(
                        id="distance-metric-dropdown",
                        options=[
                            {"label": "Gaussian (exp(-d²/(2σ²)))", "value": "gaussian"},
                            {"label": "Laplacian (exp(-d/σ))", "value": "laplacian"},
                            {"label": "Inverse (1/(1+d))", "value": "inverse"},
                            {"label": "Inverse Square (1/(1+d²))", "value": "inverse_square"},
                        ],
                        value="gaussian",
                        className="mb-2",
                        clearable=False,
                    ),
                    html.Br(),
                    html.Label("Feature Similarity Metric:", className="fw-bold"),
                    dcc.Dropdown(
                        id="feature-metric-dropdown",
                        options=[
                            {"label": "Cosine Similarity", "value": "cosine"},
                            {"label": "Pearson Correlation", "value": "correlation"},
                            {"label": "Euclidean (exp(-d))", "value": "euclidean"},
                            {"label": "Gaussian (exp(-d²/(2σ_f²)))", "value": "gaussian"},
                        ],
                        value="cosine",
                        className="mb-2",
                        clearable=False,
                    ),
                    html.Br(),
                    html.Label("Feature Sigma (for Gaussian):", className="fw-bold"),
                    dbc.Row(
                        [
                            dbc.Col(
                                dcc.Slider(
                                    id="feature-sigma-slider",
                                    min=0.1,
                                    max=10,
                                    step=0.1,
                                    value=1.0,
                                    marks={i: str(i) for i in [0.1, 1, 2, 5, 10]}, # type: ignore
                                    tooltip={"placement": "bottom", "always_visible": True},
                                ),
                                width=8,
                            ),
                            dbc.Col(
                                dcc.Input(
                                    id="feature-sigma-input",
                                    type="number",
                                    min=0.1,
                                    max=10,
                                    step=0.1,
                                    value=1.0,
                                    style={"width": "100%"},
                                ),
                                width=4,
                            ),
                        ],
                        className="mb-3",
                    ),
                    html.Small(
                        "Only used when Feature Metric is Gaussian", className="text-muted"
                    ),
                    html.Br(),
                    html.Br(),
                ]
            )

        # Add update button
        controls.append(
            dbc.Button(
                "Update Graph",
                id="update-button",
                color="primary",
                className="w-100",
                size="lg",
            )
        )

        return controls

    def _setup_callbacks(self):
        """Setup Dash callbacks for interactive updates."""

        # Setup sync callbacks between sliders and inputs
        self._setup_sync_callbacks()

        # Determine which inputs to use based on method
        inputs = [Input("update-button", "n_clicks")]
        states: list[State] = []

        if self.method == GraphCreatorType.knn:
            states.append(State("k-slider", "value"))
        elif self.method == GraphCreatorType.radius:
            states.append(State("radius-slider", "value"))
        elif self.method == GraphCreatorType.delaunay_radius:
            states.append(State("limit-radius-slider", "value"))
        elif self.method == GraphCreatorType.dilate:
            states.append(State("dilation-slider", "value"))
        elif self.method == GraphCreatorType.similarity:
            states.extend(
                [
                    State("similarity-threshold-slider", "value"),
                    State("distance-sigma-slider", "value"),
                    State("alpha-slider", "value"),
                    State("combination-method-radio", "value"),
                    State("distance-metric-dropdown", "value"),
                    State("feature-metric-dropdown", "value"),
                    State("feature-sigma-slider", "value"),
                ]
            )

        @self.app.callback( # type: ignore
            [Output("graph-plot", "figure"), Output("graph-stats", "children")],
            inputs,
            states,
            prevent_initial_call=False,
        )
        def update_graph(n_clicks: int, *params: Any): # type: ignore
            """Update the graph visualization based on parameters."""
            try:
                # Create edge creator with current parameters
                edge_creator = self._create_edge_creator(*params)

                # Create edges
                edge_indices, edge_features = edge_creator.create_edges(
                    self.positions, self.cells
                )

                # Create visualization
                fig = self._create_figure(edge_indices)

                # Create statistics
                stats = self._create_stats(edge_indices, edge_features, *params)

                return fig, stats

            except Exception as e:
                logger.error(f"Error updating graph: {e}")
                import traceback

                traceback.print_exc()

                # Return empty figure and error message
                return go.Figure(), html.Div(
                    [
                        dbc.Alert(
                            f"Error creating graph: {str(e)}", color="danger"
                        )
                    ]
                )

    def _setup_sync_callbacks(self):
        """Setup callbacks to sync sliders and input fields."""
        if self.method == GraphCreatorType.knn:
            # Sync K slider and input
            @self.app.callback( # type: ignore
                Output("k-slider", "value"),
                Input("k-input", "value"),
                prevent_initial_call=True,
            )
            def sync_k_slider(value: int | None): # type: ignore
                return value if value is not None else 5

            @self.app.callback( # type: ignore
                Output("k-input", "value"),
                Input("k-slider", "value"),
                prevent_initial_call=True,
            )
            def sync_k_input(value: int | None): # type: ignore
                return value

        elif self.method == GraphCreatorType.radius:
            # Sync radius slider and input
            @self.app.callback( # type: ignore
                Output("radius-slider", "value"),
                Input("radius-input", "value"),
                prevent_initial_call=True,
            )
            def sync_radius_slider(value: int | None): # type: ignore
                return value if value is not None else 100

            @self.app.callback( # type: ignore
                Output("radius-input", "value"),
                Input("radius-slider", "value"),
                prevent_initial_call=True,
            )
            def sync_radius_input(value: int | None): # type: ignore
                return value

        elif self.method == GraphCreatorType.delaunay_radius:
            # Sync limit radius slider and input
            @self.app.callback( # type: ignore
                Output("limit-radius-slider", "value"),
                Input("limit-radius-input", "value"),
                prevent_initial_call=True,
            )
            def sync_limit_radius_slider(value: int | None): # type: ignore
                return value if value is not None else 150

            @self.app.callback( # type: ignore
                Output("limit-radius-input", "value"),
                Input("limit-radius-slider", "value"),
                prevent_initial_call=True,
            )
            def sync_limit_radius_input(value: int | None): # type: ignore
                return value

        elif self.method == GraphCreatorType.dilate:
            # Sync dilation slider and input
            @self.app.callback( # type: ignore
                Output("dilation-slider", "value"),
                Input("dilation-input", "value"),
                prevent_initial_call=True,
            )
            def sync_dilation_slider(value: int | None): # type: ignore
                return value if value is not None else 10

            @self.app.callback( # type: ignore
                Output("dilation-input", "value"),
                Input("dilation-slider", "value"),
                prevent_initial_call=True,
            )
            def sync_dilation_input(value: int | None): # type: ignore
                return value

        elif self.method == GraphCreatorType.similarity:
            # Sync similarity threshold slider and input
            @self.app.callback( # type: ignore
                Output("similarity-threshold-slider", "value"),
                Input("similarity-threshold-input", "value"),
                prevent_initial_call=True,
            )
            def sync_threshold_slider(value: float | None): # type: ignore
                return value if value is not None else 0.5

            @self.app.callback( # type: ignore
                Output("similarity-threshold-input", "value"),
                Input("similarity-threshold-slider", "value"),
                prevent_initial_call=True,
            )
            def sync_threshold_input(value: float | None): # type: ignore
                return value

            # Sync distance sigma slider and input
            @self.app.callback( # type: ignore
                Output("distance-sigma-slider", "value"),
                Input("distance-sigma-input", "value"),
                prevent_initial_call=True,
            )
            def sync_distance_sigma_slider(value: float | None): # type: ignore
                return value if value is not None else 200

            @self.app.callback( # type: ignore
                Output("distance-sigma-input", "value"),
                Input("distance-sigma-slider", "value"),
                prevent_initial_call=True,
            )
            def sync_distance_sigma_input(value: float | None): # type: ignore
                return value

            # Sync alpha slider and input
            @self.app.callback( # type: ignore
                Output("alpha-slider", "value"),
                Input("alpha-input", "value"),
                prevent_initial_call=True,
            )
            def sync_alpha_slider(value: float | None): # type: ignore
                return value if value is not None else 0.5

            @self.app.callback( # type: ignore
                Output("alpha-input", "value"),
                Input("alpha-slider", "value"),
                prevent_initial_call=True,
            )
            def sync_alpha_input(value: float | None): # type: ignore
                return value

            # Sync feature sigma slider and input
            @self.app.callback( # type: ignore
                Output("feature-sigma-slider", "value"),
                Input("feature-sigma-input", "value"),
                prevent_initial_call=True,
            )
            def sync_feature_sigma_slider(value: float | None): # type: ignore
                return value if value is not None else 1.0

            @self.app.callback( # type: ignore
                Output("feature-sigma-input", "value"),
                Input("feature-sigma-slider", "value"),
                prevent_initial_call=True,
            )
            def sync_feature_sigma_input(value: float | None): # type: ignore
                return value

    def _create_edge_creator(self, *params: Any):
        """Create an edge creator instance with the given parameters."""
        if self.method == GraphCreatorType.knn:
            k = params[0]
            return KNNEdgeCreator(self.device, k=k)

        elif self.method == GraphCreatorType.radius:
            radius = params[0]
            return RadiusEdgeCreator(self.device, radius=radius)

        elif self.method == GraphCreatorType.delaunay_radius:
            limit_radius = params[0]
            return DelaunayEdgeCreator(
                self.device, limit_radius=limit_radius
            )

        elif self.method == GraphCreatorType.dilate:
            dilation = params[0]
            return DilateEdgeCreator(
                self.device, dilation=dilation
            )

        elif self.method == GraphCreatorType.similarity:
            similarity_threshold = params[0]
            distance_sigma = params[1]
            alpha = params[2]
            combination_method = params[3] if len(params) > 3 else "additive"
            distance_metric = params[4] if len(params) > 4 else "gaussian"
            feature_metric = params[5] if len(params) > 5 else "cosine"
            feature_sigma = params[6] if len(params) > 6 else 1.0
            return SimilarityEdgeCreator(
                self.device,
                similarity_threshold=similarity_threshold,
                distance_sigma=distance_sigma,
                alpha=alpha,
                combination_method=combination_method,
                distance_metric=distance_metric,
                feature_metric=feature_metric,
                feature_sigma=feature_sigma,
            )

        else:
            raise ValueError(f"Unsupported method: {self.method}")

    def _create_figure(self, edge_indices: torch.Tensor) -> go.Figure:
        """Create a Plotly figure for the graph visualization."""
        pos_np = cast(np.ndarray[Any, Any],
            self.positions.cpu().numpy() # type: ignore
            if self.positions.is_cuda
            else self.positions.numpy() # type: ignore
        )
        edges_np = cast(np.ndarray[Any, Any],
            edge_indices.cpu().numpy() # type: ignore
            if edge_indices.is_cuda
            else edge_indices.numpy() # type: ignore
        )

        # Create edge traces
        edge_x: List[float | None] = []
        edge_y: List[float | None] = []

        if edges_np.shape[1] > 0:
            for i in range(min(edges_np.shape[1], 50000)):  # Limit for performance
                start_idx, end_idx = int(edges_np[0, i]), int(edges_np[1, i])
                start_pos = pos_np[start_idx]
                end_pos = pos_np[end_idx]

                edge_x.extend([start_pos[0], end_pos[0], None])
                edge_y.extend([start_pos[1], end_pos[1], None])

        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=1, color="rgba(70, 130, 180, 0.4)"),
            hoverinfo="none",
            mode="lines",
            showlegend=False,
        )

        # Create node trace
        node_x = pos_np[:, 0].tolist()
        node_y = pos_np[:, 1].tolist()

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers",
            hoverinfo="text",
            text=[f"Cell {i}" for i in range(len(pos_np))],
            marker=dict(
                size=5,
                color="rgba(255, 100, 100, 0.8)",
                line=dict(width=0.5, color="rgba(255, 255, 255, 0.5)"),
            ),
            showlegend=False,
        )

        # Create figure
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title=dict(
                    text=f"Cell Graph - {self.method.upper()}",
                    font=dict(size=18, color="black"),
                    x=0.5,
                ),
                showlegend=False,
                hovermode="closest",
                margin=dict(b=20, l=20, r=20, t=50),
                xaxis=dict(
                    showgrid=True,
                    gridcolor="rgba(200, 200, 200, 0.2)",
                    zeroline=False,
                    showticklabels=True,
                    title="X coordinate",
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor="rgba(200, 200, 200, 0.2)",
                    zeroline=False,
                    showticklabels=True,
                    title="Y coordinate",
                    autorange="reversed",
                    scaleanchor="x",
                    scaleratio=1,
                ),
                plot_bgcolor="white",
                paper_bgcolor="white",
            ),
        )

        return fig

    def _create_stats(
        self, edge_indices: torch.Tensor, edge_features: torch.Tensor, *params: Any
    ) -> html.Div:
        """Create statistics display for the current graph."""
        n_nodes = len(self.cells)
        n_edges = edge_indices.shape[1]
        avg_degree = (2 * n_edges / n_nodes) if n_nodes > 0 else 0

        # Calculate edge distance statistics
        if edge_features.shape[0] > 0:
            distances = cast(np.ndarray[Any, Any], edge_features[:, 0].cpu().numpy()) # type: ignore
            avg_distance = float(np.mean(distances))
            std_distance = float(np.std(distances))
            min_distance = float(np.min(distances))
            max_distance = float(np.max(distances))
        else:
            avg_distance = std_distance = min_distance = max_distance = 0.0

        # Format parameters
        param_text = self._format_params(*params)

        stats_content: list[Any] = [
            html.H5("Graph Statistics", className="mb-3"),
            html.P([html.Strong("Nodes: "), f"{n_nodes:,}"]),
            html.P([html.Strong("Edges: "), f"{n_edges:,}"]),
            html.P([html.Strong("Avg Degree: "), f"{avg_degree:.2f}"]),
            html.Hr(),
            html.H6("Edge Distances", className="mb-2"),
            html.P([html.Strong("Mean: "), f"{avg_distance:.2f} px"]),
            html.P([html.Strong("Std: "), f"{std_distance:.2f} px"]),
            html.P([html.Strong("Min: "), f"{min_distance:.2f} px"]),
            html.P([html.Strong("Max: "), f"{max_distance:.2f} px"]),
            html.Hr(),
            html.H6("Parameters", className="mb-2"),
            html.Pre(param_text, style={"fontSize": "12px"}),
        ]

        return html.Div(stats_content)

    def _format_params(self, *params: Any) -> str:
        """Format parameters for display."""
        lines: list[str] = []

        if self.method == GraphCreatorType.knn:
            lines.append(f"K: {params[0]}")
        elif self.method == GraphCreatorType.radius:
            lines.append(f"Radius: {params[0]} px")
        elif self.method == GraphCreatorType.delaunay_radius:
            lines.append(f"Limit Radius: {params[0]} px")
        elif self.method == GraphCreatorType.dilate:
            lines.append(f"Dilation: {params[0]} px")
        elif self.method == GraphCreatorType.similarity:
            combination_method = params[3] if len(params) > 3 else "additive"
            distance_metric = params[4] if len(params) > 4 else "gaussian"
            feature_metric = params[5] if len(params) > 5 else "cosine"
            feature_sigma = params[6] if len(params) > 6 else 1.0
            
            param_lines = [
                f"Threshold/K: {params[0]}", # type: ignore
                f"Distance Sigma: {params[1]}", # type: ignore
                f"Alpha: {params[2]}", # type: ignore
                f"Combination: {combination_method}",
                f"Distance Metric: {distance_metric}",
                f"Feature Metric: {feature_metric}",
            ]
            
            # Add feature_sigma if gaussian metric is used
            if feature_metric == "gaussian":
                param_lines.append(f"Feature Sigma: {feature_sigma}")
            
            lines.extend(param_lines)

        return "\n".join(lines)

    def run(self, host: str = "127.0.0.1", port: int = 8050, debug: bool = True):
        """
        Run the Dash application.

        Args:
            port: Port to run the application on
            debug: Enable debug mode for Dash
        """
        print(f"Starting Feature Visualizer Dashboard at http://{host}:{port}")
        self.app.run(host=host, port=port, debug=debug)  # type: ignore

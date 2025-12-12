# -*- coding: utf-8 -*-
# Utility functions
#
# This modules provides utility functions for the CellMIL framework, with some functions adapted from the CellViT project.
#
# References:
# CellViT: Vision Transformers for precise cell segmentation and classification
# Fabian Hörst et al., Medical Image Analysis, 2024
# DOI: https://doi.org/10.1016/j.media.2024.103143

import importlib
import logging
import sys
import os
import torch
import types
import matplotlib.pyplot as plt
from datetime import timedelta
from torch_geometric.data import Data  # type: ignore
from timeit import default_timer as timer
from typing import Dict, List, Optional, Tuple, Union, Any, cast
from types import ModuleType
import plotly.graph_objects as go  # type: ignore
import numpy as np

from cellmil.utils import logger


def to_float(x: torch.Tensor) -> torch.Tensor:
    """Convert tensor to float32"""
    return x.float()


def to_float_normalized(x: torch.Tensor) -> torch.Tensor:
    """Convert tensor to float32 and normalize to [0, 1]"""
    return x.float() / 255.0


# Helper timing functions
def start_timer() -> float:
    """Returns the number of seconds passed since epoch. The epoch is the point where the time starts,
    and is platform dependent.

    Returns:
        float:  The number of seconds passed since epoch
    """
    return timer()


def end_timer(start_time: float, timed_event: str = "Time usage") -> None:
    """Prints the time passed from start_time.


    Args:
        start_time (float): The number of seconds passed since epoch when the timer started
        timed_event (str, optional): A string describing the activity being monitored. Defaults to "Time usage".
    """
    logger.info(f"{timed_event}: {timedelta(seconds=timer() - start_time)}")


def module_exists(
    *names: Union[List[str], str],  # type: ignore
    error: str = "ignore",
    warn_every_time: bool = False,
    __INSTALLED_OPTIONAL_MODULES: Dict[str, bool] = {},
) -> Optional[Union[Tuple[types.ModuleType | None, ...], types.ModuleType]]:
    """Try to import optional dependencies.
    Ref: https://stackoverflow.com/a/73838546/4900327

    Args:
        names (Union(List(str), str)): The module name(s) to import. Str or list of strings.
        error (str, optional): What to do when a dependency is not found:
                * raise : Raise an ImportError.
                * warn: print a warning.
                * ignore: If any module is not installed, return None, otherwise, return the module(s).
            Defaults to "ignore".
        warn_every_time (bool, optional): Whether to warn every time an import is tried. Only applies when error="warn".
            Setting this to True will result in multiple warnings if you try to import the same library multiple times.
            Defaults to False.
    Raises:
        ImportError: ImportError of Module

    Returns:
        Optional[ModuleType, Tuple[ModuleType...]]: The imported module(s), if all are found.
            None is returned if any module is not found and `error!="raise"`.
    """
    assert error in {"raise", "warn", "ignore"}
    if isinstance(names, (list, tuple, set)):  # type: ignore
        names: List[str] = list(names)  # type: ignore
    else:
        print(type(names))
        print(names)
        assert isinstance(names, str)
        names: List[str] = [names]
    modules: list[ModuleType | None] = []
    for name in names:
        try:
            module = importlib.import_module(name)
            modules.append(module)
            __INSTALLED_OPTIONAL_MODULES[name] = True
        except ImportError:
            modules.append(None)

    def error_msg(missing: Union[str, List[str]]):
        if not isinstance(missing, (list, tuple)):
            missing = [missing]
        missing_str: str = " ".join([f'"{name}"' for name in missing])
        dep_str = "dependencies"
        if len(missing) == 1:
            dep_str = "dependency"
        msg = f"Missing optional {dep_str} {missing_str}. Use pip or conda to install."
        return msg

    missing_modules: List[str] = [
        name for name, module in zip(names, modules) if module is None
    ]
    if len(missing_modules) > 0:
        if error == "raise":
            raise ImportError(error_msg(missing_modules))
        if error == "warn":
            for name in missing_modules:
                # Ensures warning is printed only once
                if warn_every_time is True or name not in __INSTALLED_OPTIONAL_MODULES:
                    logger.warning(f"Warning: {error_msg(name)}")
                    __INSTALLED_OPTIONAL_MODULES[name] = False
        return None
    if len(modules) == 1:
        return modules[0]
    return tuple(modules)


def close_logger(logger: logging.Logger) -> None:
    """Closing a logger savely

    Args:
        logger (logging.Logger): Logger to close
    """
    handlers = logger.handlers[:]
    for handler in handlers:
        logger.removeHandler(handler)
        handler.close()

    logger.handlers.clear()
    logging.shutdown()


def get_size_of_dict(d: dict[str, Any]) -> int:
    size = sys.getsizeof(d)
    for key, value in d.items():
        size += sys.getsizeof(key)
        size += sys.getsizeof(value)
    return size


def unflatten_dict(d: dict[str, Any], sep: str = ".") -> dict[str, Any]:
    """Unflatten a flattened dictionary (created a nested dictionary)

    Args:
        d (dict): Dict to be nested
        sep (str, optional): Seperator of flattened keys. Defaults to '.'.

    Returns:
        dict: Nested dict
    """
    output_dict: dict[str, Any] = {}
    for key, value in d.items():
        keys = key.split(sep)
        d = output_dict
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value

    return output_dict


def get_cpu_count() -> int:
    """Get the number of available CPU cores."""
    try:
        return os.cpu_count() or 1  # Fallback to 1 if os.cpu_count() returns None
    except Exception as e:
        logger.error(f"Error getting CPU count: {e}")
        return 1


def create_color_gradient(
    bins: int, initial_color: list[int], final_color: list[int]
) -> list[list[int]]:
    """Create a color gradient from initial_color to final_color.

    Args:
        bins (int): Number of bins in the gradient.
        initial_color (list[int]): RGB values for the initial color.
        final_color (list[int]): RGB values for the final color.

    Returns:
        list[list[int]]: List of RGB colors representing the gradient.
    """
    if len(initial_color) != 3 or len(final_color) != 3:
        raise ValueError(
            "Initial and final colors must be RGB values (lists of length 3)."
        )

    gradient: list[list[int]] = []
    for i in range(bins):
        ratio = i / (bins - 1)
        color: list[int] = [
            int(initial_color[j] + ratio * (final_color[j] - initial_color[j]))
            for j in range(3)
        ]
        gradient.append(color)

    return gradient


def plot_vector(vector: torch.Tensor, title: str = "Vector Plot") -> None:
    """Plot a 1D vector using matplotlib.

    Args:
        vector (torch.Tensor): 1D tensor to plot.
        title (str, optional): Title of the plot. Defaults to "Vector Plot".
    """

    if vector.dim() != 1:
        raise ValueError("Input tensor must be 1-dimensional.")

    plt.figure(figsize=(10, 4))  # type: ignore
    plt.plot(vector.cpu().numpy(), marker="o")  # type: ignore
    plt.title(title)  # type: ignore
    plt.xlabel("Index")  # type: ignore
    plt.ylabel("Value")  # type: ignore
    plt.grid(True)  # type: ignore
    plt.show()  # type: ignore


def plot_graph(data: Data, title: str = "Graph Visualization") -> None:
    """Plot a PyTorch Geometric graph using Plotly.

    Args:
        data (Data): PyTorch Geometric Data object containing graph structure.
        title (str, optional): Title of the plot. Defaults to "Graph Visualization".
    """

    # Extract node positions - if not available, use a simple layout
    if hasattr(data, "pos") and data.pos is not None:
        pos = cast(np.ndarray[Any, Any], data.pos.cpu().numpy())  # type: ignore
        if pos.shape[1] == 2:
            x, y = pos[:, 0], pos[:, 1]
        elif pos.shape[1] >= 3:
            x, y = pos[:, 0], pos[:, 1]
        else:
            # 1D positions, create y coordinates
            x = pos[:, 0]
            y = np.zeros_like(x)
    else:
        # Create a simple circular layout if no positions are available
        num_nodes = data.num_nodes
        angles = cast(
            np.ndarray[Any, Any],
            np.linspace(0, 2 * np.pi, num_nodes, endpoint=False),  # type: ignore
        )
        x = np.cos(angles)
        y = np.sin(angles)

    # Extract edges
    edge_index = cast(np.ndarray[Any, Any], data.edge_index.cpu().numpy())  # type: ignore

    # Create edge traces
    edge_x: list[float | np.ndarray[Any, Any] | None] = []
    edge_y: list[float | np.ndarray[Any, Any] | None] = []
    for i in range(edge_index.shape[1]):
        x0, x1 = x[edge_index[0, i]], x[edge_index[1, i]]
        y0, y1 = y[edge_index[0, i]], y[edge_index[1, i]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=1, color="#888"),
        hoverinfo="none",
        mode="lines",
        name="Edges",
    )

    # Create node trace
    node_trace = go.Scatter(
        x=x,
        y=y,
        mode="markers",
        hoverinfo="text",
        name="Nodes",
        marker=dict(size=10, color="lightblue", line=dict(width=2, color="black")),
    )

    # Add node information to hover text
    node_text: list[str] = []
    node_adjacencies: list[int] = []
    for i in range(len(x)):
        # Count connections: incoming edges + outgoing edges
        num_connections = int((edge_index[1] == i).sum() + (edge_index[0] == i).sum())
        node_adjacencies.append(num_connections)
        node_info = f"Node {i}<br>Connections: {num_connections}"

        # Add node features if available
        if hasattr(data, "x") and data.x is not None:
            if i < data.x.shape[0]:
                features = cast(np.ndarray[Any, Any], data.x[i].cpu().numpy())  # type: ignore
                if len(features) <= 5:  # Show only first 5 features
                    feature_str = ", ".join([f"{f:.3f}" for f in features])
                    node_info += f"<br>Features: [{feature_str}]"
                else:
                    feature_str = ", ".join([f"{f:.3f}" for f in features[:5]]) + "..."
                    node_info += f"<br>Features: [{feature_str}]"

        node_text.append(node_info)

    node_trace.text = node_text

    # Color nodes by number of connections
    if node_adjacencies:
        node_trace.marker.color = node_adjacencies  # type: ignore
        node_trace.marker.colorscale = "Viridis"  # type: ignore
        node_trace.marker.showscale = True  # type: ignore
        node_trace.marker.colorbar = dict(title="Number of connections")  # type: ignore

    # Create the figure
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=dict(text=title, font=dict(size=16)),
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            annotations=[
                dict(
                    text=f"Nodes: {data.num_nodes}, Edges: {data.num_edges}",
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    x=0.005,
                    y=-0.002,
                    xanchor="left",
                    yanchor="bottom",
                    font=dict(size=12),
                )
            ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, autorange="reversed"),
        ),
    )

    fig.write_html(f"graph_{title}.html")  # type: ignore

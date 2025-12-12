import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, cast
from pathlib import Path
from cellmil.interfaces import FeatureVisualizerConfig
from cellmil.interfaces.CellSegmenterConfig import TYPE_NUCLEI_DICT, ModelType
import json

import dash
from dash import dcc, html, Input, Output, dash_table
import plotly.graph_objects as go  # type: ignore
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats  # type: ignore
from plotly.subplots import make_subplots  # type: ignore
from scipy.stats import gaussian_kde  # type: ignore

from cellmil.utils import logger

COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
]

# Common style dictionaries
STYLES: dict[str, dict[str, Any]] = {
    "error": {
        "backgroundColor": "#ffcccc",
        "padding": 15,
        "borderRadius": 5,
        "border": "1px solid #f5c6cb",
        "fontFamily": "'Segoe UI', Arial, sans-serif",
    },
    "warning": {
        "padding": "20px",
        "backgroundColor": "#fff3cd",
        "border": "1px solid #ffc107",
        "borderRadius": "5px",
        "color": "#856404",
        "fontFamily": "'Segoe UI', Arial, sans-serif",
    },
    "info": {
        "padding": "20px",
        "backgroundColor": "#e7f3ff",
        "border": "1px solid #2196F3",
        "borderRadius": "5px",
        "color": "#0d47a1",
        "fontFamily": "'Segoe UI', Arial, sans-serif",
    },
    "cell_type_error": {
        "padding": "20px",
        "backgroundColor": "#f8d7da",
        "border": "1px solid #f5c6cb",
        "borderRadius": "5px",
        "color": "#721c24",
        "fontFamily": "'Segoe UI', Arial, sans-serif",
    },
    "section_combined": {
        "marginBottom": 40,
        "padding": "20px",
        "backgroundColor": "#e8f4f8",
        "borderRadius": "10px",
    },
    "section_slide": {
        "marginBottom": 30,
        "padding": "20px",
        "backgroundColor": "#ffffff",
        "borderRadius": "10px",
        "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
    },
    "section_comparison": {
        "marginBottom": 40,
        "padding": "20px",
        "backgroundColor": "#f8f9fa",
        "borderRadius": "10px",
    },
}


class FeatureVisualizer:
    def __init__(self, config: FeatureVisualizerConfig):
        self.config = config
        self.max_dropdown_levels = 5  # Maximum number of dropdown levels to support

    # ==================== Helper Methods ====================

    @staticmethod
    def _to_numpy(data: Any) -> np.ndarray[Any, Any]:
        """Convert various data types to numpy array."""
        if hasattr(data, "numpy"):
            return data.numpy()
        elif hasattr(data, "detach"):
            return data.detach().numpy()
        return np.array(data)

    @staticmethod
    def _sample_data(
        features: np.ndarray[Any, Any],
        labels: np.ndarray[Any, Any] | None,
        n_samples: int,
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any] | None, str]:
        """
        Sample data if it exceeds n_samples.
        Returns: (sampled_features, sampled_labels, sample_info_string)
        """
        if features.shape[0] > n_samples:
            np.random.seed(42)  # Set seed for reproducibility
            indices = np.random.choice(features.shape[0], n_samples, replace=False)
            sampled_features = features[indices]
            sampled_labels = labels[indices] if labels is not None else None
            sample_info = f" (sampled {n_samples} out of {features.shape[0]})"
        else:
            sampled_features = features
            sampled_labels = labels
            sample_info = ""
        return sampled_features, sampled_labels, sample_info

    @staticmethod
    def _validate_positive_int(value: int | None, default: int) -> int:
        """Validate and return a positive integer, or default if invalid."""
        return value if value and value > 0 else default

    @staticmethod
    def _adjust_perplexity(n_samples: int, requested_perplexity: int) -> int:
        """Adjust perplexity to be valid for the given number of samples."""
        max_perplexity = (n_samples - 1) // 3
        actual_perplexity = min(requested_perplexity, max_perplexity)
        if actual_perplexity != requested_perplexity:
            logger.warning(
                f"Perplexity adjusted from {requested_perplexity} to {actual_perplexity} "
                f"for {n_samples} samples"
            )
        return actual_perplexity

    @staticmethod
    def _create_error_message(
        title: str, message: str, style_key: str = "error"
    ) -> html.Div:
        """Create a standardized error message component."""
        return html.Div(
            [
                html.H4(title, style={"marginBottom": "10px"}),
                html.P(message),
            ],
            style=STYLES.get(style_key, STYLES["error"]),
        )

    @staticmethod
    def _create_cell_type_unavailable_message() -> html.Div:
        """Create a standardized message for when cell type data is unavailable."""
        return html.Div(
            [
                html.H4(
                    "Cell Type Information Not Available",
                    style={"marginBottom": "10px"},
                ),
                html.P(
                    "Cell type data could not be loaded from the slides. "
                    "Make sure cell detection data exists for the selected feature extraction path."
                ),
            ],
            style=STYLES["cell_type_error"],
        )

    def _build_path_from_values(self, *selected_values: str | None) -> List[str]:
        """Build path list from selected dropdown values."""
        current_path: list[str] = []
        for value in selected_values:
            if value is not None:
                current_path.append(value)
            else:
                break
        return current_path

    def _standardize_and_fit_pca(
        self, features: np.ndarray[Any, Any], n_components: int = 2
    ) -> tuple[np.ndarray[Any, Any], PCA]:
        """Standardize features and fit PCA."""
        n_samples, n_features = features.shape

        # Validate n_components
        max_components = min(n_samples, n_features)
        if n_components > max_components:
            logger.warning(
                f"n_components={n_components} exceeds max allowed ({max_components}). "
                f"Reducing to {max_components}."
            )
            n_components = max_components

        scaler = StandardScaler()
        features_scaled = cast(np.ndarray[Any, Any], scaler.fit_transform(features))  # type: ignore
        pca = PCA(n_components=n_components)
        pca_result = cast(np.ndarray[Any, Any], pca.fit_transform(features_scaled))  # type: ignore
        return pca_result, pca

    def _standardize_and_fit_tsne(
        self, features: np.ndarray[Any, Any], perplexity: int
    ) -> np.ndarray[Any, Any]:
        """Standardize features and fit t-SNE."""
        scaler = StandardScaler()
        features_scaled = cast(np.ndarray[Any, Any], scaler.fit_transform(features))  # type: ignore
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_jobs=-1)
        return tsne.fit_transform(features_scaled)  # type: ignore

    def _create_scatter_by_labels(
        self,
        coordinates: np.ndarray[Any, Any],
        labels: np.ndarray[Any, Any],
        label_names: Dict[int, str],
        title: str,
        xlabel: str,
        ylabel: str,
        sample_info: str = "",
    ) -> go.Figure:
        """
        Create a scatter plot colored by labels (cell types or slides).
        Reduces duplication across PCA/t-SNE by cell type methods.
        """
        fig = go.Figure()

        for i, (label_id, label_name) in enumerate(sorted(label_names.items())):
            mask = labels == label_id
            color = COLORS[i % len(COLORS)]

            fig.add_trace(  # type: ignore
                go.Scatter(
                    x=coordinates[mask, 0],
                    y=coordinates[mask, 1],
                    mode="markers",
                    name=label_name,
                    marker=dict(size=5, opacity=0.6, color=color),
                    text=[f"{label_name}<br>Cell {idx}" for idx in np.where(mask)[0]],
                    hovertemplate="<b>%{text}</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>",
                )
            )

        fig.update_layout(  # type: ignore
            title=f"{title}{sample_info}",
            xaxis_title=xlabel,
            yaxis_title=ylabel,
            legend=dict(orientation="v", yanchor="top", y=1, xanchor="right", x=1),
            hovermode="closest",
            width=800,
            height=600,
        )

        return fig

    def _create_js_divergence_table_component(
        self,
        js_df: pd.DataFrame,
        reference_cell_type_name: str,
        is_combined: bool = False,
    ) -> html.Div:
        """
        Create a standardized JS divergence table component.
        Reduces duplication between single slide and combined dataset views.
        """
        dataset_text = "Combined Dataset" if is_combined else ""

        # Prepare data for the table
        table_data = js_df.reset_index().to_dict("records")  # type: ignore
        table_columns: list[dict[str, str | dict[str, str]]] = [
            {"name": "Feature", "id": "Feature"}
        ]
        for col in js_df.columns:
            table_columns.append(
                {
                    "name": col,
                    "id": col,
                    "type": "numeric",
                    "format": {"specifier": ".4f"},
                }
            )

        # Build style_data_conditional list
        style_conditions: list[dict[str, Any]] = [
            {"if": {"row_index": "odd"}, "backgroundColor": "#f9f9f9"},
            {
                "if": {"column_id": "Feature"},
                "fontWeight": "500",
                "backgroundColor": "#ecf0f1",
            },
        ]

        # Add color coding for divergence values
        for col in js_df.columns:
            style_conditions.extend(
                [
                    {
                        "if": {"filter_query": f"{{{col}}} < 0.1", "column_id": col},
                        "backgroundColor": "#d4edda",
                        "color": "#155724",
                    },
                    {
                        "if": {
                            "filter_query": f"{{{col}}} >= 0.1 && {{{col}}} < 0.3",
                            "column_id": col,
                        },
                        "backgroundColor": "#fff3cd",
                        "color": "#856404",
                    },
                    {
                        "if": {"filter_query": f"{{{col}}} >= 0.3", "column_id": col},
                        "backgroundColor": "#f8d7da",
                        "color": "#721c24",
                    },
                ]
            )

        return html.Div(
            [
                html.H4(
                    f"Jensen-Shannon Divergence: {reference_cell_type_name} vs Other Cell Types {dataset_text}",
                    style={
                        "marginBottom": "20px",
                        "fontFamily": "'Segoe UI', Arial, sans-serif",
                        "color": "#2c3e50",
                    },
                ),
                html.P(
                    f"Values represent the Jensen-Shannon divergence between the distribution of each feature in {reference_cell_type_name} cells and other cell types"
                    + (" across all slides" if is_combined else "")
                    + ". Lower values indicate more similar distributions (0 = identical, 1 = completely different).",
                    style={
                        "marginBottom": "20px",
                        "fontFamily": "'Segoe UI', Arial, sans-serif",
                        "color": "#7f8c8d",
                        "fontSize": "14px",
                    },
                ),
                dash_table.DataTable(
                    data=table_data,  # type: ignore
                    columns=table_columns,  # type: ignore
                    style_table={
                        "overflowX": "auto",
                        "maxHeight": "600px",
                        "overflowY": "auto",
                    },
                    style_cell={
                        "textAlign": "left",
                        "padding": "10px",
                        "fontFamily": "'Segoe UI', Arial, sans-serif",
                        "fontSize": "13px",
                    },
                    style_header={
                        "backgroundColor": "#34495e",
                        "color": "white",
                        "fontWeight": "bold",
                        "textAlign": "left",
                        "padding": "12px",
                        "fontFamily": "'Segoe UI', Arial, sans-serif",
                    },
                    style_data_conditional=style_conditions,  # type: ignore
                    page_size=20,
                    sort_action="native",
                    filter_action="native",
                ),
            ],
            style={"padding": "20px"},
        )

    # ==================== Data Loading Methods ====================

    def _get_available_slides(self) -> List[str]:
        """
        Get list of available slide folders in the dataset directory.
        """
        if not self.config.dataset.exists() or not self.config.dataset.is_dir():
            logger.warning(f"Dataset path does not exist: {self.config.dataset}")
            return []

        slides: list[str] = []
        for item in self.config.dataset.iterdir():
            if item.is_dir():
                slides.append(item.name)

        return sorted(slides)

    def _explore_directory(
        self, path: Path, current_path_parts: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Recursively explore directory structure to find features.pt files.
        Returns a nested dictionary structure representing the directory tree.
        """
        if current_path_parts is None:
            current_path_parts = []

        if not path.exists() or not path.is_dir():
            return {}

        result: dict[str, Any] = {}

        for item in path.iterdir():
            if item.is_dir():
                # Check if this directory contains features.pt
                features_file = item / "features.pt"
                if features_file.exists():
                    # This is a terminal directory with features
                    result[item.name] = {
                        "_has_features": True,
                        "_path": str(item),
                        "_path_parts": current_path_parts + [item.name],
                    }
                else:
                    # Recursively explore subdirectories
                    subdirs = self._explore_directory(
                        item, current_path_parts + [item.name]
                    )
                    if subdirs:  # Only add if there are subdirectories with features
                        result[item.name] = subdirs

        return result

    def _get_available_options_at_level(
        self, structure: Dict[str, Any], path_parts: List[str]
    ) -> List[str]:
        """
        Get available options at a specific level in the directory structure.
        """
        current = structure

        # Navigate to the specified level
        for part in path_parts:
            if part in current and isinstance(current[part], dict):
                current = current[part]
            else:
                return []

        # Return available options at this level
        options: list[str] = []
        for key, _ in current.items():
            if not key.startswith("_"):  # Skip metadata keys
                options.append(key)

        return sorted(options)

    def _can_load_features(
        self, structure: Dict[str, Any], path_parts: List[str]
    ) -> bool:
        """
        Check if we can load features at the current path.
        """
        current = structure

        # Navigate to the specified level
        for part in path_parts:
            if part in current and isinstance(current[part], dict):
                current = current[part]
            else:
                return False

        return current.get("_has_features", False)

    def _get_features_path(
        self, structure: Dict[str, Any], path_parts: List[str]
    ) -> str:
        """
        Get the full path to the features.pt file for the given path parts.
        """
        current = structure

        # Navigate to the specified level
        for part in path_parts:
            if part in current and isinstance(current[part], dict):
                current = current[part]
            else:
                raise ValueError(f"Invalid path: {'/'.join(path_parts)}")

        if not current.get("_has_features", False):
            raise ValueError(f"No features available at path: {'/'.join(path_parts)}")

        return current["_path"]

    def _load_features(self, slide_name: str, path_parts: List[str]):
        """
        Load features for the specified slide and path parts.
        """
        feature_extraction_path = (
            self.config.dataset / slide_name / "feature_extraction"
        )
        directory_structure = self._explore_directory(feature_extraction_path)

        if not self._can_load_features(directory_structure, path_parts):
            raise ValueError(f"No features available at path: {'/'.join(path_parts)}")

        features_path = self._get_features_path(directory_structure, path_parts)

        return torch.load(
            Path(features_path) / "features.pt",
            map_location=torch.device("cpu"),
            weights_only=False,
        )

    def _prepare_data(self, slide_name: str, path_parts: List[str]) -> Dict[str, Any]:
        """
        Prepare data for visualization by loading features and converting to DataFrame.
        """
        feature_data = self._load_features(slide_name, path_parts)

        # Extract components
        features = feature_data["features"]  # Shape: (N, D)
        feature_names = feature_data["feature_names"]  # Column names

        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(features, columns=feature_names)

        return {
            "df": df,
            "features": features,
            "feature_names": feature_names,
            "shape": features.shape,
        }

    def _load_cell_types(
        self, slide_name: str, path_parts: List[str]
    ) -> Optional[Dict[int, int]]:
        """
        Load cell types for the specified slide and path parts.
        Returns a dictionary mapping cell_id to cell_type.
        """
        try:
            # The slide_name indicates which slide folder to look in
            # path_parts structure: might be like [extractor_name, ...] or [extractor_name, model_name, ...]

            # Build the slide path from the dataset and slide name
            slide_path = self.config.dataset / slide_name
            logger.info(f"Looking for cell types in slide path: {slide_path}")

            # Try to find segmentation model from path_parts first
            segmentation_model = None
            for part in path_parts:
                try:
                    segmentation_model = ModelType(part)
                    logger.info(
                        f"Found segmentation model in path: {segmentation_model}"
                    )
                    break
                except ValueError:
                    continue

            # If no segmentation model in path, try to find any available cell detection
            cell_detection_base = slide_path / "cell_detection"

            if not cell_detection_base.exists():
                logger.warning(
                    f"Cell detection directory does not exist: {cell_detection_base}"
                )
                return None

            # If we found a segmentation model in the path, use it
            cell_detection_path = None
            if segmentation_model:
                test_path = (
                    cell_detection_base
                    / str(segmentation_model)
                    / "cell_detection.json"
                )
                if test_path.exists():
                    cell_detection_path = test_path
                    logger.info(f"Loading cell types from: {cell_detection_path}")
                else:
                    logger.warning(f"Cell detection file not found: {test_path}")
                    segmentation_model = None

            # If no model specified or file not found, search for any available model
            if not segmentation_model:
                logger.info("Searching for available cell detection files...")
                for model_type in ModelType:
                    test_path = (
                        cell_detection_base / str(model_type) / "cell_detection.json"
                    )
                    if test_path.exists():
                        segmentation_model = model_type
                        cell_detection_path = test_path
                        logger.info(f"Found cell detection for model: {model_type}")
                        break

                if not segmentation_model or cell_detection_path is None:
                    logger.warning("No cell detection files found for any model")
                    return None

            if cell_detection_path is None:
                logger.warning("Cell detection path is None")
                return None

            # Load the cell detection data
            with open(cell_detection_path, "r") as f:
                cell_data = json.load(f)

            cells = cell_data.get("cells", [])
            cell_type_dict: Dict[int, int] = {}
            for cell in cells:
                cell_id = cell.get("cell_id")
                cell_type = cell.get("type", 0)
                if cell_id is not None:
                    cell_type_dict[cell_id] = cell_type

            logger.info(f"Loaded {len(cell_type_dict)} cell types")
            return cell_type_dict

        except Exception as e:
            logger.error(f"Error loading cell types: {e}")
            import traceback

            logger.error(traceback.format_exc())
            return None

    def _prepare_data_with_cell_types(
        self, slide_name: str, path_parts: List[str]
    ) -> Dict[str, Any]:
        """
        Prepare data with cell types for visualization.
        """
        feature_data = self._load_features(slide_name, path_parts)

        # Extract components
        features = feature_data["features"]  # Shape: (N, D)
        feature_names = feature_data["feature_names"]  # Column names
        cell_indices = feature_data.get("cell_indices", {})  # cell_id -> index mapping

        logger.info(
            f"Feature data loaded: {features.shape[0]} cells, {features.shape[1]} features"
        )
        logger.info(
            f"Cell indices available: {len(cell_indices) > 0}, count: {len(cell_indices)}"
        )

        # Load cell types
        cell_types_dict = self._load_cell_types(slide_name, path_parts)

        logger.info(f"Cell types dict loaded: {cell_types_dict is not None}")
        if cell_types_dict:
            logger.info(f"Cell types count: {len(cell_types_dict)}")

        # Convert to DataFrame
        df = pd.DataFrame(features, columns=feature_names)

        # Add cell type information if available
        cell_types = None
        cell_type_names = None

        if cell_types_dict:
            if cell_indices:
                # Map cell types to feature indices using cell_indices mapping
                logger.info("Mapping cell types using cell_indices")
                cell_types = np.zeros(len(features), dtype=int)
                mapped_count = 0
                for cell_id, feature_idx in cell_indices.items():
                    if cell_id in cell_types_dict:
                        cell_types[feature_idx] = cell_types_dict[cell_id]
                        mapped_count += 1

                logger.info(f"Mapped {mapped_count} cells with types")

                # Filter out cell type 0 (background/unknown)
                valid_mask = cell_types != 0
                features = features[valid_mask]
                cell_types = cell_types[valid_mask]
                df = df[valid_mask].reset_index(drop=True)  # type: ignore

                logger.info(
                    f"Filtered out {np.sum(~valid_mask)} cells with type 0 (background/unknown)"
                )

                # Add cell type column to dataframe
                df["cell_type"] = cell_types

                # Create cell type names (excluding type 0)
                cell_type_names = {
                    int(cell_type): TYPE_NUCLEI_DICT.get(
                        int(cell_type), f"Type {int(cell_type)}"
                    )
                    for cell_type in np.unique(cell_types)
                    if int(cell_type) != 0
                }
            else:
                # No cell_indices mapping, assume direct correspondence if counts match
                logger.info("No cell_indices mapping available")
                if len(cell_types_dict) == len(features):
                    logger.info(
                        "Assuming direct cell ID to feature index mapping (counts match)"
                    )
                    cell_types = np.zeros(len(features), dtype=int)
                    # Sort cell_ids to create a consistent mapping
                    sorted_cell_ids = sorted(cell_types_dict.keys())
                    for idx, cell_id in enumerate(sorted_cell_ids):
                        if idx < len(cell_types):
                            cell_types[idx] = cell_types_dict[cell_id]

                    # Filter out cell type 0 (background/unknown)
                    valid_mask = cell_types != 0
                    features = features[valid_mask]
                    cell_types = cell_types[valid_mask]
                    df = df[valid_mask].reset_index(drop=True)  # type: ignore

                    logger.info(
                        f"Filtered out {np.sum(~valid_mask)} cells with type 0 (background/unknown)"
                    )

                    df["cell_type"] = cell_types
                    cell_type_names = {
                        int(cell_type): TYPE_NUCLEI_DICT.get(
                            int(cell_type), f"Type {int(cell_type)}"
                        )
                        for cell_type in np.unique(cell_types)
                        if int(cell_type) != 0
                    }
                    logger.info(
                        f"Created cell type mapping with {len(cell_type_names)} types"
                    )
                else:
                    logger.warning(
                        f"Cannot map cell types: cell_types_dict has {len(cell_types_dict)} entries but features has {len(features)} rows"
                    )

        return {
            "df": df,
            "features": features,
            "feature_names": feature_names,
            "shape": features.shape,
            "cell_types": cell_types,
            "cell_type_names": cell_type_names,
        }

    def _prepare_combined_data(
        self,
        slides: List[str],
        path_parts: List[str],
        max_samples_per_slide: int | None = 1000,
    ) -> Dict[str, Any]:
        """
        Prepare combined data from multiple slides for dataset-wide analysis.
        Samples up to max_samples_per_slide from each slide.
        If max_samples_per_slide is None, use all cells from each slide.
        """
        all_features: list[np.ndarray[Any, Any]] = []
        all_slide_labels: list[int] = []
        feature_names: list[str] | None = None
        total_cells = 0

        logger.info(f"Loading combined data from {len(slides)} slides...")

        for slide_idx, slide_name in enumerate(slides):
            try:
                data = self._prepare_data(slide_name, path_parts)
                features = data["features"]

                if feature_names is None:
                    feature_names = data["feature_names"]
                elif data["feature_names"] != feature_names:
                    logger.warning(
                        f"Feature names mismatch for slide {slide_name}, skipping"
                    )
                    continue

                # Sample if needed and max_samples_per_slide is specified
                if (
                    max_samples_per_slide is not None
                    and len(features) > max_samples_per_slide
                ):
                    np.random.seed(42 + slide_idx)  # Different seed per slide
                    indices = np.random.choice(
                        len(features), max_samples_per_slide, replace=False
                    )
                    features = features[indices]

                all_features.append(features)
                all_slide_labels.extend([slide_idx] * len(features))
                total_cells += len(features)
                logger.info(f"Loaded {len(features)} cells from slide {slide_name}")

            except Exception as e:
                logger.error(f"Error loading slide {slide_name}: {e}")
                continue

        if not all_features:
            raise ValueError("No data could be loaded from any slide")

        # Combine all features
        combined_features = np.vstack(all_features)
        slide_labels = np.array(all_slide_labels)

        # Create DataFrame
        df = pd.DataFrame(combined_features, columns=feature_names)

        logger.info(f"Combined dataset: {total_cells} cells from {len(slides)} slides")

        return {
            "df": df,
            "features": combined_features,
            "feature_names": feature_names,
            "shape": combined_features.shape,
            "slide_labels": slide_labels,
            "slides": slides,
        }

    def _prepare_combined_data_with_cell_types(
        self,
        slides: List[str],
        path_parts: List[str],
        max_samples_per_slide: int | None = 1000,
    ) -> Dict[str, Any]:
        """
        Prepare combined data with cell types from multiple slides.
        If max_samples_per_slide is None, use all cells from each slide.
        """
        all_features: list[np.ndarray[Any, Any]] = []
        all_cell_types: list[np.ndarray[Any, Any]] = []
        all_slide_labels: list[int] = []
        feature_names: list[str] | None = None
        cell_type_names: dict[int, str] | None = None
        total_cells = 0

        logger.info(
            f"Loading combined data with cell types from {len(slides)} slides..."
        )

        for slide_idx, slide_name in enumerate(slides):
            try:
                data = self._prepare_data_with_cell_types(slide_name, path_parts)

                # Skip if no cell types available
                if data["cell_types"] is None:
                    logger.warning(f"No cell types for slide {slide_name}, skipping")
                    continue

                features = data["features"]
                cell_types = data["cell_types"]

                if feature_names is None:
                    feature_names = data["feature_names"]
                    cell_type_names = data["cell_type_names"]
                elif data["feature_names"] != feature_names:
                    logger.warning(
                        f"Feature names mismatch for slide {slide_name}, skipping"
                    )
                    continue

                # Sample if needed and max_samples_per_slide is specified
                if (
                    max_samples_per_slide is not None
                    and len(features) > max_samples_per_slide
                ):
                    np.random.seed(42 + slide_idx)  # Different seed per slide
                    indices = np.random.choice(
                        len(features), max_samples_per_slide, replace=False
                    )
                    features = features[indices]
                    cell_types = cell_types[indices]

                all_features.append(features)
                all_cell_types.append(cell_types)
                all_slide_labels.extend([slide_idx] * len(features))
                total_cells += len(features)
                logger.info(f"Loaded {len(features)} cells from slide {slide_name}")

            except Exception as e:
                logger.error(f"Error loading slide {slide_name}: {e}")
                continue

        if not all_features:
            return {
                "df": pd.DataFrame(),
                "features": np.array([]),
                "feature_names": [],
                "shape": (0, 0),
                "cell_types": None,
                "cell_type_names": None,
                "slide_labels": np.array([]),
                "slides": slides,
            }

        # Combine all data
        combined_features = np.vstack(all_features)
        combined_cell_types = np.concatenate(all_cell_types)
        slide_labels = np.array(all_slide_labels)

        # Filter out cell type 0 (background/unknown/unlabeled cells)
        valid_mask = combined_cell_types != 0
        combined_features = combined_features[valid_mask]
        combined_cell_types = combined_cell_types[valid_mask]
        slide_labels = slide_labels[valid_mask]

        # Update total cells count after filtering
        total_cells = len(combined_features)
        logger.info(
            f"Filtered out {np.sum(~valid_mask)} cells with type 0 (background/unknown)"
        )

        # Create complete cell_type_names dictionary from all unique cell types (excluding type 0)
        all_unique_types = np.unique(combined_cell_types)
        cell_type_names = {
            int(cell_type): TYPE_NUCLEI_DICT.get(
                int(cell_type), f"Type {int(cell_type)}"
            )
            for cell_type in all_unique_types
            if int(cell_type) != 0  # Explicitly exclude type 0
        }
        logger.info(
            f"Created cell type mapping with {len(cell_type_names)} types: {list(cell_type_names.values())}"
        )

        # Create DataFrame
        df = pd.DataFrame(combined_features, columns=feature_names)

        logger.info(f"Combined dataset: {total_cells} cells from {len(slides)} slides")

        return {
            "df": df,
            "features": combined_features,
            "feature_names": feature_names,
            "shape": combined_features.shape,
            "cell_types": combined_cell_types,
            "cell_type_names": cell_type_names,
            "slide_labels": slide_labels,
            "slides": slides,
        }

    def _calculate_first_order_stats(
        self, data: np.ndarray[Any, Any]
    ) -> Dict[str, Any]:
        """Calculate first-order statistics for features."""
        # Convert to numpy array if needed
        data = self._to_numpy(data)

        return {
            "mean": np.mean(data, axis=0),
            "std": np.std(data, axis=0),
            "min": np.min(data, axis=0),
            "max": np.max(data, axis=0),
            "median": np.median(data, axis=0),
            "q25": np.percentile(data, 25, axis=0),
            "q75": np.percentile(data, 75, axis=0),
            "skewness": stats.skew(data, axis=0),  # type: ignore
            "kurtosis": stats.kurtosis(data, axis=0),  # type: ignore
        }

    def _create_correlation_matrix(
        self, df: pd.DataFrame, feature_names: List[str]
    ) -> go.Figure:
        """Create correlation matrix heatmap for features."""
        logger.info(
            "Computing correlation matrix... This may take a moment for large datasets."
        )

        # Limit to first 20 features to avoid overcrowding
        limited_features = feature_names[:20]
        correlation_matrix = df[limited_features].corr()

        fig = go.Figure(
            data=go.Heatmap(
                z=correlation_matrix.values,  # type: ignore
                x=correlation_matrix.columns,
                y=correlation_matrix.columns,
                colorscale="RdBu",
                zmid=0,
                text=correlation_matrix.values,  # type: ignore
                texttemplate="%{text:.2f}",
                textfont={"size": 8},
                hovertemplate="<b>%{x}</b><br><b>%{y}</b><br>Correlation: %{z:.3f}<extra></extra>",
            )
        )

        fig.update_layout(  # type: ignore
            title="Feature Correlation Matrix",
            xaxis_title="Features",
            yaxis_title="Features",
            width=800,
            height=600,
        )

        return fig

    def _create_distribution_plot(
        self, df: pd.DataFrame, feature_name: str
    ) -> go.Figure:
        """Create distribution plot for a specific feature."""
        fig = go.Figure()

        # Histogram
        fig.add_trace(  # type: ignore
            go.Histogram(
                x=df[feature_name], name="Distribution", nbinsx=50, opacity=0.7
            )
        )

        fig.update_layout(  # type: ignore
            title=f"Distribution of {feature_name}",
            xaxis_title=feature_name,
            yaxis_title="Frequency",
        )

        return fig

    def _create_pca_plot(
        self,
        features: np.ndarray[Any, Any],
        feature_names: List[str],
        n_samples: int = 1000,
    ) -> go.Figure:
        """Create PCA visualization."""
        logger.info(
            f"Computing PCA with {n_samples} samples... This may take a moment for large datasets."
        )

        # Convert and sample data
        features = self._to_numpy(features)
        features, _, sample_info = self._sample_data(features, None, n_samples)

        # Standardize and perform PCA
        pca_result, pca = self._standardize_and_fit_pca(features)

        # Create scatter plot
        fig = go.Figure()

        fig.add_trace(  # type: ignore
            go.Scatter(
                x=pca_result[:, 0],
                y=pca_result[:, 1],
                mode="markers",
                marker=dict(
                    size=5,
                    opacity=0.6,
                    color=np.arange(len(pca_result)),  # type: ignore
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title="Cell Index"),
                ),
                text=[f"Cell {i}" for i in range(len(pca_result))],  # type: ignore
                hovertemplate="<b>%{text}</b><br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>",
            )
        )

        fig.update_layout(  # type: ignore
            title=f"PCA Visualization{sample_info} (Explained Variance: PC1={pca.explained_variance_ratio_[0]:.2%}, PC2={pca.explained_variance_ratio_[1]:.2%})",  # type: ignore
            xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]:.2%})",  # type: ignore
            yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]:.2%})",  # type: ignore
        )

        return fig

    def _create_tsne_plot(
        self,
        features: np.ndarray[Any, Any],
        n_samples: int = 1000,
        perplexity: int = 30,
    ) -> go.Figure:
        """Create t-SNE visualization."""
        logger.info("Computing t-SNE (this may take a while)...")

        # Convert and sample data
        features = self._to_numpy(features)
        features, _, sample_info = self._sample_data(features, None, n_samples)

        # Adjust perplexity and perform t-SNE
        actual_perplexity = self._adjust_perplexity(features.shape[0], perplexity)
        tsne_result = self._standardize_and_fit_tsne(features, actual_perplexity)

        # Create scatter plot
        fig = go.Figure()

        fig.add_trace(  # type: ignore
            go.Scatter(
                x=tsne_result[:, 0],
                y=tsne_result[:, 1],
                mode="markers",
                marker=dict(
                    size=5,
                    opacity=0.6,
                    color=np.arange(len(tsne_result)),  # type: ignore
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title="Cell Index"),
                ),
                text=[f"Cell {i}" for i in range(len(tsne_result))],  # type: ignore
                hovertemplate="<b>%{text}</b><br>t-SNE1: %{x:.2f}<br>t-SNE2: %{y:.2f}<extra></extra>",
            )
        )

        fig.update_layout(  # type: ignore
            title=f"t-SNE Visualization{sample_info}<br>(perplexity={actual_perplexity})",
            xaxis_title="t-SNE Component 1",
            yaxis_title="t-SNE Component 2",
        )

        return fig

    def _create_stats_table(
        self, stats_dict: Dict[str, Any], feature_names: List[str]
    ) -> go.Figure:
        """Create a table with first-order statistics."""

        # Create table data
        table_data = []
        for i, feature_name in enumerate(feature_names):
            if i < len(stats_dict["mean"]):  # Safety check
                table_data.append(  # type: ignore
                    [
                        feature_name,
                        f"{stats_dict['mean'][i]:.4f}",
                        f"{stats_dict['std'][i]:.4f}",
                        f"{stats_dict['min'][i]:.4f}",
                        f"{stats_dict['max'][i]:.4f}",
                        f"{stats_dict['median'][i]:.4f}",
                        f"{stats_dict['q25'][i]:.4f}",
                        f"{stats_dict['q75'][i]:.4f}",
                        f"{stats_dict['skewness'][i]:.4f}",
                        f"{stats_dict['kurtosis'][i]:.4f}",
                    ]
                )

        # Transpose the data for proper table formatting
        if table_data:
            transposed_data = list(zip(*table_data))  # type: ignore
        else:
            transposed_data = [[] for _ in range(10)]  # type: ignore

        fig = go.Figure(
            data=[
                go.Table(  # type: ignore
                    header=dict(
                        values=[
                            "Feature",
                            "Mean",
                            "Std",
                            "Min",
                            "Max",
                            "Median",
                            "Q25",
                            "Q75",
                            "Skewness",
                            "Kurtosis",
                        ],
                        fill_color="paleturquoise",
                        align="left",
                        font=dict(size=12),
                    ),
                    cells=dict(
                        values=transposed_data,  # type: ignore
                        fill_color="lavender",
                        align="left",
                        font=dict(size=10),
                    ),
                )
            ]
        )

        fig.update_layout(title="First-Order Statistics")  # type: ignore

        return fig

    def _create_distribution_comparison_plot(
        self,
        df: pd.DataFrame,
        feature_name: str,
        cell_types: np.ndarray[Any, Any],
        cell_type_names: Dict[int, str],
    ) -> go.Figure:
        """Create overlaid distribution plots for different cell types with normalized densities and KDE curves."""

        # Create figure with 2 subplots (histogram on top, KDE below)
        fig = make_subplots(
            rows=2,
            cols=1,
            row_heights=[0.55, 0.45],
            subplot_titles=(
                "Normalized Histograms",
                "Smoothed Kernel Density Estimates",
            ),
            vertical_spacing=0.12,
        )

        # Define a color palette for cell types
        colors = COLORS

        # Get overall data range for consistent x-axis
        data_min = df[feature_name].min()
        data_max = df[feature_name].max()
        data_range = data_max - data_min
        x_range = [data_min - 0.05 * data_range, data_max + 0.05 * data_range]

        # Add normalized histogram and KDE for each cell type
        for i, (cell_type, type_name) in enumerate(sorted(cell_type_names.items())):
            if cell_type == 0:  # Skip unknown type
                continue
            mask = cell_types == cell_type
            cell_count = mask.sum()
            if cell_count == 0:
                continue

            color = colors[i % len(colors)]
            # Calculate percentage of total cells
            total_cells = len(cell_types)
            percentage = (cell_count / total_cells) * 100

            data_values = cast(np.ndarray[Any, Any], df[feature_name][mask].values)  # type: ignore
            label = f"{type_name} (n={cell_count}, {percentage:.1f}%)"

            # Add histogram to first subplot
            fig.add_trace(  # type: ignore
                go.Histogram(
                    x=data_values,
                    name=label,
                    nbinsx=50,
                    opacity=0.6,
                    marker_color=color,
                    histnorm="probability density",
                    legendgroup=f"group{i}",
                    showlegend=True,
                ),
                row=1,
                col=1,
            )

            # Calculate and add KDE to second subplot
            if cell_count > 1:  # Need at least 2 points for KDE
                try:
                    kde = gaussian_kde(data_values)
                    # Create smooth x values for the KDE curve
                    x_smooth = cast(
                        np.ndarray[Any, Any], np.linspace(x_range[0], x_range[1], 300)
                    )
                    y_smooth = cast(np.ndarray[Any, Any], kde(x_smooth))

                    fig.add_trace(  # type: ignore
                        go.Scatter(
                            x=x_smooth,
                            y=y_smooth,
                            name=label,
                            mode="lines",
                            line=dict(color=color, width=2.5),
                            legendgroup=f"group{i}",
                            showlegend=False,  # Already shown in histogram
                            hovertemplate=f"<b>{type_name}</b><br>{feature_name}: %{{x:.3f}}<br>Density: %{{y:.3f}}<extra></extra>",
                        ),
                        row=2,
                        col=1,
                    )
                except Exception as e:
                    logger.warning(f"Could not compute KDE for {type_name}: {e}")

        # Update layout
        fig.update_layout(  # type: ignore
            title=f"Distribution Comparison of {feature_name} by Cell Type",
            barmode="overlay",
            legend=dict(
                orientation="v",
                yanchor="top",
                y=0.98,
                xanchor="right",
                x=0.99,
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="rgba(0, 0, 0, 0.2)",
                borderwidth=1,
            ),
            hovermode="closest",
            height=800,
        )

        # Update x and y axes
        fig.update_xaxes(title_text=feature_name, row=1, col=1, range=x_range)  # type: ignore
        fig.update_xaxes(title_text=feature_name, row=2, col=1, range=x_range)  # type: ignore
        fig.update_yaxes(title_text="Probability Density", row=1, col=1)  # type: ignore
        fig.update_yaxes(title_text="Density", row=2, col=1)  # type: ignore

        return fig

    def _calculate_js_divergence_table(
        self,
        df: pd.DataFrame,
        cell_types: np.ndarray[Any, Any],
        cell_type_names: Dict[int, str],
        reference_cell_type: int,
    ) -> pd.DataFrame:
        """
        Calculate Jensen-Shannon divergence between reference cell type and all other types
        for each feature in the dataframe.

        Returns a DataFrame where:
        - Rows are features
        - Columns are cell types (excluding reference type)
        - Values are JS divergence scores
        """
        from scipy.spatial.distance import jensenshannon

        # Get all features (columns in df)
        features = [col for col in df.columns if col != "cell_type"]

        # Get other cell types (excluding reference and unknown type 0)
        other_types = sorted(
            [
                ct
                for ct in cell_type_names.keys()
                if ct != reference_cell_type and ct != 0
            ]
        )

        # Initialize results dictionary
        results: dict[str, list[np.float64 | float]] = {
            cell_type_names[ct]: [] for ct in other_types
        }

        # Calculate JS divergence for each feature
        for feature in features:
            ref_mask = cell_types == reference_cell_type
            ref_values = cast(np.ndarray[Any, Any], df[feature][ref_mask].values)  # type: ignore

            if len(ref_values) < 2:
                # Not enough data for reference type
                for ct in other_types:
                    results[cell_type_names[ct]].append(np.nan)
                continue

            # Calculate histogram for reference type
            ref_counts, bin_edges = np.histogram(ref_values, bins=50, density=False)

            for ct in other_types:
                ct_mask = cell_types == ct
                ct_values = cast(np.ndarray[Any, Any], df[feature][ct_mask].values)  # type: ignore

                if len(ct_values) < 2:
                    results[cell_type_names[ct]].append(np.nan)
                    continue

                # Calculate histogram for comparison type using same bins
                ct_counts, _ = np.histogram(ct_values, bins=bin_edges, density=False)

                # Add small epsilon to avoid log(0) and normalize to probability
                epsilon = 1e-10
                ref_prob = (ref_counts + epsilon) / (
                    ref_counts.sum() + epsilon * len(ref_counts)
                )
                ct_prob = (ct_counts + epsilon) / (
                    ct_counts.sum() + epsilon * len(ct_counts)
                )

                # Calculate Jensen-Shannon divergence
                js_div = jensenshannon(ref_prob, ct_prob)
                results[cell_type_names[ct]].append(js_div)

        # Create DataFrame
        js_df = pd.DataFrame(results, index=features)
        js_df.index.name = "Feature"

        return js_df

    def _create_pca_by_cell_type(
        self,
        features: np.ndarray[Any, Any] | torch.Tensor,
        cell_types: np.ndarray[Any, Any],
        cell_type_names: Dict[int, str],
        n_samples: int = 1000,
    ) -> go.Figure:
        """Create PCA visualization colored by cell type."""
        logger.info("Computing PCA with cell types...")

        # Convert and sample data
        features = self._to_numpy(features)
        features, sampled_cell_types, sample_info = self._sample_data(
            features, cell_types, n_samples
        )
        # sampled_cell_types will not be None since we passed cell_types
        assert sampled_cell_types is not None
        cell_types = sampled_cell_types

        # Standardize and perform PCA
        pca_result, pca = self._standardize_and_fit_pca(features)

        # Create scatter plot using unified method
        title = (
            f"PCA by Cell Type<br>Explained Variance: "
            f"PC1={pca.explained_variance_ratio_[0]:.2%}, "  # type: ignore
            f"PC2={pca.explained_variance_ratio_[1]:.2%}"  # type: ignore
        )
        xlabel = f"PC1 ({pca.explained_variance_ratio_[0]:.2%})"  # type: ignore
        ylabel = f"PC2 ({pca.explained_variance_ratio_[1]:.2%})"  # type: ignore

        return self._create_scatter_by_labels(
            pca_result, cell_types, cell_type_names, title, xlabel, ylabel, sample_info
        )

    def _create_tsne_by_cell_type(
        self,
        features: np.ndarray[Any, Any] | torch.Tensor,
        cell_types: np.ndarray[Any, Any],
        cell_type_names: Dict[int, str],
        n_samples: int = 1000,
        perplexity: int = 30,
    ) -> go.Figure:
        """Create t-SNE visualization colored by cell type."""
        logger.info("Computing t-SNE with cell types (this may take a while)...")

        # Convert and sample data
        features = self._to_numpy(features)
        features, sampled_cell_types, sample_info = self._sample_data(
            features, cell_types, n_samples
        )
        # sampled_cell_types will not be None since we passed cell_types
        assert sampled_cell_types is not None
        cell_types = sampled_cell_types

        # Adjust perplexity and perform t-SNE
        actual_perplexity = self._adjust_perplexity(features.shape[0], perplexity)
        tsne_result = self._standardize_and_fit_tsne(features, actual_perplexity)

        # Create scatter plot using unified method
        title = f"t-SNE by Cell Type<br>(perplexity={actual_perplexity})"
        xlabel = "t-SNE Component 1"
        ylabel = "t-SNE Component 2"

        return self._create_scatter_by_labels(
            tsne_result, cell_types, cell_type_names, title, xlabel, ylabel, sample_info
        )

    def _create_combined_pca_plot(
        self,
        features: np.ndarray[Any, Any],
        slide_labels: np.ndarray[Any, Any],
        slides: List[str],
    ) -> go.Figure:
        """Create PCA visualization colored by slide for combined dataset."""
        logger.info("Computing PCA for combined dataset...")

        # Convert to numpy array if needed
        if hasattr(features, "numpy"):
            features = features.numpy()  # type: ignore
        elif hasattr(features, "detach"):
            features = features.detach().numpy()  # type: ignore
        features = np.array(features)  # type: ignore

        # Standardize and perform PCA
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)  # type: ignore
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(features_scaled)  # type: ignore

        # Create scatter plot with different colors for each slide
        fig = go.Figure()
        colors = COLORS

        for i, slide_name in enumerate(slides):
            mask = slide_labels == i
            if mask.sum() == 0:
                continue

            color = colors[i % len(colors)]
            fig.add_trace(  # type: ignore
                go.Scatter(
                    x=pca_result[mask, 0],
                    y=pca_result[mask, 1],
                    mode="markers",
                    name=slide_name,
                    marker=dict(
                        size=5,
                        opacity=0.6,
                        color=color,
                    ),
                    hovertemplate=f"<b>{slide_name}</b><br>PC1: %{{x:.2f}}<br>PC2: %{{y:.2f}}<extra></extra>",
                )
            )

        fig.update_layout(  # type: ignore
            title=f"PCA - Combined Dataset<br>Explained Variance: PC1={pca.explained_variance_ratio_[0]:.2%}, PC2={pca.explained_variance_ratio_[1]:.2%}<br>Total cells: {len(features)}",  # type: ignore
            xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]:.2%})",  # type: ignore
            yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]:.2%})",  # type: ignore
            legend=dict(orientation="v", yanchor="top", y=1, xanchor="right", x=1),
            hovermode="closest",
            width=900,
            height=700,
        )

        return fig

    def _create_combined_tsne_plot(
        self,
        features: np.ndarray[Any, Any],
        slide_labels: np.ndarray[Any, Any],
        slides: List[str],
        perplexity: int = 30,
    ) -> go.Figure:
        """Create t-SNE visualization colored by slide for combined dataset."""
        logger.info("Computing t-SNE for combined dataset...")

        # Convert to numpy array if needed
        if hasattr(features, "numpy"):
            features = features.numpy()  # type: ignore
        elif hasattr(features, "detach"):
            features = features.detach().numpy()  # type: ignore
        features = np.array(features)  # type: ignore

        # Adjust perplexity if needed
        max_perplexity = (features.shape[0] - 1) // 3
        actual_perplexity = min(perplexity, max_perplexity)
        if actual_perplexity != perplexity:
            logger.warning(
                f"Perplexity adjusted from {perplexity} to {actual_perplexity} based on dataset size"
            )

        # Standardize and perform t-SNE
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)  # type: ignore
        tsne = TSNE(
            n_components=2, perplexity=actual_perplexity, random_state=42, n_jobs=-1
        )
        tsne_result = tsne.fit_transform(features_scaled)  # type: ignore

        # Create scatter plot with different colors for each slide
        fig = go.Figure()
        colors = COLORS

        for i, slide_name in enumerate(slides):
            mask = slide_labels == i
            if mask.sum() == 0:
                continue

            color = colors[i % len(colors)]
            fig.add_trace(  # type: ignore
                go.Scatter(
                    x=tsne_result[mask, 0],
                    y=tsne_result[mask, 1],
                    mode="markers",
                    name=slide_name,
                    marker=dict(
                        size=5,
                        opacity=0.6,
                        color=color,
                    ),
                    hovertemplate=f"<b>{slide_name}</b><br>t-SNE1: %{{x:.2f}}<br>t-SNE2: %{{y:.2f}}<extra></extra>",
                )
            )

        fig.update_layout(  # type: ignore
            title=f"t-SNE - Combined Dataset<br>(perplexity={actual_perplexity})<br>Total cells: {len(features)}",
            xaxis_title="t-SNE Component 1",
            yaxis_title="t-SNE Component 2",
            legend=dict(orientation="v", yanchor="top", y=1, xanchor="right", x=1),
            hovermode="closest",
            width=900,
            height=700,
        )

        return fig

    def _create_combined_cell_type_distribution(
        self,
        cell_types: np.ndarray[Any, Any],
        cell_type_names: Dict[int, str],
        slide_labels: np.ndarray[Any, Any],
        slides: List[str],
    ) -> go.Figure:
        """Create stacked bar chart showing cell type distribution across slides."""

        # Calculate cell type counts per slide
        data_for_plot: list[dict[str, list[int] | str]] = []

        for cell_type, type_name in sorted(cell_type_names.items()):
            if cell_type == 0:  # Skip unknown
                continue

            counts: list[int] = []
            for i in range(len(slides)):
                mask = (slide_labels == i) & (cell_types == cell_type)
                counts.append(mask.sum())

            data_for_plot.append({"type_name": type_name, "counts": counts})

        # Create grouped bar chart
        fig = go.Figure()
        colors = COLORS

        for i, data in enumerate(data_for_plot):
            color = colors[i % len(colors)]
            fig.add_trace(  # type: ignore
                go.Bar(
                    name=data["type_name"],
                    x=slides,
                    y=data["counts"],
                    marker_color=color,
                    hovertemplate=f"<b>{data['type_name']}</b><br>Slide: %{{x}}<br>Count: %{{y}}<extra></extra>",
                )
            )

        fig.update_layout(  # type: ignore
            title="Cell Type Distribution Across Slides",
            xaxis_title="Slide",
            yaxis_title="Cell Count",
            barmode="group",
            legend=dict(orientation="v", yanchor="top", y=1, xanchor="right", x=1),
            height=600,
        )

        return fig

    def _create_combined_distribution_comparison(
        self,
        df: pd.DataFrame,
        feature_name: str,
        cell_types: np.ndarray[Any, Any],
        cell_type_names: Dict[int, str],
    ) -> go.Figure:
        """Create distribution comparison across all cell types for combined dataset."""

        # Create figure with 2 subplots
        fig = make_subplots(
            rows=2,
            cols=1,
            row_heights=[0.55, 0.45],
            subplot_titles=(
                "Normalized Histograms",
                "Smoothed Kernel Density Estimates",
            ),
            vertical_spacing=0.12,
        )

        colors = COLORS

        # Get overall data range
        data_min = df[feature_name].min()
        data_max = df[feature_name].max()
        data_range = data_max - data_min
        x_range = [data_min - 0.05 * data_range, data_max + 0.05 * data_range]

        # Add histogram and KDE for each cell type
        for i, (cell_type, type_name) in enumerate(sorted(cell_type_names.items())):
            if cell_type == 0:  # Skip unknown
                continue

            mask = cell_types == cell_type
            cell_count = mask.sum()
            if cell_count == 0:
                continue

            data = df[feature_name][mask]
            color = colors[i % len(colors)]
            legend_name = f"{type_name} (n={cell_count})"

            # Normalized histogram
            fig.add_trace(  # type: ignore
                go.Histogram(
                    x=data,
                    name=legend_name,
                    marker_color=color,
                    opacity=0.6,
                    histnorm="probability density",
                    nbinsx=50,
                    showlegend=True,
                    legendgroup=type_name,
                    hovertemplate=f"<b>{type_name}</b><br>{feature_name}: %{{x:.2f}}<br>Density: %{{y:.4f}}<extra></extra>",
                ),
                row=1,
                col=1,
            )

            # KDE curve
            if len(data) > 1:
                try:
                    kde = gaussian_kde(data)
                    x_vals = cast(
                        np.ndarray[Any, Any], np.linspace(x_range[0], x_range[1], 200)
                    )
                    kde_vals = cast(np.ndarray[Any, Any], kde(x_vals))

                    fig.add_trace(  # type: ignore
                        go.Scatter(
                            x=x_vals,
                            y=kde_vals,
                            name=legend_name,
                            mode="lines",
                            line=dict(color=color, width=2),
                            showlegend=False,
                            legendgroup=type_name,
                            hovertemplate=f"<b>{type_name}</b><br>{feature_name}: %{{x:.2f}}<br>Density: %{{y:.4f}}<extra></extra>",
                        ),
                        row=2,
                        col=1,
                    )
                except Exception as e:
                    logger.warning(f"Could not compute KDE for {type_name}: {e}")

        fig.update_layout(  # type: ignore
            title=f"Distribution Comparison of {feature_name} by Cell Type (Combined Dataset)",
            barmode="overlay",
            legend=dict(
                orientation="v",
                yanchor="top",
                y=0.98,
                xanchor="right",
                x=0.99,
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="rgba(0, 0, 0, 0.2)",
                borderwidth=1,
            ),
            hovermode="closest",
            height=800,
        )

        fig.update_xaxes(title_text=feature_name, row=1, col=1, range=x_range)  # type: ignore
        fig.update_xaxes(title_text=feature_name, row=2, col=1, range=x_range)  # type: ignore
        fig.update_yaxes(title_text="Probability Density", row=1, col=1)  # type: ignore
        fig.update_yaxes(title_text="Density", row=2, col=1)  # type: ignore

        return fig

    def _create_combined_pca_by_cell_type(
        self,
        features: np.ndarray[Any, Any] | torch.Tensor,
        cell_types: np.ndarray[Any, Any],
        cell_type_names: Dict[int, str],
    ) -> go.Figure:
        """Create PCA visualization colored by cell type for combined dataset."""
        logger.info("Computing PCA by cell type for combined dataset...")

        # Convert to numpy array and perform PCA
        features = self._to_numpy(features)
        pca_result, pca = self._standardize_and_fit_pca(features)

        # Create scatter plot using unified method
        title = (
            f"PCA by Cell Type - Combined Dataset<br>"
            f"Explained Variance: PC1={pca.explained_variance_ratio_[0]:.2%}, "  # type: ignore
            f"PC2={pca.explained_variance_ratio_[1]:.2%}<br>"  # type: ignore
            f"Total cells: {len(features)}"
        )
        xlabel = f"PC1 ({pca.explained_variance_ratio_[0]:.2%})"  # type: ignore
        ylabel = f"PC2 ({pca.explained_variance_ratio_[1]:.2%})"  # type: ignore

        fig = self._create_scatter_by_labels(
            pca_result, cell_types, cell_type_names, title, xlabel, ylabel
        )
        fig.update_layout(width=900, height=700)  # type: ignore
        return fig

    def _create_combined_tsne_by_cell_type(
        self,
        features: np.ndarray[Any, Any] | torch.Tensor,
        cell_types: np.ndarray[Any, Any],
        cell_type_names: Dict[int, str],
        perplexity: int = 30,
    ) -> go.Figure:
        """Create t-SNE visualization colored by cell type for combined dataset."""
        logger.info("Computing t-SNE by cell type for combined dataset...")

        # Convert to numpy array and perform t-SNE
        features = self._to_numpy(features)
        actual_perplexity = self._adjust_perplexity(features.shape[0], perplexity)
        tsne_result = self._standardize_and_fit_tsne(features, actual_perplexity)

        # Create scatter plot using unified method
        title = (
            f"t-SNE by Cell Type - Combined Dataset<br>"
            f"(perplexity={actual_perplexity})<br>"
            f"Total cells: {len(features)}"
        )
        xlabel = "t-SNE Component 1"
        ylabel = "t-SNE Component 2"

        fig = self._create_scatter_by_labels(
            tsne_result, cell_types, cell_type_names, title, xlabel, ylabel
        )
        fig.update_layout(width=900, height=700)  # type: ignore
        return fig

        fig.update_layout(  # type: ignore
            title=f"t-SNE by Cell Type - Combined Dataset<br>(perplexity={actual_perplexity})<br>Total cells: {len(features)}",
            xaxis_title="t-SNE Component 1",
            yaxis_title="t-SNE Component 2",
            legend=dict(orientation="v", yanchor="top", y=1, xanchor="right", x=1),
            hovermode="closest",
            width=900,
            height=700,
        )

        return fig

    def visualize(self, host: str = "127.0.0.1", port: int = 8050, debug: bool = True):
        """
        Launch the Dash web application for feature visualization.
        """
        app = dash.Dash(__name__)

        # Get available slides
        available_slides = self._get_available_slides()

        # Generate dynamic dropdowns based on max levels
        def create_dropdown_components():
            components: list[Any] = []
            for level in range(self.max_dropdown_levels):
                components.append(
                    html.Div(
                        [
                            html.Label(f"Level {level + 1}:"),
                            dcc.Dropdown(
                                id=f"dropdown-level-{level}",
                                style={"marginBottom": 10},
                            ),
                        ],
                        style={
                            "width": f"{90 // min(3, self.max_dropdown_levels)}%",
                            "display": "inline-block",
                            "marginRight": "2%",
                        },
                        id=f"dropdown-container-{level}",
                    )
                )
            return components

        # Define the layout
        app.layout = html.Div(
            [
                html.H1(
                    "Feature Visualizer Dashboard",
                    style={
                        "textAlign": "center",
                        "marginBottom": 30,
                        "color": "#2c3e50",
                        "fontWeight": "bold",
                        "fontFamily": "'Segoe UI', 'Helvetica Neue', Arial, sans-serif",
                        "padding": "20px",
                        "backgroundColor": "#ecf0f1",
                        "borderRadius": "10px",
                        "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
                    },
                ),
                # Analysis Mode Selection
                html.Div(
                    [
                        html.H3(
                            "Analysis Mode",
                            style={
                                "marginBottom": 15,
                                "color": "#2c3e50",
                                "fontFamily": "'Segoe UI', Arial, sans-serif",
                            },
                        ),
                        dcc.RadioItems(
                            id="analysis-mode",
                            options=[
                                {
                                    "label": " Dataset-Wide Analysis (All Slides Combined)",
                                    "value": "combined",
                                },
                                {
                                    "label": " Slide-Specific Analysis",
                                    "value": "single",
                                },
                            ],
                            value="combined",
                            labelStyle={"display": "block", "marginBottom": "10px"},
                            style={"fontSize": "16px"},
                        ),
                    ],
                    style={
                        "marginBottom": 30,
                        "padding": "20px",
                        "backgroundColor": "#ffffff",
                        "borderRadius": "10px",
                        "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
                    },
                ),
                # Slide Selection
                html.Div(
                    [
                        html.H3(
                            "Slide Selection",
                            style={
                                "marginBottom": 20,
                                "color": "#2c3e50",
                                "fontFamily": "'Segoe UI', Arial, sans-serif",
                            },
                        ),
                        html.Div(
                            [
                                html.Label("Select Slide:"),
                                dcc.Dropdown(
                                    id="slide-dropdown",
                                    options=[
                                        {"label": slide, "value": slide}
                                        for slide in available_slides
                                    ],
                                    value=available_slides[0]
                                    if available_slides
                                    else None,
                                    style={"marginBottom": 10},
                                ),
                            ],
                        ),
                    ],
                    style={
                        "marginBottom": 30,
                        "padding": "20px",
                        "backgroundColor": "#ffffff",
                        "borderRadius": "10px",
                        "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
                    },
                    id="slide-selection-container",
                ),
                # Dynamic Controls
                html.Div(
                    [
                        html.H3(
                            "Feature Path Selection",
                            style={
                                "marginBottom": 20,
                                "color": "#2c3e50",
                                "fontFamily": "'Segoe UI', Arial, sans-serif",
                            },
                        ),
                        html.Div(
                            create_dropdown_components(),
                            style={"marginBottom": 20},
                        ),
                    ],
                    style={
                        "marginBottom": 30,
                        "padding": "20px",
                        "backgroundColor": "#ffffff",
                        "borderRadius": "10px",
                        "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
                    },
                ),
                # Data info
                html.Div(id="data-info", style={"marginBottom": 30}),
                # PCA and Correlation Matrix Section
                html.Div(
                    [
                        html.H3(
                            "Overview Analysis",
                            style={
                                "marginBottom": 20,
                                "color": "#2c3e50",
                                "fontFamily": "'Segoe UI', Arial, sans-serif",
                            },
                        ),
                        # Sample Size Input
                        html.Div(
                            [
                                html.Label("Sample Size (for performance):"),
                                dcc.Input(
                                    id="pca-sample-size",
                                    type="number",
                                    value=1000,
                                    min=100,
                                    max=10000,
                                    step=100,
                                    style={"marginLeft": 10, "width": "100px"},
                                ),
                                html.Small(
                                    " (Randomly samples this many points for PCA and t-SNE)",
                                    style={"marginLeft": 10, "color": "gray"},
                                ),
                            ],
                            style={"marginBottom": 15},
                        ),
                        # t-SNE Perplexity Input
                        html.Div(
                            [
                                html.Label("t-SNE Perplexity:"),
                                dcc.Input(
                                    id="overview-tsne-perplexity",
                                    type="number",
                                    value=30,
                                    min=5,
                                    max=50,
                                    step=5,
                                    style={"marginLeft": 10, "width": "100px"},
                                ),
                                html.Small(
                                    " (Higher values preserve global structure)",
                                    style={"marginLeft": 10, "color": "gray"},
                                ),
                            ],
                            style={"marginBottom": 15},
                        ),
                        dcc.Tabs(
                            id="overview-tabs",
                            value="correlation",
                            children=[
                                dcc.Tab(
                                    label="Correlation Matrix", value="correlation"
                                ),
                                dcc.Tab(label="PCA", value="pca"),
                                dcc.Tab(label="t-SNE", value="tsne"),
                            ],
                        ),
                        dcc.Loading(
                            id="loading-overview",
                            type="default",
                            children=[
                                html.Div(id="overview-content", style={"marginTop": 20})
                            ],
                            style={"minHeight": "400px"},
                            color="#1f77b4",
                        ),
                    ],
                    style={
                        "marginBottom": 40,
                        "padding": "20px",
                        "backgroundColor": "#ffffff",
                        "borderRadius": "10px",
                        "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
                    },
                ),
                # Feature Selection Section
                html.Div(
                    [
                        html.H3(
                            "Feature Analysis",
                            style={
                                "marginBottom": 20,
                                "color": "#2c3e50",
                                "fontFamily": "'Segoe UI', Arial, sans-serif",
                            },
                        ),
                        html.Div(
                            [
                                html.Label("Select Feature:"),
                                dcc.Dropdown(
                                    id="feature-dropdown", style={"marginBottom": 20}
                                ),
                            ],
                            style={"marginBottom": 20},
                        ),
                        # Tabs for feature-specific visualizations
                        dcc.Tabs(
                            id="feature-tabs",
                            value="distribution",
                            children=[
                                dcc.Tab(label="Distribution", value="distribution"),
                                dcc.Tab(label="Statistics", value="stats"),
                            ],
                        ),
                        dcc.Loading(
                            id="loading-feature",
                            type="default",
                            children=[
                                html.Div(id="feature-content", style={"marginTop": 20})
                            ],
                            style={"minHeight": "300px"},
                            color="#1f77b4",
                        ),
                    ],
                    style={
                        "marginBottom": 40,
                        "padding": "20px",
                        "backgroundColor": "#ffffff",
                        "borderRadius": "10px",
                        "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
                    },
                ),
                # Combined Dataset Analysis Section
                html.Div(
                    [
                        html.H3(
                            "Dataset-Wide Analysis",
                            style={
                                "marginBottom": 20,
                                "color": "#2c3e50",
                                "fontWeight": "bold",
                                "fontFamily": "'Segoe UI', Arial, sans-serif",
                            },
                        ),
                        html.P(
                            "Analyze all slides combined to understand global patterns",
                            style={"color": "#7f8c8d", "marginBottom": 20},
                        ),
                        # Sample size control for combined analysis
                        html.Div(
                            [
                                html.Label("Samples per Slide:"),
                                dcc.Input(
                                    id="combined-sample-size",
                                    type="number",
                                    value=1000,
                                    min=100,
                                    max=5000,
                                    step=100,
                                    style={"marginLeft": 10, "width": "100px"},
                                ),
                                html.Small(
                                    " (Number of cells to sample from each slide)",
                                    style={"marginLeft": 10, "color": "gray"},
                                ),
                            ],
                            style={"marginBottom": 15},
                        ),
                        # t-SNE Perplexity for combined
                        html.Div(
                            [
                                html.Label("t-SNE Perplexity:"),
                                dcc.Input(
                                    id="combined-tsne-perplexity",
                                    type="number",
                                    value=30,
                                    min=5,
                                    max=50,
                                    step=5,
                                    style={"marginLeft": 10, "width": "100px"},
                                ),
                            ],
                            style={"marginBottom": 20},
                        ),
                        dcc.Tabs(
                            id="combined-tabs",
                            value="combined-pca",
                            children=[
                                dcc.Tab(label="PCA", value="combined-pca"),
                                dcc.Tab(label="t-SNE", value="combined-tsne"),
                                dcc.Tab(
                                    label="Cell Type Distribution",
                                    value="combined-celltype-dist",
                                ),
                            ],
                        ),
                        dcc.Loading(
                            id="loading-combined",
                            type="default",
                            children=[
                                html.Div(id="combined-content", style={"marginTop": 20})
                            ],
                            style={"minHeight": "400px"},
                            color="#1f77b4",
                        ),
                    ],
                    id="combined-analysis-section",
                    style={
                        "marginBottom": 40,
                        "padding": "20px",
                        "backgroundColor": "#e8f4f8",
                        "borderRadius": "10px",
                    },
                ),
                # Combined Cell Type Comparison Section
                html.Div(
                    [
                        html.H3(
                            "Combined Cell Type Comparison",
                            style={
                                "marginBottom": 20,
                                "color": "#2c3e50",
                                "fontWeight": "bold",
                                "fontFamily": "'Segoe UI', Arial, sans-serif",
                            },
                        ),
                        html.P(
                            "Compare features across cell types using all slides combined",
                            style={"color": "#7f8c8d", "marginBottom": 20},
                        ),
                        # Sample size control for combined cell type comparison
                        html.Div(
                            [
                                html.Label("Samples per Slide:"),
                                dcc.Input(
                                    id="combined-comparison-sample-size",
                                    type="number",
                                    value=1000,
                                    min=100,
                                    max=5000,
                                    step=100,
                                    style={"marginLeft": 10, "width": "100px"},
                                ),
                                html.Small(
                                    " (Number of cells to sample from each slide)",
                                    style={"marginLeft": 10, "color": "gray"},
                                ),
                            ],
                            style={"marginBottom": 15},
                        ),
                        # t-SNE Perplexity
                        html.Div(
                            [
                                html.Label("t-SNE Perplexity:"),
                                dcc.Input(
                                    id="combined-comparison-tsne-perplexity",
                                    type="number",
                                    value=30,
                                    min=5,
                                    max=50,
                                    step=5,
                                    style={"marginLeft": 10, "width": "100px"},
                                ),
                            ],
                            style={"marginBottom": 15},
                        ),
                        # Feature selector for distribution comparison
                        html.Div(
                            [
                                html.Label(
                                    "Select Feature for Distribution Comparison:"
                                ),
                                dcc.Dropdown(
                                    id="combined-comparison-feature-dropdown",
                                    style={"marginBottom": 20},
                                ),
                            ],
                            style={"marginBottom": 20},
                        ),
                        # Cell type selector for JS divergence table
                        html.Div(
                            [
                                html.Label(
                                    "Select Reference Cell Type for JS Divergence:"
                                ),
                                dcc.Dropdown(
                                    id="combined-js-celltype-dropdown",
                                    style={"marginBottom": 20},
                                ),
                            ],
                            style={"marginBottom": 20},
                        ),
                        dcc.Tabs(
                            id="combined-comparison-tabs",
                            value="combined-dist-comparison",
                            children=[
                                dcc.Tab(
                                    label="Distribution Comparison",
                                    value="combined-dist-comparison",
                                ),
                                dcc.Tab(
                                    label="PCA by Cell Type",
                                    value="combined-pca-celltype",
                                ),
                                dcc.Tab(
                                    label="t-SNE by Cell Type",
                                    value="combined-tsne-celltype",
                                ),
                                dcc.Tab(
                                    label="JS Divergence Table",
                                    value="combined-js-divergence",
                                ),
                            ],
                        ),
                        dcc.Loading(
                            id="loading-combined-comparison",
                            type="default",
                            children=[
                                html.Div(
                                    id="combined-comparison-content",
                                    style={"marginTop": 20},
                                )
                            ],
                            style={"minHeight": "400px"},
                            color="#1f77b4",
                        ),
                    ],
                    id="combined-cell-type-comparison-section",
                    style={
                        "marginBottom": 40,
                        "padding": "20px",
                        "backgroundColor": "#e8f4f8",
                        "borderRadius": "10px",
                    },
                ),
                # Cell Type Comparison Section
                html.Div(
                    [
                        html.H3(
                            "Cell Type Comparison",
                            style={
                                "marginBottom": 20,
                                "color": "#2c3e50",
                                "fontWeight": "bold",
                                "fontFamily": "'Segoe UI', Arial, sans-serif",
                            },
                        ),
                        html.P(
                            "Compare features across different cell types",
                            style={"color": "#7f8c8d", "marginBottom": 20},
                        ),
                        # Sample size control
                        html.Div(
                            [
                                html.Label("Sample Size for Dimensionality Reduction:"),
                                dcc.Input(
                                    id="comparison-sample-size",
                                    type="number",
                                    value=1000,
                                    min=100,
                                    max=10000,
                                    step=100,
                                    style={"marginLeft": 10, "width": "100px"},
                                ),
                                html.Small(
                                    " (Affects PCA and t-SNE plots)",
                                    style={"marginLeft": 10, "color": "gray"},
                                ),
                            ],
                            style={"marginBottom": 15},
                        ),
                        # t-SNE perplexity control
                        html.Div(
                            [
                                html.Label("t-SNE Perplexity:"),
                                dcc.Input(
                                    id="tsne-perplexity",
                                    type="number",
                                    value=30,
                                    min=5,
                                    max=50,
                                    step=5,
                                    style={"marginLeft": 10, "width": "100px"},
                                ),
                                html.Small(
                                    " (Higher values preserve global structure)",
                                    style={"marginLeft": 10, "color": "gray"},
                                ),
                            ],
                            style={"marginBottom": 15},
                        ),
                        # Feature selector for distribution comparison
                        html.Div(
                            [
                                html.Label(
                                    "Select Feature for Distribution Comparison:"
                                ),
                                dcc.Dropdown(
                                    id="comparison-feature-dropdown",
                                    style={"marginBottom": 20},
                                ),
                            ],
                            style={"marginBottom": 20},
                        ),
                        # Cell type selector for JS divergence table
                        html.Div(
                            [
                                html.Label(
                                    "Select Reference Cell Type for Divergence Analysis:"
                                ),
                                dcc.Dropdown(
                                    id="js-celltype-dropdown",
                                    style={"marginBottom": 20},
                                ),
                            ],
                            style={"marginBottom": 20},
                        ),
                        dcc.Tabs(
                            id="comparison-tabs",
                            value="dist-comparison",
                            children=[
                                dcc.Tab(
                                    label="Distribution Comparison",
                                    value="dist-comparison",
                                ),
                                dcc.Tab(label="PCA by Cell Type", value="pca-celltype"),
                                dcc.Tab(
                                    label="t-SNE by Cell Type", value="tsne-celltype"
                                ),
                                dcc.Tab(
                                    label="JS Divergence Table", value="js-divergence"
                                ),
                            ],
                        ),
                        dcc.Loading(
                            id="loading-comparison",
                            type="default",
                            children=[
                                html.Div(
                                    id="comparison-content", style={"marginTop": 20}
                                )
                            ],
                            style={"minHeight": "400px"},
                            color="#1f77b4",
                        ),
                    ],
                    id="cell-type-comparison-section",
                    style={
                        "marginBottom": 40,
                        "padding": "20px",
                        "backgroundColor": "#f8f9fa",
                        "borderRadius": "10px",
                    },
                ),
                # Hidden div to store directory structure
                html.Div(id="directory-structure", style={"display": "none"}),
            ],
            style={
                "padding": "20px",
                "maxWidth": "1400px",
                "margin": "0 auto",
                "fontFamily": "'Segoe UI', 'Helvetica Neue', Arial, sans-serif",
                "backgroundColor": "#f5f7fa",
            },
        )

        # Callback to control visibility of sections based on analysis mode
        @app.callback(  # type: ignore
            Output("combined-analysis-section", "style"),
            Output("combined-cell-type-comparison-section", "style"),
            Output("slide-selection-container", "style"),
            Output("cell-type-comparison-section", "style"),
            Input("analysis-mode", "value"),
        )
        def toggle_analysis_mode(  # type: ignore
            mode: str,
        ) -> tuple[
            dict[str, str | int],
            dict[str, str | int],
            dict[str, str | int],
            dict[str, str | int],
        ]:
            base_combined_style = STYLES["section_combined"]
            base_slide_style = STYLES["section_slide"]
            base_comparison_style = STYLES["section_comparison"]

            if mode == "combined":
                # Show combined sections, hide slide-specific sections
                return (
                    base_combined_style,
                    base_combined_style,
                    {**base_slide_style, "display": "none"},
                    {**base_comparison_style, "display": "none"},
                )
            else:  # single
                # Hide combined sections, show slide-specific sections
                return (
                    {**base_combined_style, "display": "none"},
                    {**base_combined_style, "display": "none"},
                    base_slide_style,
                    base_comparison_style,
                )

        # Callback for combined analysis content
        @app.callback(  # type: ignore
            Output("combined-content", "children"),
            Input("combined-tabs", "value"),
            Input("combined-sample-size", "value"),
            Input("combined-tsne-perplexity", "value"),
            *[
                Input(f"dropdown-level-{i}", "value")
                for i in range(self.max_dropdown_levels)
            ],
        )
        def update_combined_content(  # type: ignore
            active_tab: str,
            sample_size: int,
            perplexity: int,
            *selected_values: str | None,
        ):
            # Build path from selected values
            current_path = self._build_path_from_values(*selected_values)

            if not current_path:
                return html.Div(
                    "Please select a feature path to view combined analysis.",
                    style=STYLES["warning"],
                )

            try:
                # Validate inputs
                samples_per_slide = self._validate_positive_int(sample_size, 1000)
                tsne_perp = self._validate_positive_int(perplexity, 30)

                if active_tab == "combined-pca":
                    data = self._prepare_combined_data(
                        available_slides, current_path, samples_per_slide
                    )
                    fig = self._create_combined_pca_plot(
                        data["features"], data["slide_labels"], data["slides"]
                    )
                    return dcc.Graph(figure=fig)

                elif active_tab == "combined-tsne":
                    data = self._prepare_combined_data(
                        available_slides, current_path, samples_per_slide
                    )
                    fig = self._create_combined_tsne_plot(
                        data["features"],
                        data["slide_labels"],
                        data["slides"],
                        tsne_perp,
                    )
                    return dcc.Graph(figure=fig)

                elif active_tab == "combined-celltype-dist":
                    # Use all cells for distribution (no sampling)
                    data = self._prepare_combined_data_with_cell_types(
                        available_slides, current_path, max_samples_per_slide=None
                    )
                    if data["cell_types"] is None:
                        return html.Div(
                            [
                                html.H4("Cell Type Information Not Available"),
                                html.P(
                                    "Cell type data could not be loaded from the slides."
                                ),
                            ],
                            style={
                                "padding": "20px",
                                "backgroundColor": "#f8d7da",
                                "border": "1px solid #f5c6cb",
                                "borderRadius": "5px",
                                "color": "#721c24",
                            },
                        )
                    fig = self._create_combined_cell_type_distribution(
                        data["cell_types"],
                        data["cell_type_names"],
                        data["slide_labels"],
                        data["slides"],
                    )
                    return dcc.Graph(figure=fig)

            except Exception as e:
                logger.error(f"Error generating combined visualization: {e}")
                import traceback

                logger.error(traceback.format_exc())
                return self._create_error_message(
                    "Error",
                    f"Failed to generate combined visualization: {str(e)}",
                )

            return html.Div("Select a tab to view combined analysis.")

        # Callback for updating combined comparison feature dropdown
        @app.callback(  # type: ignore
            Output("combined-comparison-feature-dropdown", "options"),
            Output("combined-comparison-feature-dropdown", "value"),
            Input("analysis-mode", "value"),
            *[
                Input(f"dropdown-level-{i}", "value")
                for i in range(self.max_dropdown_levels)
            ],
        )
        def update_combined_comparison_feature_dropdown(  # type: ignore
            analysis_mode: str,
            *selected_values: str | None,
        ) -> tuple[list[Any], str | None]:
            # Only populate in combined mode
            if analysis_mode != "combined":
                return [], None

            # Build path from selected values
            current_path: list[str] = []
            for value in selected_values:
                if value is not None:
                    current_path.append(value)
                else:
                    break

            if not current_path:
                return [], None

            try:
                data = self._prepare_combined_data_with_cell_types(
                    available_slides,
                    current_path,
                    100,  # Just load a small sample to get feature names
                )
                if data["cell_types"] is None or len(data["features"]) == 0:
                    return [], None
                feature_names = data["feature_names"]
                options = [{"label": f, "value": f} for f in feature_names]
                value = feature_names[0] if feature_names else None
                return options, value
            except Exception as e:
                logger.error(f"Error loading features for combined comparison: {e}")
                return [], None

        # Callback for updating combined JS divergence cell type dropdown
        @app.callback(  # type: ignore
            Output("combined-js-celltype-dropdown", "options"),
            Output("combined-js-celltype-dropdown", "value"),
            Input("analysis-mode", "value"),
            Input("combined-comparison-tabs", "value"),
            *[
                Input(f"dropdown-level-{i}", "value")
                for i in range(self.max_dropdown_levels)
            ],
        )
        def update_combined_js_celltype_dropdown(  # type: ignore
            analysis_mode: str,
            active_tab: str,
            *selected_values: str | None,
        ) -> tuple[list[Any], str | None]:
            # Only populate in combined mode
            if analysis_mode != "combined":
                return [], None

            # Build path from selected values
            current_path: list[str] = []
            for value in selected_values:
                if value is not None:
                    current_path.append(value)
                else:
                    break

            if not current_path:
                logger.info("Combined JS celltype dropdown: No path selected")
                return [], None

            try:
                logger.info(
                    f"Combined JS celltype dropdown: Loading data for path {current_path}"
                )
                # Use None to load all cells (not just a sample) to ensure we get all cell types
                data = self._prepare_combined_data_with_cell_types(
                    available_slides,
                    current_path,
                    max_samples_per_slide=None,  # Load all cells to get all cell types
                )

                logger.info(
                    f"Combined JS celltype dropdown: Loaded {len(data['features'])} cells"
                )

                if data["cell_types"] is None:
                    logger.warning("Combined JS celltype dropdown: No cell types found")
                    return [], None

                if len(data["features"]) == 0:
                    logger.warning("Combined JS celltype dropdown: No features found")
                    return [], None

                cell_type_names = data["cell_type_names"]
                if cell_type_names is None or len(cell_type_names) == 0:
                    logger.warning(
                        "Combined JS celltype dropdown: No cell type names found"
                    )
                    return [], None

                unique_types = sorted(set(data["cell_types"]))
                logger.info(
                    f"Combined JS celltype dropdown: Found {len(unique_types)} unique cell types"
                )

                options = [
                    {"label": cell_type_names[i], "value": i} for i in unique_types
                ]
                value = options[0]["value"] if options else None
                return options, value
            except Exception as e:
                logger.error(
                    f"Error loading cell types for combined JS divergence: {e}"
                )
                import traceback

                logger.error(traceback.format_exc())
                return [], None

        # Callback for updating combined cell type comparison content
        @app.callback(  # type: ignore
            Output("combined-comparison-content", "children"),
            Input("combined-comparison-tabs", "value"),
            Input("combined-comparison-feature-dropdown", "value"),
            Input("combined-js-celltype-dropdown", "value"),
            Input("combined-comparison-sample-size", "value"),
            Input("combined-comparison-tsne-perplexity", "value"),
            *[
                Input(f"dropdown-level-{i}", "value")
                for i in range(self.max_dropdown_levels)
            ],
        )
        def update_combined_comparison_content(  # type: ignore
            active_tab: str,
            selected_feature: str,
            selected_celltype: int,
            sample_size: int,
            perplexity: int,
            *selected_values: str | None,
        ):
            # Build path from selected values
            current_path: list[str] = []
            for value in selected_values:
                if value is not None:
                    current_path.append(value)
                else:
                    break

            if not current_path:
                return html.Div(
                    "Please select a feature path to enable combined cell type comparison.",
                    style={
                        "padding": "20px",
                        "backgroundColor": "#fff3cd",
                        "border": "1px solid #ffc107",
                        "borderRadius": "5px",
                        "color": "#856404",
                    },
                )

            try:
                # Validate inputs
                samples_per_slide = (
                    sample_size if sample_size and sample_size > 0 else 1000
                )
                tsne_perp = perplexity if perplexity and perplexity > 0 else 30

                if active_tab == "combined-dist-comparison":
                    # Use all cells for distribution comparison (no sampling)
                    data = self._prepare_combined_data_with_cell_types(
                        available_slides, current_path, max_samples_per_slide=None
                    )

                    # Check if cell types are available
                    if data["cell_types"] is None or data["cell_type_names"] is None:
                        return html.Div(
                            [
                                html.H4(
                                    "Cell Type Information Not Available",
                                    style={"marginBottom": "10px"},
                                ),
                                html.P(
                                    "Cell type data could not be loaded from the slides. "
                                    "Make sure cell detection data exists for the selected feature extraction path."
                                ),
                            ],
                            style={
                                "padding": "20px",
                                "backgroundColor": "#f8d7da",
                                "border": "1px solid #f5c6cb",
                                "borderRadius": "5px",
                                "color": "#721c24",
                            },
                        )

                    if not selected_feature:
                        return html.Div("Please select a feature.")
                    fig = self._create_combined_distribution_comparison(
                        data["df"],
                        selected_feature,
                        data["cell_types"],
                        data["cell_type_names"],
                    )
                    return dcc.Graph(figure=fig)

                elif active_tab == "combined-pca-celltype":
                    # Use sampled data for PCA
                    data = self._prepare_combined_data_with_cell_types(
                        available_slides, current_path, samples_per_slide
                    )

                    # Check if cell types are available
                    if data["cell_types"] is None or data["cell_type_names"] is None:
                        return html.Div(
                            [
                                html.H4(
                                    "Cell Type Information Not Available",
                                    style={"marginBottom": "10px"},
                                ),
                                html.P(
                                    "Cell type data could not be loaded from the slides. "
                                    "Make sure cell detection data exists for the selected feature extraction path."
                                ),
                            ],
                            style={
                                "padding": "20px",
                                "backgroundColor": "#f8d7da",
                                "border": "1px solid #f5c6cb",
                                "borderRadius": "5px",
                                "color": "#721c24",
                            },
                        )

                    fig = self._create_combined_pca_by_cell_type(
                        data["features"],
                        data["cell_types"],
                        data["cell_type_names"],
                    )
                    return dcc.Graph(figure=fig)

                elif active_tab == "combined-tsne-celltype":
                    # Use sampled data for t-SNE
                    data = self._prepare_combined_data_with_cell_types(
                        available_slides, current_path, samples_per_slide
                    )

                    # Check if cell types are available
                    if data["cell_types"] is None or data["cell_type_names"] is None:
                        return html.Div(
                            [
                                html.H4(
                                    "Cell Type Information Not Available",
                                    style={"marginBottom": "10px"},
                                ),
                                html.P(
                                    "Cell type data could not be loaded from the slides. "
                                    "Make sure cell detection data exists for the selected feature extraction path."
                                ),
                            ],
                            style={
                                "padding": "20px",
                                "backgroundColor": "#f8d7da",
                                "border": "1px solid #f5c6cb",
                                "borderRadius": "5px",
                                "color": "#721c24",
                            },
                        )

                    fig = self._create_combined_tsne_by_cell_type(
                        data["features"],
                        data["cell_types"],
                        data["cell_type_names"],
                        tsne_perp,
                    )
                    return dcc.Graph(figure=fig)

                elif active_tab == "combined-js-divergence":
                    if selected_celltype is None:  # type: ignore
                        return html.Div(
                            "Please select a reference cell type.",
                            style={
                                "padding": "20px",
                                "textAlign": "center",
                                "color": "#7f8c8d",
                            },
                        )

                    # Use all cells for JS divergence calculation
                    data = self._prepare_combined_data_with_cell_types(
                        available_slides, current_path, max_samples_per_slide=None
                    )

                    # Check if cell types are available
                    if data["cell_types"] is None or data["cell_type_names"] is None:
                        return html.Div(
                            [
                                html.H4(
                                    "Cell Type Information Not Available",
                                    style={"marginBottom": "10px"},
                                ),
                                html.P(
                                    "Cell type data could not be loaded from the slides. "
                                    "Make sure cell detection data exists for the selected feature extraction path."
                                ),
                            ],
                            style={
                                "padding": "20px",
                                "backgroundColor": "#f8d7da",
                                "border": "1px solid #f5c6cb",
                                "borderRadius": "5px",
                                "color": "#721c24",
                            },
                        )

                    js_df = self._calculate_js_divergence_table(
                        data["df"],
                        data["cell_types"],
                        data["cell_type_names"],
                        selected_celltype,
                    )

                    # Get reference cell type name
                    ref_name = data["cell_type_names"].get(selected_celltype, "Unknown")

                    # Prepare data for the table
                    table_data = js_df.reset_index().to_dict("records")  # type: ignore
                    table_columns: list[dict[str, str | dict[str, str]]] = [
                        {"name": "Feature", "id": "Feature"}
                    ]
                    for col in js_df.columns:
                        table_columns.append(
                            {
                                "name": col,
                                "id": col,
                                "type": "numeric",
                                "format": {"specifier": ".4f"},
                            }
                        )

                    # Build style_data_conditional list
                    style_conditions: list[dict[str, Any]] = [
                        {
                            "if": {"row_index": "odd"},
                            "backgroundColor": "#f9f9f9",
                        },
                        {
                            "if": {"column_id": "Feature"},
                            "fontWeight": "500",
                            "backgroundColor": "#ecf0f1",
                        },
                    ]

                    # Add color coding for divergence values (low = green, medium = yellow, high = red)
                    for col in js_df.columns:
                        # Low divergence (< 0.1) - green
                        style_conditions.append(
                            {
                                "if": {
                                    "filter_query": f"{{{col}}} < 0.1",
                                    "column_id": col,
                                },
                                "backgroundColor": "#d4edda",
                                "color": "#155724",
                            }
                        )
                        # Medium divergence (0.1 - 0.3) - yellow
                        style_conditions.append(
                            {
                                "if": {
                                    "filter_query": f"{{{col}}} >= 0.1 && {{{col}}} < 0.3",
                                    "column_id": col,
                                },
                                "backgroundColor": "#fff3cd",
                                "color": "#856404",
                            }
                        )
                        # High divergence (>= 0.3) - red
                        style_conditions.append(
                            {
                                "if": {
                                    "filter_query": f"{{{col}}} >= 0.3",
                                    "column_id": col,
                                },
                                "backgroundColor": "#f8d7da",
                                "color": "#721c24",
                            }
                        )

                    return html.Div(
                        [
                            html.H4(
                                f"Jensen-Shannon Divergence: {ref_name} vs Other Cell Types (Combined Dataset)",
                                style={
                                    "marginBottom": "20px",
                                    "fontFamily": "'Segoe UI', Arial, sans-serif",
                                    "color": "#2c3e50",
                                },
                            ),
                            html.P(
                                f"Values represent the Jensen-Shannon divergence between the distribution of each feature in {ref_name} cells and other cell types across all slides. "
                                "Lower values indicate more similar distributions (0 = identical, 1 = completely different).",
                                style={
                                    "marginBottom": "20px",
                                    "fontFamily": "'Segoe UI', Arial, sans-serif",
                                    "color": "#7f8c8d",
                                    "fontSize": "14px",
                                },
                            ),
                            dash_table.DataTable(
                                data=table_data,  # type: ignore
                                columns=table_columns,  # type: ignore
                                style_table={
                                    "overflowX": "auto",
                                    "maxHeight": "600px",
                                    "overflowY": "auto",
                                },
                                style_cell={
                                    "textAlign": "left",
                                    "padding": "10px",
                                    "fontFamily": "'Segoe UI', Arial, sans-serif",
                                    "fontSize": "13px",
                                },
                                style_header={
                                    "backgroundColor": "#34495e",
                                    "color": "white",
                                    "fontWeight": "bold",
                                    "textAlign": "left",
                                    "padding": "12px",
                                    "fontFamily": "'Segoe UI', Arial, sans-serif",
                                },
                                style_data_conditional=style_conditions,  # type: ignore
                                page_size=20,
                                sort_action="native",
                                filter_action="native",
                            ),
                        ],
                        style={"padding": "20px"},
                    )

            except Exception as e:
                logger.error(f"Error generating combined cell type comparison: {e}")
                import traceback

                logger.error(traceback.format_exc())
                return html.Div(
                    [
                        html.H4("Error"),
                        html.P(
                            f"Failed to generate combined cell type comparison: {str(e)}"
                        ),
                    ],
                    style={
                        "backgroundColor": "#ffcccc",
                        "padding": 15,
                        "borderRadius": 5,
                    },
                )

            return html.Div(
                "Select a tab above to view combined cell type comparisons.",
                style={
                    "padding": "20px",
                    "textAlign": "center",
                    "color": "#7f8c8d",
                },
            )

        # Initialize the first dropdown
        @app.callback(  # type: ignore
            [
                Output(f"dropdown-level-{i}", "options")
                for i in range(self.max_dropdown_levels)
            ]
            + [
                Output(f"dropdown-level-{i}", "value")
                for i in range(self.max_dropdown_levels)
            ]
            + [
                Output(f"dropdown-container-{i}", "style")
                for i in range(self.max_dropdown_levels)
            ],
            [Input("slide-dropdown", "value")]
            + [
                Input(f"dropdown-level-{i}", "value")
                for i in range(self.max_dropdown_levels)
            ],
        )
        def update_dropdowns(  # type: ignore
            selected_slide: str | None, *selected_values: str | None
        ) -> list[dict[str, str] | str | list[dict[str, str]] | None]:
            # If no slide is selected, return empty
            if selected_slide is None:
                return cast(
                    list[dict[str, str] | str | list[dict[str, str]] | None],
                    (
                        [[] for _ in range(self.max_dropdown_levels)]
                        + [None for _ in range(self.max_dropdown_levels)]
                        + [{"display": "none"} for _ in range(self.max_dropdown_levels)]
                    ),
                )

            # Explore directory structure for the selected slide
            feature_extraction_path = (
                self.config.dataset / selected_slide / "feature_extraction"
            )
            directory_structure = self._explore_directory(feature_extraction_path)
            # Prepare outputs
            options_outputs: list[list[dict[str, str]]] = [
                [] for _ in range(self.max_dropdown_levels)
            ]
            value_outputs: list[str | None] = [
                None for _ in range(self.max_dropdown_levels)
            ]
            style_outputs: list[dict[str, str]] = []

            # Build path from selected values
            current_path: list[str] = []
            for _, value in enumerate(selected_values):
                if value is not None:
                    current_path.append(value)
                else:
                    break

            # Update options for each level
            for level in range(self.max_dropdown_levels):
                if level == 0:
                    # First level: show top-level directories
                    options = self._get_available_options_at_level(
                        directory_structure, []
                    )
                    if options:
                        options_outputs[level] = [
                            {"label": opt, "value": opt} for opt in options
                        ]
                        if level < len(current_path):
                            value_outputs[level] = current_path[level]
                        # Remove the auto-selection logic for first dropdown
                        # Keep value_outputs[level] as None if no path is selected
                    style_outputs.append(
                        {
                            "width": f"{90 // min(3, self.max_dropdown_levels)}%",
                            "display": "inline-block",
                            "marginRight": "2%",
                        }
                    )
                else:
                    # Subsequent levels: show options based on current path
                    if level <= len(current_path):
                        path_to_check = current_path[:level]
                        options = self._get_available_options_at_level(
                            directory_structure, path_to_check
                        )

                        if options:
                            options_outputs[level] = [
                                {"label": opt, "value": opt} for opt in options
                            ]
                            if level < len(current_path):
                                value_outputs[level] = current_path[level]
                            style_outputs.append(
                                {
                                    "width": f"{90 // min(3, self.max_dropdown_levels)}%",
                                    "display": "inline-block",
                                    "marginRight": "2%",
                                }
                            )
                        else:
                            # No more options available, hide this dropdown
                            style_outputs.append({"display": "none"})
                    else:
                        # Hide this dropdown
                        style_outputs.append({"display": "none"})

            return options_outputs + value_outputs + style_outputs

        # Callback for updating feature dropdown
        @app.callback(  # type: ignore
            Output("feature-dropdown", "options"),
            Output("feature-dropdown", "value"),
            [Input("slide-dropdown", "value")]
            + [
                Input(f"dropdown-level-{i}", "value")
                for i in range(self.max_dropdown_levels)
            ],
        )
        def update_feature_dropdown(  # type: ignore
            selected_slide: str | None,
            *selected_values: str | None,
        ) -> tuple[list[Any], str | None]:
            if selected_slide is None:
                return [], None

            # Explore directory structure for the selected slide
            feature_extraction_path = (
                self.config.dataset / selected_slide / "feature_extraction"
            )
            directory_structure = self._explore_directory(feature_extraction_path)

            # Build path from selected values
            current_path: list[str] = []
            for value in selected_values:
                if value is not None:
                    current_path.append(value)
                else:
                    break

            if current_path and self._can_load_features(
                directory_structure, current_path
            ):
                try:
                    data = self._prepare_data(selected_slide, current_path)
                    feature_names = data["feature_names"]
                    options = [{"label": f, "value": f} for f in feature_names]
                    value = feature_names[0] if feature_names else None
                    return options, value
                except Exception as e:
                    print(f"Error loading features: {e}")
                    return [], None
            return [], None

        # Callback for updating data info
        @app.callback(  # type: ignore
            Output("data-info", "children"),
            [Input("slide-dropdown", "value")]
            + [
                Input(f"dropdown-level-{i}", "value")
                for i in range(self.max_dropdown_levels)
            ],
        )
        def update_data_info(selected_slide: str | None, *selected_values: str | None):  # type: ignore
            if selected_slide is None:
                return html.Div()

            # Explore directory structure for the selected slide
            feature_extraction_path = (
                self.config.dataset / selected_slide / "feature_extraction"
            )
            directory_structure = self._explore_directory(feature_extraction_path)

            # Build path from selected values
            current_path: list[str] = []
            for value in selected_values:
                if value is not None:
                    current_path.append(value)
                else:
                    break

            if current_path and self._can_load_features(
                directory_structure, current_path
            ):
                try:
                    data = self._prepare_data(selected_slide, current_path)
                    shape = data["shape"]
                    return html.Div(
                        [
                            html.H4("Data Information"),
                            html.P(f"Shape: {shape[0]} cells × {shape[1]} features"),
                            html.P(f"Path: {' → '.join(current_path)}"),
                        ],
                        style={
                            "backgroundColor": "#f0f0f0",
                            "padding": 15,
                            "borderRadius": 5,
                        },
                    )
                except Exception as e:
                    return html.Div(
                        [html.H4("Error"), html.P(f"Failed to load data: {str(e)}")],
                        style={
                            "backgroundColor": "#ffcccc",
                            "padding": 15,
                            "borderRadius": 5,
                        },
                    )
            elif current_path:
                return html.Div(
                    [
                        html.H4("Path Selection"),
                        html.P(f"Current path: {' → '.join(current_path)}"),
                    ],
                    style={
                        "backgroundColor": "#fff3cd",
                        "padding": 15,
                        "borderRadius": 5,
                    },
                )
            return html.Div()

        # Callback for updating overview content (PCA, t-SNE, and Correlation Matrix)
        @app.callback(  # type: ignore
            Output("overview-content", "children"),
            Input("overview-tabs", "value"),
            Input("pca-sample-size", "value"),
            Input("overview-tsne-perplexity", "value"),
            Input("slide-dropdown", "value"),
            *[
                Input(f"dropdown-level-{i}", "value")
                for i in range(self.max_dropdown_levels)
            ],
        )
        def update_overview_content(  # type: ignore
            active_tab: str,
            pca_sample_size: int,
            tsne_perplexity: int,
            selected_slide: str | None,
            *selected_values: str | None,
        ):
            if selected_slide is None:
                return html.Div("Please select a slide.")

            # Explore directory structure for the selected slide
            feature_extraction_path = (
                self.config.dataset / selected_slide / "feature_extraction"
            )
            directory_structure = self._explore_directory(feature_extraction_path)

            # Build path from selected values
            current_path: list[str] = []
            for value in selected_values:
                if value is not None:
                    current_path.append(value)
                else:
                    break

            if not current_path or not self._can_load_features(
                directory_structure, current_path
            ):
                return html.Div("Please select a complete path to features.")

            try:
                data = self._prepare_data(selected_slide, current_path)
                df = data["df"]
                features = data["features"]
                feature_names = data["feature_names"]

                # Validate inputs
                sample_size = (
                    pca_sample_size if pca_sample_size and pca_sample_size > 0 else 1000
                )
                perplexity = (
                    tsne_perplexity if tsne_perplexity and tsne_perplexity > 0 else 30
                )

                if active_tab == "pca":
                    fig = self._create_pca_plot(features, feature_names, sample_size)
                    return html.Div(
                        dcc.Graph(figure=fig),
                        style={"display": "flex", "justifyContent": "center"},
                    )

                elif active_tab == "tsne":
                    fig = self._create_tsne_plot(features, sample_size, perplexity)
                    return html.Div(
                        dcc.Graph(figure=fig),
                        style={"display": "flex", "justifyContent": "center"},
                    )

                elif active_tab == "correlation":
                    fig = self._create_correlation_matrix(df, feature_names)
                    return html.Div(
                        dcc.Graph(figure=fig),
                        style={"display": "flex", "justifyContent": "center"},
                    )

            except Exception as e:
                return html.Div(
                    [
                        html.H4("Error"),
                        html.P(f"Failed to generate visualization: {str(e)}"),
                    ],
                    style={
                        "backgroundColor": "#ffcccc",
                        "padding": 15,
                        "borderRadius": 5,
                    },
                )

            return html.Div("Select a tab to view visualizations.")

        # Callback for updating feature content (Distribution and Statistics)
        @app.callback(  # type: ignore
            Output("feature-content", "children"),
            Input("feature-tabs", "value"),
            Input("feature-dropdown", "value"),
            Input("slide-dropdown", "value"),
            *[
                Input(f"dropdown-level-{i}", "value")
                for i in range(self.max_dropdown_levels)
            ],
        )
        def update_feature_content(  # type: ignore
            active_tab: str,
            selected_feature: str,
            selected_slide: str | None,
            *selected_values: str | None,
        ):
            if selected_slide is None:
                return html.Div("Please select a slide.")

            # Explore directory structure for the selected slide
            feature_extraction_path = (
                self.config.dataset / selected_slide / "feature_extraction"
            )
            directory_structure = self._explore_directory(feature_extraction_path)

            # Build path from selected values
            current_path: list[str] = []
            for value in selected_values:
                if value is not None:
                    current_path.append(value)
                else:
                    break

            if not current_path or not self._can_load_features(
                directory_structure, current_path
            ):
                return html.Div("Please select a complete path to features.")

            try:
                data = self._prepare_data(selected_slide, current_path)
                df = data["df"]
                features = data["features"]
                feature_names = data["feature_names"]

                if active_tab == "distribution":
                    if selected_feature:
                        fig = self._create_distribution_plot(df, selected_feature)
                        return dcc.Graph(figure=fig)
                    else:
                        return html.Div("Please select a feature.")

                elif active_tab == "stats":
                    stats_dict = self._calculate_first_order_stats(features)
                    fig = self._create_stats_table(stats_dict, feature_names)
                    return dcc.Graph(figure=fig)

            except Exception as e:
                return html.Div(
                    [
                        html.H4("Error"),
                        html.P(f"Failed to generate visualization: {str(e)}"),
                    ],
                    style={
                        "backgroundColor": "#ffcccc",
                        "padding": 15,
                        "borderRadius": 5,
                    },
                )

            return html.Div("Select a tab to view visualizations.")

        # Callback for updating comparison feature dropdown
        @app.callback(  # type: ignore
            Output("comparison-feature-dropdown", "options"),
            Output("comparison-feature-dropdown", "value"),
            [Input("slide-dropdown", "value")]
            + [
                Input(f"dropdown-level-{i}", "value")
                for i in range(self.max_dropdown_levels)
            ],
        )
        def update_comparison_feature_dropdown(  # type: ignore
            selected_slide: str | None,
            *selected_values: str | None,
        ) -> tuple[list[Any], str | None]:
            if selected_slide is None:
                return [], None

            # Explore directory structure for the selected slide
            feature_extraction_path = (
                self.config.dataset / selected_slide / "feature_extraction"
            )
            directory_structure = self._explore_directory(feature_extraction_path)

            # Build path from selected values
            current_path: list[str] = []
            for value in selected_values:
                if value is not None:
                    current_path.append(value)
                else:
                    break

            if current_path and self._can_load_features(
                directory_structure, current_path
            ):
                try:
                    data = self._prepare_data_with_cell_types(
                        selected_slide, current_path
                    )
                    if data["cell_types"] is None:
                        return [], None
                    feature_names = data["feature_names"]
                    options = [{"label": f, "value": f} for f in feature_names]
                    value = feature_names[0] if feature_names else None
                    return options, value
                except Exception as e:
                    print(f"Error loading features for comparison: {e}")
                    return [], None
            return [], None

        # Callback for updating JS divergence cell type dropdown
        @app.callback(  # type: ignore
            Output("js-celltype-dropdown", "options"),
            Output("js-celltype-dropdown", "value"),
            [Input("slide-dropdown", "value")]
            + [
                Input(f"dropdown-level-{i}", "value")
                for i in range(self.max_dropdown_levels)
            ],
        )
        def update_js_celltype_dropdown(  # type: ignore
            selected_slide: str | None,
            *selected_values: str | None,
        ) -> tuple[list[Any], int | None]:
            if selected_slide is None:
                return [], None

            # Explore directory structure for the selected slide
            feature_extraction_path = (
                self.config.dataset / selected_slide / "feature_extraction"
            )
            directory_structure = self._explore_directory(feature_extraction_path)

            # Build path from selected values
            current_path: list[str] = []
            for value in selected_values:
                if value is not None:
                    current_path.append(value)
                else:
                    break

            if current_path and self._can_load_features(
                directory_structure, current_path
            ):
                try:
                    data = self._prepare_data_with_cell_types(
                        selected_slide, current_path
                    )
                    if data["cell_types"] is None or data["cell_type_names"] is None:
                        return [], None

                    cell_type_names = data["cell_type_names"]
                    # Exclude unknown type (0)
                    options = [
                        {"label": name, "value": ct}
                        for ct, name in sorted(cell_type_names.items())
                        if ct != 0
                    ]
                    # Default to first non-zero cell type
                    value = options[0]["value"] if options else None
                    return options, value
                except Exception as e:
                    print(f"Error loading cell types for JS divergence: {e}")
                    return [], None
            return [], None

        # Callback for updating cell type comparison content
        @app.callback(  # type: ignore
            Output("comparison-content", "children"),
            Input("comparison-tabs", "value"),
            Input("comparison-feature-dropdown", "value"),
            Input("comparison-sample-size", "value"),
            Input("tsne-perplexity", "value"),
            Input("js-celltype-dropdown", "value"),
            Input("slide-dropdown", "value"),
            *[
                Input(f"dropdown-level-{i}", "value")
                for i in range(self.max_dropdown_levels)
            ],
        )
        def update_comparison_content(  # type: ignore
            active_tab: str,
            selected_feature: str,
            sample_size: int,
            perplexity: int,
            selected_celltype: int,
            selected_slide: str | None,
            *selected_values: str | None,
        ):
            if selected_slide is None:
                return html.Div("Please select a slide.")

            # Explore directory structure for the selected slide
            feature_extraction_path = (
                self.config.dataset / selected_slide / "feature_extraction"
            )
            directory_structure = self._explore_directory(feature_extraction_path)

            # Build path from selected values
            current_path: list[str] = []
            for value in selected_values:
                if value is not None:
                    current_path.append(value)
                else:
                    break

            if not current_path or not self._can_load_features(
                directory_structure, current_path
            ):
                message = html.Div(
                    "Please select a complete path to features to enable cell type comparison.",
                    style={
                        "padding": "20px",
                        "backgroundColor": "#fff3cd",
                        "border": "1px solid #ffc107",
                        "borderRadius": "5px",
                        "color": "#856404",
                        "fontFamily": "'Segoe UI', Arial, sans-serif",
                    },
                )
                return message

            try:
                data = self._prepare_data_with_cell_types(selected_slide, current_path)

                # Check if cell types are available
                if data["cell_types"] is None or data["cell_type_names"] is None:
                    # Create debug info
                    debug_info = f"Path: {current_path}, Cell types: {data['cell_types'] is not None}, Cell type names: {data['cell_type_names'] is not None}"
                    logger.warning(f"Cell type comparison not available: {debug_info}")

                    message = html.Div(
                        [
                            html.H4(
                                "Cell Type Information Not Available",
                                style={
                                    "marginBottom": "10px",
                                    "fontFamily": "'Segoe UI', Arial, sans-serif",
                                },
                            ),
                            html.P(
                                "Cell type data was not found for this dataset. Check the browser console and terminal logs for details.",
                                style={"fontFamily": "'Segoe UI', Arial, sans-serif"},
                            ),
                            html.P(
                                f"Debug info: {debug_info}",
                                style={
                                    "fontFamily": "'Segoe UI', Arial, sans-serif",
                                    "fontSize": "12px",
                                    "marginTop": "10px",
                                },
                            ),
                            html.P(
                                "This may be because:",
                                style={
                                    "fontFamily": "'Segoe UI', Arial, sans-serif",
                                    "marginTop": "10px",
                                },
                            ),
                            html.Ul(
                                [
                                    html.Li(
                                        "Cell detection JSON file doesn't exist",
                                        style={
                                            "fontFamily": "'Segoe UI', Arial, sans-serif"
                                        },
                                    ),
                                    html.Li(
                                        "Cell indices are missing from the feature file",
                                        style={
                                            "fontFamily": "'Segoe UI', Arial, sans-serif"
                                        },
                                    ),
                                    html.Li(
                                        "Segmentation model path doesn't match",
                                        style={
                                            "fontFamily": "'Segoe UI', Arial, sans-serif"
                                        },
                                    ),
                                ]
                            ),
                        ],
                        style={
                            "padding": "20px",
                            "backgroundColor": "#f8d7da",
                            "border": "1px solid #f5c6cb",
                            "borderRadius": "5px",
                            "color": "#721c24",
                            "fontFamily": "'Segoe UI', Arial, sans-serif",
                        },
                    )
                    return message

                df = data["df"]
                features = data["features"]
                cell_types = data["cell_types"]
                cell_type_names = data["cell_type_names"]

                # Validate sample size
                sample_size = sample_size if sample_size and sample_size > 0 else 1000
                perplexity = perplexity if perplexity and perplexity > 0 else 30

                if active_tab == "dist-comparison":
                    if selected_feature:
                        fig = self._create_distribution_comparison_plot(
                            df, selected_feature, cell_types, cell_type_names
                        )
                        return dcc.Graph(figure=fig)
                    else:
                        return html.Div(
                            "Please select a feature from the dropdown above.",
                            style={
                                "padding": "20px",
                                "backgroundColor": "#e7f3ff",
                                "border": "1px solid #2196F3",
                                "borderRadius": "5px",
                                "color": "#0d47a1",
                                "fontFamily": "'Segoe UI', Arial, sans-serif",
                            },
                        )

                elif active_tab == "pca-celltype":
                    fig = self._create_pca_by_cell_type(
                        features, cell_types, cell_type_names, sample_size
                    )
                    return html.Div(
                        dcc.Graph(figure=fig),
                        style={"display": "flex", "justifyContent": "center"},
                    )

                elif active_tab == "tsne-celltype":
                    fig = self._create_tsne_by_cell_type(
                        features, cell_types, cell_type_names, sample_size, perplexity
                    )
                    return html.Div(
                        dcc.Graph(figure=fig),
                        style={"display": "flex", "justifyContent": "center"},
                    )

                elif active_tab == "js-divergence":
                    if selected_celltype is None:  # type: ignore
                        return html.Div(
                            "Please select a reference cell type from the dropdown above.",
                            style={
                                "padding": "20px",
                                "backgroundColor": "#e7f3ff",
                                "border": "1px solid #2196F3",
                                "borderRadius": "5px",
                                "color": "#0d47a1",
                                "fontFamily": "'Segoe UI', Arial, sans-serif",
                            },
                        )

                    # Calculate JS divergence table
                    js_df = self._calculate_js_divergence_table(
                        df, cell_types, cell_type_names, selected_celltype
                    )

                    # Get reference cell type name
                    ref_name = cell_type_names.get(selected_celltype, "Unknown")

                    # Create the table
                    # Prepare data for the table
                    table_data = js_df.reset_index().to_dict("records")  # type: ignore
                    table_columns: list[dict[str, str | dict[str, str]]] = [
                        {"name": "Feature", "id": "Feature"}
                    ]
                    for col in js_df.columns:
                        table_columns.append(
                            {
                                "name": col,
                                "id": col,
                                "type": "numeric",
                                "format": {"specifier": ".4f"},
                            }
                        )

                    # Build style_data_conditional list
                    style_conditions: list[dict[str, Any]] = [
                        {
                            "if": {"row_index": "odd"},
                            "backgroundColor": "#f9f9f9",
                        },
                        {
                            "if": {"column_id": "Feature"},
                            "fontWeight": "500",
                            "backgroundColor": "#ecf0f1",
                        },
                    ]

                    # Add color coding for divergence values (low = green, medium = yellow, high = red)
                    for col in js_df.columns:
                        # Low divergence (< 0.1) - green
                        style_conditions.append(
                            {
                                "if": {
                                    "filter_query": f"{{{col}}} < 0.1",
                                    "column_id": col,
                                },
                                "backgroundColor": "#d4edda",
                                "color": "#155724",
                            }
                        )
                        # Medium divergence (0.1 - 0.3) - yellow
                        style_conditions.append(
                            {
                                "if": {
                                    "filter_query": f"{{{col}}} >= 0.1 && {{{col}}} < 0.3",
                                    "column_id": col,
                                },
                                "backgroundColor": "#fff3cd",
                                "color": "#856404",
                            }
                        )
                        # High divergence (>= 0.3) - red
                        style_conditions.append(
                            {
                                "if": {
                                    "filter_query": f"{{{col}}} >= 0.3",
                                    "column_id": col,
                                },
                                "backgroundColor": "#f8d7da",
                                "color": "#721c24",
                            }
                        )

                    return html.Div(
                        [
                            html.H4(
                                f"Jensen-Shannon Divergence: {ref_name} vs Other Cell Types",
                                style={
                                    "marginBottom": "20px",
                                    "fontFamily": "'Segoe UI', Arial, sans-serif",
                                    "color": "#2c3e50",
                                },
                            ),
                            html.P(
                                f"Values represent the Jensen-Shannon divergence between the distribution of each feature in {ref_name} cells and other cell types. "
                                "Lower values indicate more similar distributions (0 = identical, 1 = completely different).",
                                style={
                                    "marginBottom": "20px",
                                    "fontFamily": "'Segoe UI', Arial, sans-serif",
                                    "color": "#7f8c8d",
                                    "fontSize": "14px",
                                },
                            ),
                            dash_table.DataTable(
                                data=table_data,  # type: ignore
                                columns=table_columns,  # type: ignore
                                style_table={
                                    "overflowX": "auto",
                                    "maxHeight": "600px",
                                    "overflowY": "auto",
                                },
                                style_cell={
                                    "textAlign": "left",
                                    "padding": "10px",
                                    "fontFamily": "'Segoe UI', Arial, sans-serif",
                                    "fontSize": "13px",
                                    "minWidth": "120px",
                                },
                                style_header={
                                    "backgroundColor": "#2c3e50",
                                    "color": "white",
                                    "fontWeight": "bold",
                                    "textAlign": "center",
                                    "fontSize": "14px",
                                    "padding": "12px",
                                },
                                style_data={
                                    "backgroundColor": "white",
                                    "border": "1px solid #ddd",
                                },
                                style_data_conditional=style_conditions,  # type: ignore
                                page_size=20,
                                sort_action="native",
                                filter_action="native",
                            ),
                        ],
                        style={
                            "padding": "20px",
                            "backgroundColor": "white",
                            "borderRadius": "10px",
                            "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
                        },
                    )

            except Exception as e:
                error_div = html.Div(
                    [
                        html.H4(
                            "Error",
                            style={
                                "marginBottom": "10px",
                                "fontFamily": "'Segoe UI', Arial, sans-serif",
                            },
                        ),
                        html.P(
                            f"Failed to generate cell type comparison: {str(e)}",
                            style={"fontFamily": "'Segoe UI', Arial, sans-serif"},
                        ),
                    ],
                    style={
                        "backgroundColor": "#ffcccc",
                        "padding": 15,
                        "borderRadius": 5,
                        "border": "1px solid #f5c6cb",
                        "fontFamily": "'Segoe UI', Arial, sans-serif",
                    },
                )
                return error_div

            return html.Div(
                "Select a tab above to view cell type comparisons.",
                style={
                    "padding": "20px",
                    "backgroundColor": "#e7f3ff",
                    "border": "1px solid #2196F3",
                    "borderRadius": "5px",
                    "color": "#0d47a1",
                    "fontFamily": "'Segoe UI', Arial, sans-serif",
                },
            )

        # Run the app
        print(f"Starting Feature Visualizer Dashboard at http://{host}:{port}")
        app.run(host=host, port=port, debug=debug)  # type: ignore

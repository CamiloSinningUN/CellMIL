"""
SHAP visualization module.

Creates various plots for SHAP values including feature importance,
beeswarm plots, and distribution visualizations.
Uses SHAP's built-in plotting when possible, and Plotly for custom visualizations.
"""

from pathlib import Path
from typing import List, Optional, Any
import numpy as np
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
import plotly.graph_objects as go  # type: ignore
import shap  # type: ignore
from cellmil.interfaces.SHAPExplainerConfig import SHAPExplainerConfig
from cellmil.utils import logger


class SHAPVisualizer:
    """Creates visualizations for SHAP explanations."""

    def __init__(self, config: SHAPExplainerConfig):
        """
        Initialize the SHAP visualizer.

        Args:
            config: Configuration for SHAP explanation
        """
        self.config = config

    def create_visualizations(
        self,
        shap_values: np.ndarray[Any, Any],
        sampled_features: np.ndarray[Any, Any],
        feature_importance: np.ndarray[Any, Any],
        top_features_idx: np.ndarray[Any, Any],
        output_dir: Path,
        feature_names: Optional[List[str]] = None,
    ) -> List[Path]:
        """
        Create all SHAP visualizations.

        Args:
            shap_values: SHAP values array [n_samples, n_features]
            sampled_features: Feature values [n_samples, n_features]
            feature_importance: Mean absolute SHAP values per feature
            top_features_idx: Indices of top important features
            output_dir: Directory to save plots
            feature_names: Optional list of feature names

        Returns:
            List of paths to created plot files
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        saved_files: List[Path] = []

        # Feature importance bar plot using SHAP's built-in plot
        try:
            bar_plot = self._create_feature_importance_bar(
                shap_values,
                sampled_features,
                top_features_idx,
                output_dir,
                feature_names,
            )
            saved_files.append(bar_plot)
        except Exception as e:
            logger.error(f"Error creating feature importance bar plot: {e}")

        # SHAP summary beeswarm plot using SHAP's built-in plot
        try:
            beeswarm_plot = self._create_beeswarm_plot(
                shap_values, sampled_features, output_dir, feature_names
            )
            saved_files.append(beeswarm_plot)
        except Exception as e:
            logger.error(f"Error creating beeswarm plot: {e}")

        # Feature importance distribution using Plotly
        try:
            dist_plot = self._create_importance_distribution(
                feature_importance, output_dir
            )
            saved_files.append(dist_plot)
        except Exception as e:
            logger.error(f"Error creating importance distribution: {e}")

        logger.info(f"Created {len(saved_files)} visualization files")
        return saved_files

    def _create_feature_importance_bar(
        self,
        shap_values: np.ndarray[Any, Any],
        sampled_features: np.ndarray[Any, Any],
        top_features_idx: np.ndarray[Any, Any],
        output_dir: Path,
        feature_names: Optional[List[str]] = None,
    ) -> Path:
        """Create horizontal bar plot of top feature importances using SHAP's built-in plot."""

        # Create feature names array if provided
        if feature_names is not None:
            feature_names_array = np.array(feature_names)
        else:
            feature_names_array = np.array(
                [f"Feature {i}" for i in range(shap_values.shape[1])]
            )

        # Use SHAP's bar plot
        plt.figure(figsize=(10, 8))  # type: ignore
        shap.plots.bar(
            shap.Explanation(
                values=shap_values,
                data=sampled_features,
                feature_names=feature_names_array,
            ),
            max_display=len(top_features_idx),
            show=False,
        )

        # Save
        output_file = output_dir / "feature_importance_bar.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")  # type: ignore
        plt.close()

        logger.info(f"Saved feature importance bar plot to {output_file}")
        return output_file

    def _create_beeswarm_plot(
        self,
        shap_values: np.ndarray[Any, Any],
        sampled_features: np.ndarray[Any, Any],
        output_dir: Path,
        feature_names: Optional[List[str]] = None,
    ) -> Path:
        """Create SHAP summary beeswarm plot using SHAP's built-in plot."""

        # Create feature names array if provided
        if feature_names is not None:
            feature_names_array = np.array(feature_names)
        else:
            feature_names_array = None

        plt.figure(figsize=(10, 8))  # type: ignore

        # Use SHAP's beeswarm plot
        shap.plots.beeswarm(
            shap.Explanation(
                values=shap_values,
                data=sampled_features,
                feature_names=feature_names_array,
            ),
            max_display=self.config.top_features,
            show=False,
            group_remaining_features=False,
        )

        # Save
        output_file = output_dir / "shap_summary_beeswarm.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")  # type: ignore
        plt.close()

        logger.info(f"Saved SHAP beeswarm plot to {output_file}")
        return output_file

    def _create_importance_distribution(
        self,
        feature_importance: np.ndarray[Any, Any],
        output_dir: Path,
    ) -> Path:
        """Create histogram of feature importance distribution using Plotly."""

        # Calculate statistics
        mean_imp = float(feature_importance.mean())
        median_imp = float(np.median(feature_importance))

        # Create interactive histogram with Plotly
        fig = go.Figure()

        # Add histogram
        fig.add_trace(  # type: ignore
            go.Histogram(
                x=feature_importance,
                nbinsx=50,
                name="Feature Importance",
                marker_color="steelblue",
                marker_line_color="black",
                marker_line_width=1,
                opacity=0.7,
            )
        )

        # Add mean line
        fig.add_vline(  # type: ignore
            x=mean_imp,
            line_dash="dash",
            line_color="red",
            line_width=2,
            annotation_text=f"Mean: {mean_imp:.4f}",
            annotation_position="top right",
        )

        # Add median line
        fig.add_vline(  # type: ignore
            x=median_imp,
            line_dash="dash",
            line_color="orange",
            line_width=2,
            annotation_text=f"Median: {median_imp:.4f}",
            annotation_position="top left",
        )

        # Update layout
        fig.update_layout(  # type: ignore
            title="Distribution of Feature Importance Scores",
            xaxis_title="Mean |SHAP value|",
            yaxis_title="Number of features",
            font=dict(size=12),
            showlegend=True,
            hovermode="x unified",
            template="plotly_white",
        )

        # Save as HTML (interactive)
        output_file_html = output_dir / "feature_importance_distribution.html"
        fig.write_html(str(output_file_html))  # type: ignore

        logger.info(
            f"Saved importance distribution to {output_file_html}"
        )
        return output_file_html

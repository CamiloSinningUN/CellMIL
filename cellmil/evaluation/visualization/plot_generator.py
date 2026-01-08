"""Plotly chart generation for experiment results."""

import pandas as pd
import plotly.graph_objects as go  # type: ignore
from pathlib import Path
from cellmil.utils import logger


class PlotGenerator:
    """Generate interactive Plotly charts from experiment results."""

    def __init__(self, output_dir: Path):
        """
        Initialize plot generator.

        Args:
            output_dir: Base directory for saving plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def create_plots(
        self,
        df: pd.DataFrame,
        metrics: list[str],
        group_by_column: str = "experiment_id",
    ):
        """
        Generate plots for all metrics and tasks.

        Args:
            df: DataFrame with experiment results (must have columns: group_by_column, task, and metric columns)
            metrics: List of metric column names to plot
            group_by_column: Column name to group results by (e.g., "experiment_id")
        """
        logger.info("Starting plot generation...")

        tasks = df["task"].unique().tolist()  # type: ignore

        for metric in metrics:
            logger.info(f"Generating plots for metric: {metric}")

            # Create metric directory
            metric_dir = self.output_dir / str(metric)
            metric_dir.mkdir(exist_ok=True)

            for task in tasks:
                logger.info(f"  Processing task: {task}")

                # Filter data for this task and metric
                task_df = self._filter_data_for_task_metric(df, task, str(metric))

                if task_df.empty:
                    logger.warning(f"  No data for task {task} with metric {metric}")
                    continue

                # Create and save plot
                fig = self._create_plot_for_task(
                    task_df, task, str(metric), group_by_column
                )
                self._save_plot(fig, metric_dir, task)

        logger.info(f"All plots saved to: {self.output_dir}")

    def _filter_data_for_task_metric(
        self, df: pd.DataFrame, task: str, metric: str
    ) -> pd.DataFrame:
        """Filter DataFrame for a specific task and metric."""
        task_df = df[df["task"] == task].copy()
        task_df = task_df[task_df[metric].notna()].copy()
        return task_df

    def _create_plot_for_task(
        self,
        task_df: pd.DataFrame,
        task: str,
        metric: str,
        group_by_column: str,
    ) -> go.Figure:
        """Create a plotly horizontal bar chart for a task and metric."""
        # Group by experiment and calculate mean and std
        grouped = (
            task_df.groupby(group_by_column)[metric]  # type: ignore
            .agg(["mean", "std"])
            .reset_index()
        )
        grouped.columns = [group_by_column, "mean", "std"]

        # Sort by mean (descending)
        grouped = grouped.sort_values("mean", ascending=True)  # type: ignore

        # Create horizontal scatter plot with error bars
        fig = go.Figure()

        fig.add_trace(  # type: ignore
            go.Scatter(
                y=grouped[group_by_column],
                x=grouped["mean"],
                error_x=dict(
                    type="data", array=grouped["std"], visible=True, thickness=2
                ),
                mode="markers",
                marker=dict(
                    size=10,
                    color=grouped["mean"],
                    colorscale="RdYlGn",
                    showscale=True,
                    colorbar=dict(title=metric),
                    line=dict(width=1, color="DarkSlateGrey"),
                ),
                text=[f"{m:.3f}" for m in grouped["mean"]],
                hovertemplate="<b>%{y}</b><br>Mean: %{x:.4f}<br>Std: %{customdata:.4f}<extra></extra>",
                customdata=grouped["std"],
            )
        )

        # Update layout
        fig.update_layout(  # type: ignore
            title=f"{task} - {metric}",
            xaxis_title=f"Mean {metric} (with StDev)",
            yaxis_title="Experiment",
            height=max(400, len(grouped) * 20),
            showlegend=False,
            template="plotly_white",
            font=dict(size=10),
            margin=dict(l=300, r=100, t=80, b=60),
        )

        return fig

    def _save_plot(self, fig: go.Figure, metric_dir: Path, task: str):
        """Save a plotly figure as HTML."""
        output_file = metric_dir / f"{task}.html"
        fig.write_html(str(output_file))  # type: ignore
        logger.info(f"    Saved plot to: {output_file}")

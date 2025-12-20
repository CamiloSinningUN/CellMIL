import wandb
import pandas as pd
import plotly.graph_objects as go  # type: ignore
from pathlib import Path
from typing import Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from cellmil.utils.wandb import WandbClient
from cellmil.interfaces.EvaluationReporterConfig import EvaluationReporterConfig
from cellmil.utils import logger


class EvaluationReporter:
    COLUMN_EXPERIMENT_ID = "EXPERIMENT_ID"
    COLUMN_TASK = "TASK"

    def __init__(self, config: EvaluationReporterConfig):
        self.config = config
        wandb.login()

        self.tasks: list[str] = [
            "ADENOvsSQUA",
            "PDL1(BIN)",
            "DCR",
            "OS6",
            "OS24",
            "ORR",
            "CBR",
            "OS",
            "PFS",
        ]

        self.df = pd.DataFrame()
        self.wandb_client = WandbClient(
            team=self.config.team, projects=self.config.projects, tasks=self.tasks
        )
        self.runs = self.wandb_client.get_runs(preprocess=True)
        logger.info(f"Total accessible runs after preprocessing: {len(self.runs)}")

        self._load_runs_into_df()

        if self.df.empty:
            raise RuntimeError(
                "DataFrame is empty after loading runs. Check logs for errors during run processing."
            )

    def _load_runs_into_df(self):
        """Load wandb runs into a DataFrame with all available metrics."""

        def process_run(run: Any) -> dict[str, str | None | float]:
            """Process a single run and return its data."""
            experiment_id = self.wandb_client.get_experiment_id(run)
            task = self.wandb_client.get_task(experiment_id)

            run_data: dict[str, str | None | float] = {
                self.COLUMN_EXPERIMENT_ID: experiment_id,
                self.COLUMN_TASK: task,
            }

            # Try to get all possible metrics
            for metric in self.config.metrics:
                metric_value = self._get_metric_value(run, str(metric))
                run_data[str(metric)] = metric_value

            return run_data

        data: list[dict[str, Any]] = []

        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=8) as executor:
            future_to_run = {
                executor.submit(process_run, run): run for run in self.runs
            }

            with tqdm(total=len(self.runs), desc="Loading runs into DataFrame") as pbar:
                for future in as_completed(future_to_run):
                    try:
                        run_data = future.result()
                        data.append(run_data)
                    except Exception as e:
                        run = future_to_run[future]
                        logger.warning(f"Failed to process run {run.name}: {e}")
                    finally:
                        pbar.update(1)

        self.df = pd.DataFrame(data)
        logger.info(f"Loaded {len(self.df)} runs into DataFrame.")

    def _get_metric_value(self, run: Any, metric: str) -> float | None:
        """Get the highest validation metric for a run, or None if not available.

        Args:
            run: A wandb run object
            metric: The metric name (e.g., "f1_score", "c_index", "auc")

        Returns:
            The highest validation metric score or None if not available
        """
        try:
            return self.wandb_client.get_metric(run, metric)
        except (ValueError, KeyError):
            return None

    def create(self):
        """Generate plots for all metrics and tasks."""
        logger.info("Starting plot generation...")

        # Create base output directory
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(exist_ok=True)

        for metric in self.config.metrics:
            logger.info(f"Generating plots for metric: {metric}")

            # Create metric directory
            metric_dir = output_dir / str(metric)
            metric_dir.mkdir(exist_ok=True)

            for task in self.tasks:
                logger.info(f"  Processing task: {task}")

                # Filter data for this task and metric
                task_df = self._filter_data_for_task_metric(task, str(metric))

                if task_df.empty:
                    logger.warning(f"  No data for task {task} with metric {metric}")
                    continue

                # Create and save plot
                fig = self._create_plot_for_task(task_df, task, str(metric))
                self._save_plot(fig, metric_dir, task)

        logger.info(f"All plots saved to: {output_dir}")

    def _filter_data_for_task_metric(self, task: str, metric: str) -> pd.DataFrame:
        """Filter DataFrame for a specific task and metric.

        Args:
            task: The task name
            metric: The metric name

        Returns:
            Filtered DataFrame with only runs that have this metric
        """
        task_df = self.df[self.df[self.COLUMN_TASK] == task].copy()

        # Remove rows where the metric is None or NaN
        task_df = task_df[task_df[metric].notna()].copy()

        return task_df

    def _create_plot_for_task(
        self, task_df: pd.DataFrame, task: str, metric: str
    ) -> go.Figure:
        """Create a plotly horizontal bar chart for a task and metric.

        Args:
            task_df: DataFrame filtered for the task and metric
            task: The task name
            metric: The metric name

        Returns:
            Plotly Figure object
        """
        # Group by experiment_id and calculate mean and std
        grouped = (
            task_df.groupby(self.COLUMN_EXPERIMENT_ID)[metric]  # type: ignore
            .agg(["mean", "std"])
            .reset_index()
        )
        grouped.columns = [self.COLUMN_EXPERIMENT_ID, "mean", "std"]

        # Sort by mean (descending)
        grouped = grouped.sort_values("mean", ascending=True)  # type: ignore

        # Create horizontal scatter plot with error bars
        fig = go.Figure()

        fig.add_trace(  # type: ignore
            go.Scatter(
                y=grouped[self.COLUMN_EXPERIMENT_ID],
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
            xaxis_title=f"Mean Max val/{metric} (with StDev)",
            yaxis_title="Experiment",
            height=max(
                400, len(grouped) * 20
            ),  # Dynamic height based on number of experiments
            showlegend=False,
            template="plotly_white",
            font=dict(size=10),
            margin=dict(
                l=300, r=100, t=80, b=60
            ),  # Left margin for long experiment names
        )

        return fig

    def _save_plot(self, fig: go.Figure, metric_dir: Path, task: str):
        """Save a plotly figure as HTML.

        Args:
            fig: The plotly Figure object
            metric_dir: Directory to save the plot
            task: The task name (used for filename)
        """
        output_file = metric_dir / f"{task}.html"
        fig.write_html(str(output_file))  # type: ignore
        logger.info(f"    Saved plot to: {output_file}")

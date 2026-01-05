import wandb
import pandas as pd
import plotly.graph_objects as go  # type: ignore
import numpy as np
from pathlib import Path
from typing import Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from cellmil.utils.wandb import (
    WandbClient,
    COLUMN_EXPERIMENT_ID,
    COLUMN_TASK,
    COLUMN_FEATURES,
    COLUMN_MODEL,
    COLUMN_REG,
    COLUMN_STRA,
)
from cellmil.interfaces.EvaluationReporterConfig import EvaluationReporterConfig
from cellmil.interfaces.TableConfig import TableConfig
from cellmil.utils import logger


class EvaluationReporter:
    def __init__(self, config: EvaluationReporterConfig):
        self.config = config
        wandb.login()

        self.df = pd.DataFrame()
        self.wandb_client = WandbClient(
            team=self.config.team, projects=self.config.projects
        )
        self.runs = self.wandb_client.get_runs(preprocess=True)
        logger.info(f"Total accessible runs after preprocessing: {len(self.runs)}")

        self._load_runs_into_df()

        self.tasks = self.df[COLUMN_TASK].unique().tolist()  # type: ignore

        if self.df.empty:
            raise RuntimeError(
                "DataFrame is empty after loading runs. Check logs for errors during run processing."
            )

    def _load_runs_into_df(self):
        """Load wandb runs into a DataFrame with all available metrics."""

        def process_run(run: Any) -> dict[str, str | None | float]:
            """Process a single run and return its data."""
            experiment_id = self.wandb_client.get_experiment_id(run)
            components = self.wandb_client.parse_experiment_components(experiment_id)

            run_data: dict[str, str | None | float] = {
                COLUMN_EXPERIMENT_ID: experiment_id,
                COLUMN_TASK: components.task,
                COLUMN_FEATURES: components.features,
                COLUMN_MODEL: components.model,
                COLUMN_REG: components.regularization,
                COLUMN_STRA: components.stratification,
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
        """Generate plots for all metrics and tasks (legacy method)."""
        self.create_plots()

        # Common configuration for all tables
        base_config: dict[str, Any] = {
            "baseline_features": ["RESNET", "GIGAPATH"],
            "task_mapping": {
                "DCR": "DCR",
                "ORR": "ORR",
                "CBR": "CBR",
                "OS6": "OS6",
                "OS24": "OS24",
                "PDL1": "PDL1",
                "ADENOvsSQUA": "Adeno. vs Squa.",
            },
            "feature_mapping": {
                "RESNET": "ResNet50",
                "GIGAPATH": "GigaPath",
                "PYRAD": "Radiomics",
                "MORPHO": "Morphological",
                "TOPO": "Topological",
                "ALL": "All",
            },
            "model_mapping": {
                "ABMIL": "ABMIL",
                "HEAD4TYPE": "Head4Type",
                "CLAM": "CLAM",
            },
            "metric_mapping": {
                "f1": "F1",
                "recall": "Bal. Acc.",
                "c_index": "C-Index",
            },
            "support_mapping": {
                "DCR": 343,
                "ORR": 343,
                "CBR": 343,
                "OS6": 339,
                "OS24": 295,
                "PDL1": 306,
                "ADENOvsSQUA": 280,
            },
        }

        classification_stratified_config = TableConfig(
            output_file=str(self.config.output_dir / "classification_stratified.tex"),
            table_type="classification",
            include_stratified=True,
            classification_metrics=["f1", "recall"],
            **base_config,
        )

        classification_non_stratified_config = TableConfig(
            output_file=str(
                self.config.output_dir / "classification_non_stratified.tex"
            ),
            table_type="classification",
            include_stratified=False,
            classification_metrics=["f1", "recall"],
            **base_config,
        )

        survival_stratified_config = TableConfig(
            output_file=str(self.config.output_dir / "survival_stratified.tex"),
            table_type="survival",
            include_stratified=True,
            survival_metrics=["c_index"],
            **base_config,
        )

        survival_non_stratified_config = TableConfig(
            output_file=str(self.config.output_dir / "survival_non_stratified.tex"),
            table_type="survival",
            include_stratified=False,
            survival_metrics=["c_index"],
            **base_config,
        )

        self.create_tables(
            [
                classification_stratified_config,
                classification_non_stratified_config,
                survival_stratified_config,
                survival_non_stratified_config,
            ]
        )

    def create_plots(self):
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
        task_df = self.df[self.df[COLUMN_TASK] == task].copy()

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
            task_df.groupby(COLUMN_EXPERIMENT_ID)[metric]  # type: ignore
            .agg(["mean", "std"])
            .reset_index()
        )
        grouped.columns = [COLUMN_EXPERIMENT_ID, "mean", "std"]

        # Sort by mean (descending)
        grouped = grouped.sort_values("mean", ascending=True)  # type: ignore

        # Create horizontal scatter plot with error bars
        fig = go.Figure()

        fig.add_trace(  # type: ignore
            go.Scatter(
                y=grouped[COLUMN_EXPERIMENT_ID],
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

    def _filter_dataframe(self, config: TableConfig) -> pd.DataFrame:
        """Filter DataFrame based on table configuration.

        Args:
            config: TableConfig with filtering criteria

        Returns:
            Filtered DataFrame
        """
        df = self.df.copy()

        # Filter by tasks
        if config.tasks is not None:
            df = df[df[COLUMN_TASK].isin(config.tasks)]  # type: ignore

        # Filter by features
        if config.features is not None:
            df = df[df[COLUMN_FEATURES].isin(config.features)]  # type: ignore

        # Filter by models
        if config.models is not None:
            df = df[df[COLUMN_MODEL].isin(config.models)]  # type: ignore

        # Filter by regularization
        if not config.include_regularized:
            df = df[df[COLUMN_REG] == "*"]
        if not config.include_non_regularized:
            df = df[df[COLUMN_REG] != "*"]

        # Filter by stratification
        if not config.include_stratified:
            df = df[df[COLUMN_STRA] == "*"]
        else:
            df = df[df[COLUMN_STRA] != "*"]

        return df

    def _get_display_name(self, value: str, mapping: dict[str, str]) -> str:
        """Get display name from mapping or return original value.

        Args:
            value: The original value
            mapping: Dictionary mapping original to display names

        Returns:
            Display name or original value
        """
        return mapping.get(value, value)

    def _format_mean_std(self, values: list[float | None], precision: int = 3) -> str:
        """Format list of values as mean ± std.

        Args:
            values: List of metric values
            precision: Number of decimal places

        Returns:
            Formatted string like \"0.534±0.026\"
        """
        if not values or all(v is None or np.isnan(v) for v in values):
            return "-"

        valid_values = [v for v in values if v is not None and not np.isnan(v)]
        if not valid_values:
            return "-"

        mean = np.mean(valid_values)
        std = np.std(valid_values) if len(valid_values) > 1 else 0.0

        return f"${mean:.{precision}f}\\pm{std:.{precision}f}$"

    def _get_metric_values(
        self,
        df: pd.DataFrame,
        task: str,
        features: str,
        model: str,
        reg: str,
        metric: str,
    ) -> list[float | None]:
        """Get all metric values for a specific combination.

        Args:
            df: DataFrame to query
            task: Task name
            features: Features name
            model: Model name
            reg: Regularization status
            metric: Metric name

        Returns:
            List of metric values (one per fold)
        """
        filtered = df[
            (df[COLUMN_TASK] == task)
            & (df[COLUMN_FEATURES] == features)
            & (df[COLUMN_MODEL] == model)
            & (df[COLUMN_REG] == reg)
        ]

        if filtered.empty or metric not in filtered.columns:
            return []

        values = filtered[metric].dropna().tolist()
        return [float(v) for v in values]

    def _find_best_values_per_task(
        self, df: pd.DataFrame, tasks: list[str], metrics: list[str]
    ) -> dict[tuple[str, str], float]:
        """Find the best (highest) value for each task-metric combination.

        Args:
            df: DataFrame to analyze
            tasks: List of tasks
            metrics: List of metrics

        Returns:
            Dictionary mapping (task, metric) -> best_mean_value
        """
        best_values: dict[tuple[str, str], float] = {}

        for task in tasks:
            for metric in metrics:
                task_df = df[df[COLUMN_TASK] == task]
                if task_df.empty or metric not in task_df.columns:
                    continue

                # Group by features, model, AND regularization to get separate means
                groups = task_df.groupby(  # type: ignore
                    [COLUMN_FEATURES, COLUMN_MODEL, COLUMN_REG]
                )[metric].mean()

                if not groups.empty:
                    best_values[(task, metric)] = float(groups.max())

        return best_values

    def _format_cell_with_bold(
        self,
        values: list[float | None],
        task: str,
        metric: str,
        best_values: dict[tuple[str, str], float],
        precision: int = 3,
    ) -> str:
        """Format cell value, making it bold if it's the best for that task-metric.

        Args:
            values: List of metric values
            task: Task name
            metric: Metric name
            best_values: Dictionary of best values per task-metric
            precision: Number of decimal places

        Returns:
            Formatted LaTeX string
        """
        formatted = self._format_mean_std(values, precision)

        if formatted == "-":
            return "-"

        # Extract mean value to compare
        valid_values = [v for v in values if v is not None and not np.isnan(v)]
        if not valid_values:
            return formatted

        mean = np.mean(valid_values)
        best_value = best_values.get((task, metric))

        # Consider values equal if within floating point tolerance
        if best_value is not None and np.isclose(mean, best_value, rtol=1e-6):
            # Make it bold
            return f"\\boldmath{{{formatted}}}"

        return formatted

    def create_tables(self, configs: list[TableConfig]):
        """Generate LaTeX tables based on configurations.

        Args:
            configs: List of TableConfig objects
        """
        logger.info("Starting table generation...")

        for config in configs:
            logger.info(f"Generating table: {config.output_file}")
            self._create_single_table(config)

        logger.info("All tables generated successfully")

    def _create_single_table(self, config: TableConfig):
        """Create a single LaTeX table based on configuration.

        Args:
            config: TableConfig object
        """
        # Filter data
        df = self._filter_dataframe(config)

        if df.empty:
            logger.warning(f"No data for table {config.output_file}")
            return

        # Get metrics based on table type (convert Metrics enum to string)
        metrics_enums = (
            config.classification_metrics
            if config.table_type == "classification"
            else config.survival_metrics
        )
        metrics = [str(m) for m in metrics_enums]

        # Get unique tasks (filtered)
        tasks = sorted(df[COLUMN_TASK].unique().tolist())  # type: ignore

        # Find best values for bolding
        best_values = self._find_best_values_per_task(self.df, tasks, metrics)

        # Generate LaTeX table
        latex = self._generate_table(df, tasks, metrics, config, best_values)

        # Write to file
        output_path = Path(config.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(latex)
        logger.info(f"  Table saved to: {output_path}")

    def _generate_table(
        self,
        df: pd.DataFrame,
        tasks: list[str],
        metrics: list[str],
        config: TableConfig,
        best_values: dict[tuple[str, str], float],
    ) -> str:
        """Generate LaTeX code for table.

        Args:
            df: Filtered DataFrame
            tasks: List of tasks to include
            metrics: List of metrics to include
            config: Table configuration
            best_values: Best values for bolding

        Returns:
            LaTeX table code
        """
        # Get unique features and models
        features_list = sorted(df[COLUMN_FEATURES].unique().tolist())  # type: ignore
        models_list = sorted(df[COLUMN_MODEL].unique().tolist())  # type: ignore

        # Build header
        num_metrics = len(metrics)
        num_tasks = len(tasks)
        total_cols = 2 + num_tasks * num_metrics  # FEATURES + MODEL + task columns

        # Column specification
        col_spec = "|p{3cm}|C{1.8cm}||"
        for _ in range(num_tasks):
            col_spec += "|".join(["C{1.3cm}"] * num_metrics) + "|"

        latex_lines: list[str] = []
        latex_lines.append("\\footnotesize")
        latex_lines.append("\\setlength{\\tabcolsep}{2pt}")
        latex_lines.append("\\renewcommand{\\arraystretch}{1.5}")
        latex_lines.append(f"\\begin{{longtable}}{{{col_spec}}}")

        # Caption and label
        if config.caption:
            latex_lines.append(f"    \\caption{{{config.caption}}}")
        if config.label:
            latex_lines.append(f"    \\label{{{config.label}}}")

        # Header
        latex_lines.append("    \\toprule")

        # Task names header
        task_headers: list[str] = []
        for task in tasks:
            display_task = self._get_display_name(task, config.task_mapping)
            colspan = num_metrics
            task_headers.append(
                f"\\multicolumn{{{colspan}}}{{|c|}}{{\\textbf{{{display_task}}}}}"
            )

        header_line = f"    \\textbf{{FEATURES}} & \\textbf{{MODEL}} & {' & '.join(task_headers)}\\\\"
        latex_lines.append(header_line)
        latex_lines.append("    \\midrule")

        # Metric names header
        metric_headers: list[str] = []
        for _ in tasks:
            for metric in metrics:
                display_metric = self._get_display_name(metric, config.metric_mapping)
                metric_headers.append(display_metric)

        metric_line = (
            f"     \\multicolumn{{2}}{{|c||}}{{}} & {' & '.join(metric_headers)}\\\\"
        )
        latex_lines.append(metric_line)
        latex_lines.append("    \\midrule")
        latex_lines.append("    \\endfirsthead")
        latex_lines.append("    ")

        # Continued header
        latex_lines.append(
            "    \\multicolumn{"
            + str(total_cols)
            + "}{c}{\\textit{(Continued from previous page)}}\\\\"
        )
        latex_lines.append("    \\toprule")
        latex_lines.append(header_line)
        latex_lines.append("    \\midrule")
        latex_lines.append(metric_line)
        latex_lines.append("    \\midrule")
        latex_lines.append("    \\endhead")
        latex_lines.append("    ")
        latex_lines.append("    \\midrule")
        latex_lines.append(
            "    \\multicolumn{"
            + str(total_cols)
            + "}{r}{\\textit{(Continued on next page)}}\\\\"
        )
        latex_lines.append("    \\endfoot")
        latex_lines.append("    ")

        # Last foot with support
        if config.support_mapping:
            support_cells: list[str] = []
            for task in tasks:
                support = config.support_mapping.get(task, "-")
                colspan = num_metrics
                support_cells.append(f"\\multicolumn{{{colspan}}}{{c|}}{{{support}}}")

            support_line = f"    \\multicolumn{{2}}{{|c||}}{{\\textbf{{SUPPORT}}}} & {' & '.join(support_cells)} \\\\"
            latex_lines.append("    \\midrule")
            latex_lines.append(support_line)

        latex_lines.append("    \\bottomrule")
        latex_lines.append("    \\endlastfoot")
        latex_lines.append("    ")

        # Table rows
        # Generate baseline section if configured
        if config.baseline_features:
            baseline_df = df[df[COLUMN_FEATURES].isin(config.baseline_features)]  # type: ignore
            if not baseline_df.empty:
                latex_lines.append(
                    "    \\multicolumn{" + str(total_cols) + "}{c}{} \\\\"
                )
                latex_lines.append(
                    "    \\multicolumn{"
                    + str(total_cols)
                    + "}{c}{\\textbf{Patch Embeddings (Baseline)}} \\\\"
                )
                latex_lines.append("    \\midrule")
                baseline_features = sorted(
                    baseline_df[COLUMN_FEATURES].unique().tolist()  # type: ignore
                )
                baseline_models = sorted(baseline_df[COLUMN_MODEL].unique().tolist())  # type: ignore
                latex_lines.extend(
                    self._generate_table_rows(
                        baseline_df,
                        baseline_features,
                        baseline_models,
                        tasks,
                        metrics,
                        config,
                        best_values,
                    )
                )
                latex_lines.append("    \\midrule")
            # Remove baseline features from main sections
            df = df[~df[COLUMN_FEATURES].isin(config.baseline_features)]  # type: ignore
            features_list = sorted(df[COLUMN_FEATURES].unique().tolist())  # type: ignore
            models_list = sorted(df[COLUMN_MODEL].unique().tolist())  # type: ignore

        # Main table sections
        # Separate by regularization first
        if config.include_non_regularized:
            df_no_reg = df[df[COLUMN_REG] == "*"]
            if not df_no_reg.empty:
                latex_lines.append(
                    "    \\multicolumn{" + str(total_cols) + "}{c}{} \\\\"
                )
                latex_lines.append(
                    "    \\multicolumn{"
                    + str(total_cols)
                    + "}{c}{\\textbf{Cell features}} \\\\"
                )
                latex_lines.append("    \\midrule")
                latex_lines.extend(
                    self._generate_table_rows(
                        df_no_reg,
                        features_list,
                        models_list,
                        tasks,
                        metrics,
                        config,
                        best_values,
                    )
                )
                latex_lines.append("    \\midrule")

        if config.include_regularized:
            df_reg = df[df[COLUMN_REG] != "*"]
            if not df_reg.empty:
                latex_lines.append(
                    "    \\multicolumn{" + str(total_cols) + "}{|c|}{} \\\\"
                )
                latex_lines.append(
                    "    \\multicolumn{"
                    + str(total_cols)
                    + "}{|c|}{\\textbf{Attention Entropy Maximization + Subsampling + L2 Regularization}} \\\\"
                )
                latex_lines.append("    \\midrule")
                latex_lines.extend(
                    self._generate_table_rows(
                        df_reg,
                        features_list,
                        models_list,
                        tasks,
                        metrics,
                        config,
                        best_values,
                    )
                )

        latex_lines.append("    \\midrule")
        latex_lines.append("\\end{longtable}")

        return "\n".join(latex_lines)

    def _generate_table_rows(
        self,
        df: pd.DataFrame,
        features_list: list[str],
        models_list: list[str],
        tasks: list[str],
        metrics: list[str],
        config: TableConfig,
        best_values: dict[tuple[str, str], float],
    ) -> list[str]:
        """Generate table rows for given data.

        Args:
            df: DataFrame to process
            features_list: List of feature types
            models_list: List of model types
            tasks: List of tasks
            metrics: List of metrics
            config: Table configuration
            best_values: Best values for bolding

        Returns:
            List of LaTeX row strings
        """
        rows: list[str] = []

        for model in models_list:
            for features in features_list:
                # Get unique regularization values for this combination
                combo_df = df[
                    (df[COLUMN_FEATURES] == features) & (df[COLUMN_MODEL] == model)
                ]

                if combo_df.empty:
                    continue

                # Iterate over each regularization value (each creates a separate row)
                for reg in sorted(combo_df[COLUMN_REG].unique()):  # type: ignore
                    # Get display names
                    display_features = self._get_display_name(
                        features, config.feature_mapping
                    )
                    display_model = self._get_display_name(model, config.model_mapping)

                    row_cells = [display_features, display_model]

                    # Add metric values for each task
                    for task in tasks:
                        for metric in metrics:
                            values = self._get_metric_values(
                                df, task, features, model, reg, metric
                            )
                            formatted = self._format_cell_with_bold(
                                values, task, metric, best_values, config.precision
                            )
                            row_cells.append(formatted)

                    row = "    " + " & ".join(row_cells) + "\\\\"
                    rows.append(row)
            rows.append("    \\midrule")

        return rows

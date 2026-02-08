"""LaTeX table generation for experiment results."""

import pandas as pd
import numpy as np
from pathlib import Path
from cellmil.interfaces.TableConfig import TableConfig
from cellmil.utils import logger


class TableGenerator:
    """Generate LaTeX tables from experiment results."""

    # Standard column names expected in the DataFrame
    COLUMN_EXPERIMENT_ID = "experiment_id"
    COLUMN_TASK = "task"
    COLUMN_FEATURES = "features"
    COLUMN_MODEL = "model"
    COLUMN_REG = "regularization"
    COLUMN_STRA = "stratification"

    def __init__(self, df: pd.DataFrame):
        """
        Initialize table generator.

        Args:
            df: DataFrame with experiment results. Must contain columns:
                - experiment_id: Unique identifier for each experiment configuration
                - task: Task name (e.g., "ADENOvsSQUA", "OS24")
                - features: Feature type (e.g., "RESNET", "MORPHO")
                - model: Model type (e.g., "ABMIL", "CLAM")
                - regularization: "*" for no reg, or reg details
                - stratification: "*" for no strat, or strat details
                - [metric columns]: One column per metric with values
        """
        self.df = df.copy()
        self._validate_dataframe()

    def _validate_dataframe(self):
        """Validate that DataFrame has required columns."""
        required_cols = [
            self.COLUMN_TASK,
            self.COLUMN_FEATURES,
            self.COLUMN_MODEL,
            self.COLUMN_REG,
            self.COLUMN_STRA,
        ]
        missing = [col for col in required_cols if col not in self.df.columns]
        if missing:
            raise ValueError(f"DataFrame missing required columns: {missing}")

    def create_tables(self, configs: list[TableConfig]):
        """
        Generate LaTeX tables based on configurations.

        Args:
            configs: List of TableConfig objects
        """
        logger.info("Starting table generation...")

        for config in configs:
            logger.info(f"Generating table: {config.output_file}")
            self._create_single_table(config)

        logger.info("All tables generated successfully")

    def _create_single_table(self, config: TableConfig):
        """Create a single LaTeX table based on configuration."""
        # Filter data
        df = self._filter_dataframe(config)

        if df.empty:
            logger.warning(f"No data for table {config.output_file}")
            return

        # Get metrics based on table type
        metrics = (
            config.classification_metrics
            if config.table_type == "classification"
            else config.survival_metrics
        )

        # Get unique tasks (filtered) - preserve order from mapping
        available_tasks = set(df[self.COLUMN_TASK].unique().tolist())  # type: ignore
        tasks = [t for t in config.task_mapping.keys() if t in available_tasks]

        # Find best values for bolding
        best_values = self._find_best_values_per_task(self.df, tasks, metrics)

        # Generate LaTeX table
        latex = self._generate_table(df, tasks, metrics, config, best_values)

        # Write to file
        output_path = Path(config.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(latex)
        logger.info(f"  Table saved to: {output_path}")

    def _filter_dataframe(self, config: TableConfig) -> pd.DataFrame:
        """Filter DataFrame based on table configuration."""
        df = self.df.copy()
        
        # Exclude partial models
        df = df[~df[self.COLUMN_FEATURES].str.contains(r"\(", regex=True, na=False)]  # type: ignore

        # Filter by tasks
        if config.tasks is not None:
            df = df[df[self.COLUMN_TASK].isin(config.tasks)]  # type: ignore

        # Filter by features
        if config.features is not None:
            df = df[df[self.COLUMN_FEATURES].isin(config.features)]  # type: ignore

        # Filter by models
        if config.models is not None:
            df = df[df[self.COLUMN_MODEL].isin(config.models)]  # type: ignore

        # Filter by regularization
        if not config.include_regularized:
            df = df[df[self.COLUMN_REG] == "*"]
        if not config.include_non_regularized:
            df = df[df[self.COLUMN_REG] != "*"]

        # Filter by stratification
        if not config.include_stratified:
            df = df[df[self.COLUMN_STRA] == "*"]
        else:
            df = df[df[self.COLUMN_STRA] != "*"]

        return df

    def _get_display_name(self, value: str, mapping: dict[str, str]) -> str:
        """Get display name from mapping or return original value."""
        return mapping.get(value, value)

    def _format_mean_std(self, values: list[float | None], precision: int = 3) -> str:
        """Format list of values as mean ± std."""
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
        """Get all metric values for a specific combination."""
        filtered = df[
            (df[self.COLUMN_TASK] == task)
            & (df[self.COLUMN_FEATURES] == features)
            & (df[self.COLUMN_MODEL] == model)
            & (df[self.COLUMN_REG] == reg)
        ]

        if filtered.empty or metric not in filtered.columns:
            return []

        values = filtered[metric].dropna().tolist()
        return [float(v) for v in values]

    def _find_best_values_per_task(
        self, df: pd.DataFrame, tasks: list[str], metrics: list[str]
    ) -> dict[tuple[str, str], float]:
        """Find the best (highest) value for each task-metric combination."""
        best_values: dict[tuple[str, str], float] = {}

        for task in tasks:
            for metric in metrics:
                task_df = df[df[self.COLUMN_TASK] == task]
                if task_df.empty or metric not in task_df.columns:
                    continue

                # Group by all dimensions to get separate means
                groups = task_df.groupby(  # type: ignore
                    [
                        self.COLUMN_FEATURES,
                        self.COLUMN_MODEL,
                        self.COLUMN_REG,
                        self.COLUMN_STRA,
                    ]
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
        """Format cell value, making it bold if it's the best for that task-metric."""
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
            return f"\\boldmath{{{formatted}}}"

        return formatted

    def _generate_table(
        self,
        df: pd.DataFrame,
        tasks: list[str],
        metrics: list[str],
        config: TableConfig,
        best_values: dict[tuple[str, str], float],
    ) -> str:
        """Generate LaTeX code for table."""
        # Get unique features and models - preserve order from mappings
        available_features = set(df[self.COLUMN_FEATURES].unique().tolist())  # type: ignore
        features_list = [
            f for f in config.feature_mapping.keys() if f in available_features
        ]
        available_models = set(df[self.COLUMN_MODEL].unique().tolist())  # type: ignore
        models_list = [m for m in config.model_mapping.keys() if m in available_models]

        # Build header
        num_metrics = len(metrics)
        num_tasks = len(tasks)
        total_cols = 2 + num_tasks * num_metrics

        # Column specification
        if num_metrics == 1:
            col_spec = "|p{2.3cm}|C{2cm}||"
            for _ in range(num_tasks):
                col_spec += "C{2.2cm}|"
        else:
            col_spec = "|p{2cm}|C{1.8cm}||"
            for _ in range(num_tasks):
                col_spec += "|".join(["C{1.1cm}"] * num_metrics) + "|"

        latex_lines: list[str] = []
        latex_lines.append("{")
        latex_lines.append("\\scriptsize")
        latex_lines.append("\\setlength{\\tabcolsep}{2pt}")
        latex_lines.append("\\renewcommand{\\arraystretch}{1.5}")
        latex_lines.append(f"\\begin{{longtable}}{{{col_spec}}}")

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

        # Caption and label at bottom
        if config.caption:
            latex_lines.append(f"    \\caption{{{config.caption}}}")
        if config.label:
            latex_lines.append(f"    \\label{{{config.label}}}")

        latex_lines.append("    \\endlastfoot")
        latex_lines.append("    ")

        # Table rows - generate baseline section if configured
        if config.baseline_features:
            baseline_df = df[df[self.COLUMN_FEATURES].isin(config.baseline_features)]  # type: ignore
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
                available_baseline_features = set(
                    baseline_df[self.COLUMN_FEATURES].unique().tolist()  # type: ignore
                )
                baseline_features = [
                    f
                    for f in config.feature_mapping.keys()
                    if f in available_baseline_features
                ]
                available_baseline_models = set(
                    baseline_df[self.COLUMN_MODEL].unique().tolist()  # type: ignore
                )
                baseline_models = [
                    m
                    for m in config.model_mapping.keys()
                    if m in available_baseline_models
                ]

                # Non-regularized baseline models
                if config.include_non_regularized:
                    baseline_df_no_reg = baseline_df[
                        baseline_df[self.COLUMN_REG] == "*"
                    ]
                    if not baseline_df_no_reg.empty:
                        latex_lines.extend(
                            self._generate_table_rows(
                                baseline_df_no_reg,
                                baseline_features,
                                baseline_models,
                                tasks,
                                metrics,
                                config,
                                best_values,
                            )
                        )

                # Regularized baseline models
                if config.include_regularized:
                    baseline_df_reg = baseline_df[baseline_df[self.COLUMN_REG] != "*"]
                    if not baseline_df_reg.empty:
                        latex_lines.append(
                            "    \\multicolumn{" + str(total_cols) + "}{|c|}{} \\\\"
                        )
                        if num_metrics == 1:
                            latex_lines.append(
                                "    \\multicolumn{"
                                + str(total_cols)
                                + "}{|c|}{\\textbf{Attention Entropy Maximization +}} \\\\"
                            )
                            latex_lines.append(
                                "    \\multicolumn{"
                                + str(total_cols)
                                + "}{|c|}{\\textbf{Subsampling + L2 Regularization}} \\\\"
                            )
                        else:
                            latex_lines.append(
                                "    \\multicolumn{"
                                + str(total_cols)
                                + "}{|c|}{\\textbf{Attention Entropy Maximization + Subsampling + L2 Regularization}} \\\\"
                            )
                        latex_lines.append("    \\midrule")
                        latex_lines.extend(
                            self._generate_table_rows(
                                baseline_df_reg,
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
            df = df[~df[self.COLUMN_FEATURES].isin(config.baseline_features)]  # type: ignore
            available_features = set(df[self.COLUMN_FEATURES].unique().tolist())  # type: ignore
            features_list = [
                f for f in config.feature_mapping.keys() if f in available_features
            ]
            available_models = set(df[self.COLUMN_MODEL].unique().tolist())  # type: ignore
            models_list = [
                m for m in config.model_mapping.keys() if m in available_models
            ]

        # Main table sections
        if config.include_non_regularized:
            df_no_reg = df[df[self.COLUMN_REG] == "*"]
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
            df_reg = df[df[self.COLUMN_REG] != "*"]
            if not df_reg.empty:
                latex_lines.append(
                    "    \\multicolumn{" + str(total_cols) + "}{|c|}{} \\\\"
                )
                if num_metrics == 1:
                    latex_lines.append(
                        "    \\multicolumn{"
                        + str(total_cols)
                        + "}{|c|}{\\textbf{Attention Entropy Maximization +}} \\\\"
                    )
                    latex_lines.append(
                        "    \\multicolumn{"
                        + str(total_cols)
                        + "}{|c|}{\\textbf{Subsampling + L2 Regularization}} \\\\"
                    )
                else:
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
        latex_lines.append("}")

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
        """Generate table rows for given data."""
        rows: list[str] = []

        for model in models_list:
            for features in features_list:
                combo_df = df[
                    (df[self.COLUMN_FEATURES] == features)
                    & (df[self.COLUMN_MODEL] == model)
                ]

                if combo_df.empty:
                    continue

                # Iterate over each regularization value
                for reg in sorted(combo_df[self.COLUMN_REG].unique()):  # type: ignore
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

import wandb
import pandas as pd
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
from cellmil.evaluation.visualization import TableGenerator, PlotGenerator
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

        # Common configuration shared by classification and survival tables
        common_config: dict[str, Any] = {
            "baseline_features": ["RESNET", "GIGAPATH"],
            "feature_mapping": {
                "RESNET": "ResNet50",
                "GIGAPATH": "GigaPath",
                "MORPHO": "Morphological",
                "PYRAD": "Radiomics",
                "TOPO": "Topological",
                "ALL": "All",
            },
            "model_mapping": {
                "ABMIL": "ABMIL",
                "CLAM": "CLAM",
                "HEAD4TYPE": "Head4Type",
            },
            "metric_mapping": {
                "f1": "F1",
                "recall": "Bal. Acc.",
                "c_index": "C-Index",
            },
        }

        # Classification-specific configuration
        classification_config: dict[str, Any] = {
            **common_config,
            "task_mapping": {
                "ADENOvsSQUA": "Adeno. vs Squa.",
                "PDL1": "PDL1",
                "DCR": "DCR",
                "OS6": "OS6",
                "OS24": "OS24",
                "ORR": "ORR",
                "CBR": "CBR",
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

        # Survival-specific configuration
        survival_config: dict[str, Any] = {
            **common_config,
            "task_mapping": {
                "OS": "OS",
                "PFS": "PFS",
            },
            "support_mapping": {
                "OS": 343,
                "PFS": 343,
            },
        }

        classification_stratified_config = TableConfig(
            output_file=str(self.config.output_dir / "classification_stratified.tex"),
            table_type="classification",
            include_stratified=True,
            classification_metrics=["f1", "recall"],
            caption="\\textbf{Classification performance across all tasks (cell-stratified).} Performance measured by Macro-F1 and Balanced Accuracy for binary classification tasks. Results show mean $\\pm$ standard deviation across label-stratified 5-fold cross-validation with additional stratification by cell cardinality. ``All'' refers to the use of all feature groups with correlation filtering ($\\rho < 0.95$). Bold values indicate best performance per task and metric.",
            label="tab:results_classification_stratified",
            **classification_config,
        )

        classification_non_stratified_config = TableConfig(
            output_file=str(
                self.config.output_dir / "classification_non_stratified.tex"
            ),
            table_type="classification",
            include_stratified=False,
            classification_metrics=["f1", "recall"],
            caption="\\textbf{Classification performance across all tasks.} Performance measured by Macro-F1 and Balanced Accuracy for binary classification tasks. Results show mean $\\pm$ standard deviation across label-stratified 5-fold cross-validation. ``All'' refers to the use of all feature groups with correlation filtering ($\\rho < 0.95$). Bold values indicate best performance per task and metric.",
            label="tab:results_classification",
            **classification_config,
        )

        survival_stratified_config = TableConfig(
            output_file=str(self.config.output_dir / "survival_stratified.tex"),
            table_type="survival",
            include_stratified=True,
            survival_metrics=["c_index"],
            caption="\\textbf{Survival analysis performance across survival tasks (cell-stratified).} Performance measured by Concordance Index (C-Index) for survival prediction tasks. Results show mean $\\pm$ standard deviation across label-stratified 5-fold cross-validation with additional stratification by cell cardinality. ``All'' refers to the use of all feature groups with correlation filtering ($\\rho < 0.95$). Bold values indicate best performance per task.",
            label="tab:results_survival_stratified",
            **survival_config,
        )

        survival_non_stratified_config = TableConfig(
            output_file=str(self.config.output_dir / "survival_non_stratified.tex"),
            table_type="survival",
            include_stratified=False,
            survival_metrics=["c_index"],
            caption="\\textbf{Survival analysis performance across survival tasks.} Performance measured by Concordance Index (C-Index) for survival prediction tasks. Results show mean $\\pm$ standard deviation across label-stratified 5-fold cross-validation. ``All'' refers to the use of all feature groups with correlation filtering ($\\rho < 0.95$). Bold values indicate best performance per task.",
            label="tab:results_survival",
            **survival_config,
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
        output_dir = Path(self.config.output_dir)
        plot_generator = PlotGenerator(output_dir)

        # Convert metrics to strings for plotting
        metrics_str = [str(m) for m in self.config.metrics]

        # Rename columns to match PlotGenerator expectations
        df_renamed = self.df.rename(columns={COLUMN_TASK: "task"})

        plot_generator.create_plots(
            df=df_renamed,
            metrics=metrics_str,
            group_by_column=COLUMN_EXPERIMENT_ID,
        )

    def create_tables(self, configs: list[TableConfig]):
        """Generate LaTeX tables based on configurations using shared TableGenerator."""
        # Rename columns to match TableGenerator expectations
        df_renamed = self.df.rename(
            columns={
                COLUMN_EXPERIMENT_ID: TableGenerator.COLUMN_EXPERIMENT_ID,
                COLUMN_TASK: TableGenerator.COLUMN_TASK,
                COLUMN_FEATURES: TableGenerator.COLUMN_FEATURES,
                COLUMN_MODEL: TableGenerator.COLUMN_MODEL,
                COLUMN_REG: TableGenerator.COLUMN_REG,
                COLUMN_STRA: TableGenerator.COLUMN_STRA,
            }
        )

        table_generator = TableGenerator(df_renamed)
        table_generator.create_tables(configs)

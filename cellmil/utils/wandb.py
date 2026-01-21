import wandb
import re
from typing import Any, cast
from collections import defaultdict
from dataclasses import dataclass
from cellmil.utils import logger

COLUMN_EXPERIMENT_ID = "EXPERIMENT_ID"
COLUMN_TASK = "TASK"
COLUMN_FEATURES = "FEATURES"
COLUMN_MODEL = "MODEL"
COLUMN_REG = "REGULARIZATION"
COLUMN_STRA = "STRATIFICATION"


@dataclass
class ExperimentComponents:
    """Parsed components of an experiment ID."""
    task: str
    features: str
    model: str
    regularization: str  # "REG" or "*"
    stratification: str  # "STRA" or "*"
    
    @property
    def has_regularization(self) -> bool:
        return self.regularization != "*"
    
    @property
    def has_stratification(self) -> bool:
        return self.stratification != "*"


class WandbClient:
    """Client for retrieving and processing wandb runs."""

    def __init__(self, team: str, projects: list[str], tasks: list[str] | None = None):
        """Initialize the WandB client.

        Args:
            team: Team name where the projects belong in wandb.
            projects: List of project names to retrieve runs from.
            tasks: Optional list of valid tasks for filtering runs.
        """
        self.team = team
        self.projects = projects
        self.tasks = tasks

        try:
            self.api = wandb.Api(timeout=10000)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize W&B API: {e}") from e

    def get_runs(self, preprocess: bool = True) -> list[Any]:
        """Retrieve wandb runs for configured projects and team.

        Args:
            preprocess: Whether to preprocess the runs (default: True).

        Returns:
            List of wandb runs (preprocessed if requested).
        """
        runs: list[Any] = []
        inaccessible: list[str] = []

        for project in self.projects:
            project_path = f"{self.team}/{project}"
            try:
                project_runs = cast(Any, self.api.runs(project_path))
                runs.extend(project_runs)
            except Exception as e:
                inaccessible.append(project_path)
                print(f"Warning: cannot access project '{project_path}': {e}")

        if inaccessible and not runs:
            raise RuntimeError(
                f"No accessible projects. Inaccessible: {', '.join(inaccessible)}"
            )
        elif inaccessible:
            print(f"Partial access. Inaccessible projects: {', '.join(inaccessible)}")

        return self._preprocess_runs(runs) if preprocess else runs

    def _preprocess_runs(self, runs: list[Any]) -> list[Any]:
        # Start with all runs
        new_runs = runs
        logger.info(f"Starting preprocessing with {len(new_runs)} total runs")

        # Filter out crashed or failed runs
        logger.info("Filtering runs with crashed or failed state")
        initial_count = len(new_runs)
        new_runs = [run for run in new_runs if run.state not in ["crashed", "failed"]]
        filtered_count = initial_count - len(new_runs)
        logger.info(
            f"Preprocessing runs: filtered out {filtered_count} crashed/failed runs, remaining runs {len(new_runs)}"
        )

        logger.info("Filtering runs, which task is ADENO or any other invalid task")
        filtered_runs: list[Any] = []
        for run in new_runs:
            try:
                exp_id = self.get_experiment_id(run)
                if self.get_task(exp_id):
                    filtered_runs.append(run)
            except ValueError:
                # Skip runs that don't have a valid task
                continue
        new_runs = filtered_runs

        logger.info(
            f"Preprocessing runs: filtered out ADENO runs, remaining runs {len(new_runs)}"
        )

        # Check that each experiment ID has exactly 6 runs (6-fold cross-validation including FINAL)
        logger.info(
            "Checking that each experiment ID has exactly 6 runs (including FINAL)"
        )
        experiment_counts: dict[str, list[Any]] = defaultdict(list)

        for run in new_runs:
            exp_id = self.get_experiment_id(run)
            experiment_counts[exp_id].append(run)

        expected_count = 6
        total_experiments = len(experiment_counts)
        incorrect_experiments: dict[str, int] = {}

        for exp_id, exp_runs in experiment_counts.items():
            if len(exp_runs) != expected_count:
                incorrect_experiments[exp_id] = len(exp_runs)

        if not incorrect_experiments:
            logger.info(
                f"All {total_experiments} experiment IDs have exactly {expected_count} runs"
            )
        else:
            logger.warning(
                f"{len(incorrect_experiments)} experiment ID(s) do not have exactly {expected_count} runs:"
            )
            for exp_id, count in incorrect_experiments.items():
                logger.warning(
                    f"  - {exp_id}: {count} runs (expected {expected_count})"
                )

            # Remove runs with incorrect experiment IDs
            incorrect_experiment_ids = set(incorrect_experiments.keys())
            new_runs = [
                run
                for run in new_runs
                if self.get_experiment_id(run) not in incorrect_experiment_ids
            ]
            logger.info(
                f"Removed runs with incorrect experiment IDs. New total runs: {len(new_runs)}"
            )

        # Check FINAL runs for DataLoader worker errors
        # logger.info("Checking FINAL runs for DataLoader worker errors")
        # experiments_with_errors: set[str] = set()

        # for exp_id, exp_runs in experiment_counts.items():
        #     # Skip if already marked as incorrect
        #     if exp_id in incorrect_experiments:
        #         continue

        #     # Find the FINAL run
        #     final_runs = [run for run in exp_runs if run.name.startswith("FINAL_")]
        #     if not final_runs:
        #         continue

        #     final_run = final_runs[0]
        #     if self._has_dataloader_error(final_run, exp_id):
        #         experiments_with_errors.add(exp_id)
        #         logger.warning(
        #             f"  - {exp_id}: DataLoader worker error detected in FINAL run"
        #         )

        # if experiments_with_errors:
        #     logger.warning(
        #         f"Found {len(experiments_with_errors)} experiment(s) with DataLoader worker errors in FINAL runs"
        #     )
        #     # Remove all runs from experiments with errors
        #     new_runs = [
        #         run
        #         for run in new_runs
        #         if self.get_experiment_id(run) not in experiments_with_errors
        #     ]
        #     logger.info(
        #         f"Removed runs with DataLoader errors. New total runs: {len(new_runs)}"
        #     )
        # else:
        #     logger.info("No DataLoader worker errors found in FINAL runs")

        # Now filter out FINAL_ runs after verifying the count
        logger.info(f"Filtering out FINAL_ runs with total runs {len(new_runs)}")
        initial_count = len(new_runs)
        new_runs = [run for run in new_runs if not run.name.startswith("FINAL_")]
        filtered_final = initial_count - len(new_runs)
        logger.info(
            f"Preprocessing runs: filtered out {filtered_final} FINAL_ runs, remaining runs {len(new_runs)}"
        )

        # Summary statistics
        run_counts = [
            len(exp_runs)
            for exp_id, exp_runs in experiment_counts.items()
            if exp_id not in incorrect_experiments
        ]
        if run_counts:
            logger.info(
                f"Total unique experiment IDs: {len(experiment_counts) - len(incorrect_experiments)}"
            )
            logger.info(f"Total runs after filtering: {len(new_runs)}")
            logger.info(
                f"Run counts - Min: {min(run_counts)}, Max: {max(run_counts)}, Mean: {sum(run_counts) / len(run_counts):.2f}"
            )

        return new_runs

    def get_experiment_id(self, run: Any) -> str:
        """Get the experiment ID from a run.

        Handles both formats:
        - FOLD_N_EXPERIMENTID_YYYY-MM-DD_HH-MM-SS
        - FINAL_EXPERIMENTID_YYYY-MM-DD_HH-MM-SS

        Args:
            run: The wandb run object

        Returns:
            The extracted experiment ID
        """
        name = run.name

        # Remove FOLD_N_ prefix if present
        if name.startswith("FOLD_"):
            # Skip "FOLD_N_"
            name = name.split("_", 2)[2] if len(name.split("_", 2)) >= 3 else name
        # Remove FINAL_ prefix if present
        elif name.startswith("FINAL_"):
            # Skip "FINAL_"
            name = name[6:]

        # Now extract experiment ID (everything before the timestamp _YYYY-MM-DD)
        # Match everything until we find _YYYY (4 digits starting with 20)
        match = re.match(r"(.+?)_\d{4}-\d{2}-\d{2}", name)
        if match:
            experiment_id = match.group(1)
        else:
            # Fallback: use everything before the last timestamp-like pattern
            experiment_id = name

        return experiment_id

    @staticmethod
    def parse_experiment_components(experiment_id: str) -> ExperimentComponents:
        """Parse an experiment ID into its components.
        
        Format: TASK+FEATURES+MODEL+REG+STRA
        Example: DCR+ALL+ABMIL+REG+* or OS+PYRAD+HEAD4TYPE+*+STRA
        
        Args:
            experiment_id: The experiment ID string
        
        Returns:
            ExperimentComponents with parsed values
        
        Raises:
            ValueError: If experiment ID doesn't match expected format
        """
        parts = experiment_id.split("+")
        if len(parts) != 5:
            raise ValueError(
                f"Experiment ID '{experiment_id}' does not match expected format TASK+FEATURES+MODEL+REG+STRA"
            )
        
        return ExperimentComponents(
            task=parts[0],
            features=parts[1],
            model=parts[2],
            regularization=parts[3],
            stratification=parts[4],
        )

    def get_task(self, experiment_id: str) -> str | None:
        """Get the task associated with a given experiment ID.

        Args:
            experiment_id: The ID of the experiment

        Returns:
            The name of the task, or None if not valid

        Raises:
            ValueError: If tasks are specified and the experiment ID doesn't correspond to a known task
        """
        task = experiment_id.split("+")[0]

        # If no tasks are specified, infer the task from the experiment ID
        if self.tasks is None:
            return task

        # If tasks are specified, validate against the list
        if task in self.tasks:
            return task
        else:
            raise ValueError(
                f"Experiment ID '{experiment_id}' does not correspond to a known task."
            )

    def _has_dataloader_error(self, run: Any, exp_id: str) -> bool:
        """Check if a run has a DataLoader worker error in its logs.

        Args:
            run: A wandb run object
            exp_id: The experiment ID to look for in the error message

        Returns:
            True if the error is found, False otherwise
        """
        try:
            # Get the run's logs
            logs = run.history(keys=["_log"])
            if logs.empty or "_log" not in logs.columns:
                return False

            # Check each log entry for the error pattern
            for log_entry in logs["_log"].dropna():
                if (
                    "ERROR - Error in experiment" in str(log_entry)
                    and exp_id in str(log_entry)
                    and "DataLoader worker" in str(log_entry)
                    and "exited unexpectedly" in str(log_entry)
                ):
                    return True

            return False
        except Exception as e:
            logger.warning(f"Could not check logs for run {run.name}: {e}")
            return False

    @staticmethod
    def get_metric(run: Any, metric: str) -> float:
        """Get the highest validation metric across all epochs for a given run.

        Args:
            run: A wandb run object
            metric: The metric name (e.g., "f1", "c_index", "balacc")

        Returns:
            The highest validation metric score

        Raises:
            ValueError: If the metric is not found or has no valid values
        """
        history = run.history(keys=[f"val/{metric}"])
        if history.empty or f"val/{metric}" not in history.columns:
            raise ValueError(
                f"Run {run.name} has no 'val/{metric}' metric in its history"
            )

        # Drop NaN values and get the maximum
        metric_values = history[f"val/{metric}"].dropna()

        if metric_values.empty:
            raise ValueError(
                f"Run {run.name} has no valid 'val/{metric}' values in its history"
            )

        max_metric = float(metric_values.max())

        return max_metric

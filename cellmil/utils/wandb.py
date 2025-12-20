import wandb
import re
from typing import Any, cast
from collections import defaultdict
from cellmil.utils import logger

COLUMN_EXPERIMENT_ID = "EXPERIMENT_ID"
COLUMN_TASK = "TASK"


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
            self.api = wandb.Api()
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
        # Filter all runs that start with FINAL_
        logger.info(
            f"Preprocessing runs: filtering out FINAL_ runs with total runs {len(runs)}"
        )

        new_runs = [run for run in runs if not run.name.startswith("FINAL_")]

        logger.info(
            f"Preprocessing runs: filtered out FINAL_ runs, remaining runs {len(new_runs)}"
        )

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

        # Check that each experiment ID has exactly 5 runs (5-fold cross-validation)
        logger.info("Checking that each experiment ID has exactly 5 runs")
        experiment_counts: dict[str, list[Any]] = defaultdict(list)

        for run in new_runs:
            exp_id = self.get_experiment_id(run)
            experiment_counts[exp_id].append(run)

        expected_count = 5
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
                logger.warning(f"  - {exp_id}: {count} runs (expected {expected_count})")

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
        
        Args:
            run: The wandb run object
        
        Returns:
            The extracted experiment ID
        """
        parts = run.name.split("_", 2)
        if len(parts) >= 3:
            # Extract from parts[2] until we find _\d
            match = re.match(r"([^_]+(?:_[^0-9][^_]*)*)", parts[2])
            experiment_id = match.group(1) if match else parts[2]
        else:
            experiment_id = run.name

        return experiment_id
    
    def get_task(self, experiment_id: str) -> str | None:
        """Get the task associated with a given experiment ID.
        
        Args:
            experiment_id: The ID of the experiment
        
        Returns:
            The name of the task
        
        Raises:
            ValueError: If the experiment ID doesn't correspond to a known task
        """
        task = experiment_id.split("+")[0]
        if self.tasks is not None and task in self.tasks:
            return task
        else:
            raise ValueError(
                f"Experiment ID '{experiment_id}' does not correspond to a known task."
            )
    
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

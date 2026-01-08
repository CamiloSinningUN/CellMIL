from pathlib import Path
from typing import Any, cast
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from lightning import Trainer
from sklearn.metrics import (
    accuracy_score,  # type: ignore
    precision_recall_fscore_support,  # type: ignore
    roc_auc_score,  # type: ignore
)
from torch.utils.data import DataLoader as DataLoaderTorch
from torch_geometric.loader import DataLoader as DataLoaderPyG  # type: ignore
from scipy import stats  # type: ignore

from cellmil.datamodels.datasets.cell_gnn_mil_dataset import CellGNNMILDataset
from cellmil.datamodels.datasets.patch_gnn_mil_dataset import PatchGNNMILDataset
from cellmil.datamodels.datasets.patch_mil_dataset import PatchMILDataset
from cellmil.datamodels.datasets.cell_mil_dataset import CellMILDataset
from cellmil.interfaces.EvaluationExternalValidatorConfig import (
    EvaluationExternalValidatorConfig,
    FinalModel,
    AggregationMethod,
)
from cellmil.interfaces.TableConfig import TableConfig
from cellmil.datamodels.model import ModelStorage
from cellmil.datamodels.datasets import MILDataset
from cellmil.evaluation.visualization import TableGenerator, PlotGenerator
from cellmil.utils import logger
from cellmil.utils.train.metrics import ConcordanceIndex
from cellmil.utils.train.evals.utils import is_survival_model
from cellmil.utils.train import get_lit_model_creator
from cellmil.utils.train import get_extractors_from_name, preprocess_df
from cellmil.interfaces.CellSegmenterConfig import ModelType
from cellmil.interfaces.GraphCreatorConfig import GraphCreatorType


class ExternalValidator:
    """
    External validator for evaluating trained models on independent test sets.

    This class:
    1. Loads models from directories (using ModelStorage)
    2. Runs them on external validation datasets
    3. Calculates metrics from predictions
    4. Generates plots and tables similar to EvaluationReporter
    """

    def __init__(self, config: EvaluationExternalValidatorConfig):
        """
        Initialize external validator.

        Args:
            config: Configuration for external validation
        """
        self.config = config
        self.df = pd.DataFrame()

        # TODO: Make configurable
        self.n_bins = 4

        # TODO: ------

        # Ensure output directory exists
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"External Validator initialized with output dir: {self.config.output_dir}"
        )

    def validate(self):
        """
        Run external validation on all models and generate reports.

        This is the main entry point that:
        1. Discovers all model directories
        2. Loads external validation dataset for each model
        3. Runs predictions and calculates metrics
        4. Aggregates results into DataFrame
        5. Generates plots and tables
        """
        logger.info("Starting external validation process...")

        # Discover all model directories
        model_dirs = self._discover_model_directories()
        logger.info(f"Found {len(model_dirs)} model directories to validate")

        # Process each model and collect results
        self._process_models(model_dirs)

        if self.df.empty:
            logger.error(
                "No models were successfully processed. Cannot generate reports."
            )
            return

        logger.info(f"Successfully processed {len(self.df)} model configurations")

        # Generate plots and tables
        self.create_plots()
        self.create_tables()

        logger.info("External validation completed successfully!")

    def _discover_model_directories(self) -> list[Path]:
        """
        Discover all model directories in the models_dir.

        Returns:
            List of paths to valid model directories
        """
        models_dir = Path(self.config.models_dir)

        if not models_dir.exists():
            raise FileNotFoundError(f"Models directory not found: {models_dir}")

        # Find directories containing experiment_metadata.json
        model_dirs: list[Path] = []
        for path in models_dir.iterdir():
            if path.is_dir() and (path / "experiment_metadata.json").exists():
                model_dirs.append(path)

        return sorted(model_dirs)

    def _process_models(self, model_dirs: list[Path]):
        """
        Process all model directories and collect results.

        Args:
            model_dirs: List of model directory paths
        """

        def process_single_model(model_dir: Path) -> dict[str, Any] | None:
            """Process a single model directory."""
            try:
                return self._evaluate_model(model_dir)
            except Exception as e:
                logger.error(f"Failed to process model {model_dir.name}: {e}")
                return None

        results: list[dict[str, Any]] = []

        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=8) as executor:
            future_to_dir = {
                executor.submit(process_single_model, model_dir): model_dir
                for model_dir in model_dirs
            }

            with tqdm(total=len(model_dirs), desc="Evaluating models") as pbar:
                for future in as_completed(future_to_dir):
                    result = future.result()
                    if result is not None:
                        results.append(result)
                    pbar.update(1)

        # Convert to DataFrame
        self.df = pd.DataFrame(results)
        logger.info(f"Collected results from {len(results)} models")

    def _evaluate_model(self, model_dir: Path) -> dict[str, Any]:
        """
        Evaluate a single model on external validation dataset.

        Args:
            model_dir: Path to model directory

        Returns:
            Dictionary with experiment info and metrics
        """
        # Load model storage
        model_storage = ModelStorage.from_directory(model_dir)

        if not model_storage.experiment_metadata:
            raise ValueError(f"No experiment metadata found for {model_dir}")

        experiment_name = model_storage.experiment_metadata.name
        logger.info(f"Evaluating model: {experiment_name}")

        # Parse experiment name: TASK+FEATURES+MODEL+REG+STRA
        components = self._parse_experiment_name(experiment_name)

        # Create dataset for this model configuration
        dataset = self._create_dataset(
            task=components["task"],
            features=components["features"],
            model=components["model"],
        )

        # Create dataloader based on dataset type
        if isinstance(dataset, (CellGNNMILDataset, PatchGNNMILDataset)):
            dataloader = DataLoaderPyG(dataset, batch_size=1, shuffle=False)
        else:
            dataloader = DataLoaderTorch(dataset, batch_size=1, shuffle=False)

        # Get lit_model_creator
        lit_model_creator = get_lit_model_creator(
            model=components["model"],
            task=components["task"],
            n_bins=self.n_bins,
            feature=components["features"],
            df=self._load_metadata_df(),
            regularization=(components["reg"] == "REG"),
        )

        # Run predictions
        predictions_df = self._run_predictions(
            model_storage=model_storage,
            dataloader=dataloader,
            lit_model_creator=lit_model_creator,
        )

        # Calculate metrics
        task_type = self._detect_task_type(predictions_df)
        metrics = self._calculate_metrics(predictions_df, task_type)

        # Build result dictionary
        result: dict[str, Any] = {
            "experiment_id": experiment_name,
            "task": components["task"],
            "features": components["features"],
            "model": components["model"],
            "reg": components["reg"],
            "stra": components["stra"],
            **metrics,
        }

        logger.info(f"Completed evaluation for {experiment_name}: {metrics}")

        return result

    def _parse_experiment_name(self, name: str) -> dict[str, str]:
        """
        Parse experiment name in format: TASK+FEATURES+MODEL+REG+STRA

        Args:
            name: Experiment name

        Returns:
            Dictionary with parsed components
        """
        parts = name.split("+")
        if len(parts) != 5:
            raise ValueError(f"Invalid experiment name format: {name}")

        return {
            "task": parts[0],
            "features": parts[1],
            "model": parts[2],
            "reg": parts[3],
            "stra": parts[4],
        }

    def _create_dataset(
        self, task: str, features: str, model: str
    ) -> CellMILDataset | PatchMILDataset:
        """
        Create external validation dataset.

        Args:
            task: Task name
            features: Feature type
            model: Model name

        Returns:
            MILDataset for external validation
        """

        # Load metadata
        df = self._load_metadata_df()
        df = preprocess_df(df, task)

        # Get extractors
        extractors = get_extractors_from_name(features)

        # Create dataset
        dataset = MILDataset(
            root=self.config.root_dir,
            label=task if task not in ["OS", "PFS"] else ("duration", "event"),
            folder=self.config.dataset_dir,
            data=df,
            extractor=extractors,
            segmentation_model=ModelType.cellvit,
            graph_creator=GraphCreatorType.delaunay_radius,
            cell_type=True if model == "HEAD4TYPE" else False,
        )

        return dataset

    def _load_metadata_df(self) -> pd.DataFrame:
        """Load metadata DataFrame."""
        return pd.read_excel(self.config.dp_metadata_file)  # type: ignore

    def _run_predictions(
        self,
        model_storage: ModelStorage,
        dataloader: Any,
        lit_model_creator: Any,
    ) -> pd.DataFrame:
        """
        Run predictions using either final model or ensemble.

        Args:
            model_storage: Model storage object
            dataloader: Data loader
            lit_model_creator: Function to create lightning module

        Returns:
            DataFrame with predictions and labels
        """
        if self.config.final_model == FinalModel.final:
            return self._predict_final(model_storage, dataloader, lit_model_creator)
        else:
            return self._predict_ensemble(
                model_storage,
                dataloader,
                lit_model_creator,
                self.config.aggregation_method,
            )

    def _predict_final(
        self,
        model_storage: ModelStorage,
        dataloader: Any,
        lit_model_creator: Any,
    ) -> pd.DataFrame:
        """Generate predictions using final model."""
        checkpoint_path = model_storage.load_final_checkpoint()
        model = lit_model_creator().load_from_checkpoint(checkpoint_path)
        model.eval()

        trainer = Trainer(
            accelerator="auto",
            devices=1,
            logger=False,
            enable_progress_bar=False,
            enable_checkpointing=False,
        )

        return self._get_predictions_from_trainer(trainer, model, dataloader)

    def _predict_ensemble(
        self,
        model_storage: ModelStorage,
        dataloader: Any,
        lit_model_creator: Any,
        method: AggregationMethod,
    ) -> pd.DataFrame:
        """Generate ensemble predictions from all folds."""
        fold_indices = model_storage.list_folds()

        trainer = Trainer(
            accelerator="auto",
            devices=1,
            logger=False,
            enable_progress_bar=False,
            enable_checkpointing=False,
        )

        all_fold_predictions: list[pd.DataFrame] = []

        for fold_idx in fold_indices:
            checkpoint_path = model_storage.load_fold_checkpoint(fold_idx)
            model = lit_model_creator().load_from_checkpoint(checkpoint_path)
            model.eval()

            fold_preds = self._get_predictions_from_trainer(trainer, model, dataloader)
            all_fold_predictions.append(fold_preds)

        return self._aggregate_fold_predictions(all_fold_predictions, method)

    def _get_predictions_from_trainer(
        self,
        trainer: Trainer,
        model: Any,
        dataloader: Any,
    ) -> pd.DataFrame:
        """Extract predictions using Lightning Trainer."""
        y_pred = cast(list[Any], trainer.predict(model, dataloader))

        # Extract true labels based on dataloader type
        if isinstance(dataloader, DataLoaderPyG):
            y_true = [batch.y for batch in dataloader]
        else:
            y_true = [batch[-1] for batch in dataloader]

        # Check if survival model
        is_surv = is_survival_model(model)

        if is_surv:
            return self._process_survival_predictions(y_pred, y_true)
        else:
            return self._process_classification_predictions(y_pred, y_true)

    def _process_survival_predictions(
        self, y_pred: list[Any], y_true: list[Any]
    ) -> pd.DataFrame:
        """Process survival model predictions."""

        def _extract_survival_tensors(
            target: Any,
        ) -> tuple[torch.Tensor, torch.Tensor] | None:
            """Normalize different target formats to (duration, event) tensors."""

            def _to_tensor(value: Any) -> torch.Tensor:
                tensor = torch.as_tensor(value)
                if tensor.ndim == 0:
                    tensor = tensor.unsqueeze(0)
                return tensor

            if isinstance(target, dict):
                key_duration = next(  # type: ignore
                    (
                        k
                        for k in target  # type: ignore
                        if k.lower() in {"duration", "durations", "time"}  # type: ignore
                    ),
                    None,
                )
                key_event = next(  # type: ignore
                    (k for k in target if k.lower() in {"event", "events", "status"}),  # type: ignore
                    None,
                )
                if key_duration is not None and key_event is not None:
                    return _to_tensor(target[key_duration]), _to_tensor(
                        target[key_event]
                    )

            if isinstance(target, (list, tuple)) and len(target) == 2:  # type: ignore
                return _to_tensor(target[0]), _to_tensor(target[1])

            if torch.is_tensor(target):
                tensor = target
                if tensor.ndim == 1 and tensor.numel() == 2:
                    return tensor[0].view(1), tensor[1].view(1)
                if tensor.ndim >= 1 and tensor.shape[-1] == 2:
                    durations = tensor[..., 0].reshape(-1)
                    events = tensor[..., 1].reshape(-1)
                    return durations, events

            return None

        # Extract durations and events from labels
        durations_list: list[torch.Tensor] = []
        events_list: list[torch.Tensor] = []

        for label in y_true:
            parsed = _extract_survival_tensors(label)
            if parsed is None:
                logger.warning(f"Unexpected label format: {label}")
                continue
            dur_tensor, evt_tensor = parsed
            durations_list.append(dur_tensor)
            events_list.append(evt_tensor)

        if not durations_list or not events_list:
            raise ValueError("No survival data found")

        # Convert predictions to tensor (logits)
        if len(y_pred) > 0:
            if isinstance(y_pred[0], torch.Tensor):
                # Predictions are logits with shape [1, num_bins] per sample
                logits = torch.cat([pred.cpu() for pred in y_pred], dim=0)  # type: ignore
            else:
                logger.error("Unexpected prediction format")
                logits = torch.zeros((len(durations_list), 1))  # type: ignore
        else:
            logger.error("No predictions returned")
            logits = torch.zeros((len(durations_list), 1))  # type: ignore

        # Convert durations and events to tensors
        durations = torch.cat([d.cpu().flatten() for d in durations_list])  # type: ignore
        events = torch.cat([e.cpu().flatten() for e in events_list])  # type: ignore

        # Store logits directly - ConcordanceIndex will convert them to risk scores internally
        # We need to store each sample's logits as a list for proper DataFrame storage
        logits_list = [logits[i].numpy().tolist() for i in range(logits.shape[0])]  # type: ignore

        return pd.DataFrame(
            {
                "duration": durations.numpy(),  # type: ignore
                "event": events.numpy(),  # type: ignore
                "logits": logits_list,  # Store logits, not risk scores
            }
        )

    def _process_classification_predictions(
        self, y_pred: list[Any], y_true: list[Any]
    ) -> pd.DataFrame:
        """Process classification model predictions."""
        # Extract predictions
        if isinstance(y_pred[0], torch.Tensor):
            y_pred_flat = [pred.cpu().numpy().flatten()[0] for pred in y_pred]  # type: ignore
        else:
            y_pred_flat = [  # type: ignore
                pred.flatten()[0] if hasattr(pred, "flatten") else pred  # type: ignore
                for pred in y_pred  # type: ignore
            ]

        # Extract true labels
        if y_true and isinstance(y_true[0], torch.Tensor):
            y_true_flat = [true.cpu().numpy().flatten()[0] for true in y_true]
        else:
            y_true_flat = [
                true.flatten()[0] if hasattr(true, "flatten") else true
                for true in y_true
            ]

        return pd.DataFrame(
            {
                "y_true": np.array(y_true_flat),
                "y_pred": np.array(y_pred_flat),  # type: ignore
            }
        )

    def _aggregate_fold_predictions(
        self,
        fold_predictions: list[pd.DataFrame],
        method: AggregationMethod,
    ) -> pd.DataFrame:
        """Aggregate predictions from multiple folds."""
        # Check if survival or classification
        is_survival = "logits" in fold_predictions[0].columns

        if is_survival:
            # For survival: aggregate logits (not risk scores)
            # Convert logits lists back to tensors
            all_logits: list[torch.Tensor] = []
            for df in fold_predictions:
                logits_list = df["logits"].tolist()
                logits_tensor = torch.tensor(logits_list, dtype=torch.float32)
                all_logits.append(logits_tensor)

            # Stack and aggregate
            stacked_logits = torch.stack(all_logits)  # [n_folds, n_samples, n_bins]

            if (
                method == AggregationMethod.mean
                or method == AggregationMethod.weighted_mean
            ):
                aggregated_logits = torch.mean(stacked_logits, dim=0)
            elif method == AggregationMethod.median:
                aggregated_logits = torch.median(stacked_logits, dim=0)[0]
            else:
                raise ValueError(
                    f"Unsupported aggregation method for survival: {method}"
                )

            # Convert back to list format for DataFrame
            aggregated_logits_list = [
                aggregated_logits[i].numpy().tolist()  # type: ignore
                for i in range(aggregated_logits.shape[0])
            ]

            return pd.DataFrame(
                {
                    "duration": fold_predictions[0]["duration"].values,  # type: ignore
                    "event": fold_predictions[0]["event"].values,  # type: ignore
                    "logits": aggregated_logits_list,
                }
            )
        else:
            # For classification: use majority voting
            all_preds = np.stack([df["y_pred"].values for df in fold_predictions])  # type: ignore

            # Use mode for majority vote

            y_pred = stats.mode(all_preds, axis=0, keepdims=False)[0]  # type: ignore

            return pd.DataFrame(
                {
                    "y_true": fold_predictions[0]["y_true"].values,  # type: ignore
                    "y_pred": y_pred,
                }
            )

    def _detect_task_type(self, predictions_df: pd.DataFrame) -> str:
        """Detect whether task is classification or survival."""
        if "logits" in predictions_df.columns:
            return "survival"
        else:
            return "classification"

    def _calculate_metrics(
        self, predictions_df: pd.DataFrame, task_type: str
    ) -> dict[str, float]:
        """Calculate metrics based on task type."""
        if task_type == "survival":
            return self._calculate_survival_metrics(predictions_df)
        else:
            return self._calculate_classification_metrics(predictions_df)

    # TODO: Review this
    def _calculate_survival_metrics(
        self, predictions_df: pd.DataFrame
    ) -> dict[str, float]:
        """Calculate survival metrics (C-Index)."""
        durations = torch.tensor(predictions_df["duration"].values)  # type: ignore
        events = torch.tensor(predictions_df["event"].values)  # type: ignore
        risk_scores = torch.tensor(predictions_df["risk_score"].values)  # type: ignore

        # Reshape risk_scores for ConcordanceIndex if needed
        if risk_scores.ndim == 1:
            risk_scores = risk_scores.unsqueeze(-1)

        c_index_metric = ConcordanceIndex()
        c_index_metric.update(risk_scores, (durations, events))
        c_index = c_index_metric.compute()

        return {
            "c_index": float(c_index),
            "n_samples": len(durations),
            "n_events": int(events.sum()),
        }

    def _calculate_classification_metrics(
        self, predictions_df: pd.DataFrame
    ) -> dict[str, float]:
        """Calculate classification metrics."""
        y_true = cast(np.ndarray[Any, Any], predictions_df["y_true"].values)  # type: ignore
        y_pred = cast(np.ndarray[Any, Any], predictions_df["y_pred"].values)  # type: ignore

        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = cast(
            tuple[float, float, float, float],
            precision_recall_fscore_support(  # type: ignore
                y_true, y_pred, average="macro", zero_division=0
            ),
        )

        metrics = {
            "accuracy": float(accuracy),
            "f1": float(f1),
            "recall": float(recall),
            "precision": float(precision),
        }

        # Try to calculate AUC if probabilities are available
        prob_columns = [
            col for col in predictions_df.columns if col.startswith("prob_class_")
        ]
        if prob_columns:
            try:
                n_classes = len(prob_columns)
                if n_classes == 2:
                    y_score = cast(list[Any], predictions_df["prob_class_1"].values)  # type: ignore
                    auc = roc_auc_score(y_true, y_score)
                    metrics["auc"] = float(auc)
            except Exception as e:
                logger.warning(f"Failed to calculate AUC: {e}")

        return metrics

    def create_plots(self):
        """Generate plots for all metrics."""
        plot_generator = PlotGenerator(self.config.output_dir)

        # Get metrics from config
        metrics_str = [str(m) for m in self.config.metrics]

        # Rename columns to match PlotGenerator expectations
        df_renamed = self.df.rename(columns={"task": "task"})

        plot_generator.create_plots(
            df=df_renamed,
            metrics=metrics_str,
            group_by_column="experiment_id",
        )

        logger.info("Plots generated successfully")

    def create_tables(self):
        """Generate LaTeX tables."""
        # Common configuration for all tables
        base_config: dict[str, Any] = {
            "baseline_features": ["RESNET", "GIGAPATH"],
            "task_mapping": {
                "ADENOvsSQUA": "Adeno. vs Squa.",
                "PDL1": "PDL1",
                "DCR": "DCR",
                "OS6": "OS6",
                "OS24": "OS24",
                "ORR": "ORR",
                "CBR": "CBR",
            },
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

        # Define table configurations
        configs = [
            TableConfig(
                output_file=str(
                    self.config.output_dir / "external_classification_stratified.tex"
                ),
                table_type="classification",
                include_stratified=True,
                classification_metrics=["f1", "recall"],
                caption="\\textbf{External validation - Classification performance (cell-stratified).}",
                label="tab:external_classification_stratified",
                **base_config,
            ),
            TableConfig(
                output_file=str(
                    self.config.output_dir
                    / "external_classification_non_stratified.tex"
                ),
                table_type="classification",
                include_stratified=False,
                classification_metrics=["f1", "recall"],
                caption="\\textbf{External validation - Classification performance.}",
                label="tab:external_classification",
                **base_config,
            ),
            TableConfig(
                output_file=str(
                    self.config.output_dir / "external_survival_stratified.tex"
                ),
                table_type="survival",
                include_stratified=True,
                survival_metrics=["c_index"],
                caption="\\textbf{External validation - Survival analysis performance (cell-stratified).}",
                label="tab:external_survival_stratified",
                **base_config,
            ),
            TableConfig(
                output_file=str(
                    self.config.output_dir / "external_survival_non_stratified.tex"
                ),
                table_type="survival",
                include_stratified=False,
                survival_metrics=["c_index"],
                caption="\\textbf{External validation - Survival analysis performance.}",
                label="tab:external_survival",
                **base_config,
            ),
        ]

        # Rename columns to match TableGenerator expectations
        df_renamed = self.df.rename(
            columns={
                "experiment_id": TableGenerator.COLUMN_EXPERIMENT_ID,
                "task": TableGenerator.COLUMN_TASK,
                "features": TableGenerator.COLUMN_FEATURES,
                "model": TableGenerator.COLUMN_MODEL,
                "reg": TableGenerator.COLUMN_REG,
                "stra": TableGenerator.COLUMN_STRA,
            }
        )

        table_generator = TableGenerator(df_renamed)
        table_generator.create_tables(configs)

        logger.info("Tables generated successfully")

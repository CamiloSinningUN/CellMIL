from pathlib import Path
from typing import Any, cast
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback
import json
import pandas as pd
import numpy as np
import torch
import torchmetrics
from tqdm import tqdm
from lightning import Trainer
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
        self.use_parallelism = False if self.config.final_model == FinalModel.ensemble else True  # It has too many error cases currently with ensenmble
        self.max_workers = 8
        self.use_gpu = False
        self.dataloader_num_workers = 0 if self.use_parallelism else 8
        
        self.parallelize_ensemble_folds = True  # Process all folds in parallel for ensemble
        self.ensemble_max_workers = 5  # One worker per fold (typically 5 folds)
        self.ensemble_dataloader_workers = 0
        
        # CPU thread limiting (to avoid overusing resources)
        self.cpu_threads = 32 # Limit CPU threads per model
        # TODO: ------

        # Limit CPU thread usage to avoid overwhelming the system
        if not self.use_gpu:
            torch.set_num_threads(self.cpu_threads)
            torch.set_num_interop_threads(self.cpu_threads)
            logger.info(f"Limited PyTorch to {self.cpu_threads} CPU threads")

        # Ensure output directory exists
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define log file path
        self.log_file = self.config.output_dir / "results_log.json"
        
        # Create predictions directory
        self.predictions_dir = self.config.output_dir / "predictions"
        self.predictions_dir.mkdir(parents=True, exist_ok=True)

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

        # Check if we need to run all aggregation methods
        if self.config.final_model == FinalModel.ensemble and self.config.aggregation_method == AggregationMethod.everything:
            logger.info("Running validation with all aggregation methods...")
            methods = [AggregationMethod.majority, AggregationMethod.median]
            
            for method in methods:
                logger.info(f"\n{'='*60}")
                logger.info(f"Processing with aggregation method: {method}")
                logger.info(f"{'='*60}")
                
                # Create subfolder for this method
                method_output_dir = self.config.output_dir / str(method)
                method_output_dir.mkdir(parents=True, exist_ok=True)
                
                # Temporarily override aggregation method, output directory, and log file
                original_method = self.config.aggregation_method
                original_output_dir = self.config.output_dir
                original_log_file = self.log_file
                original_predictions_dir = self.predictions_dir
                self.config.aggregation_method = method
                self.config.output_dir = method_output_dir
                self.log_file = method_output_dir / "results_log.json"
                self.predictions_dir = method_output_dir / "predictions"
                self.predictions_dir.mkdir(parents=True, exist_ok=True)
                
                # Process models with this method
                self._process_models(model_dirs)
                
                if self.df.empty:
                    logger.warning(
                        f"No models were successfully processed for method {method}. Skipping reports."
                    )
                else:
                    logger.info(f"Successfully processed {len(self.df)} model configurations for {method}")
                    # Generate plots and tables for this method
                    self.create_plots()
                    self.create_tables()
                
                # Restore original configuration
                self.config.aggregation_method = original_method
                self.config.output_dir = original_output_dir
                self.log_file = original_log_file
                self.predictions_dir = original_predictions_dir
                
            logger.info("\nExternal validation completed for all aggregation methods!")
        else:
            # Single aggregation method
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

    def _load_results_log(self) -> dict[str, dict[str, Any]]:
        """
        Load existing results from log file.
        
        Returns:
            Dictionary mapping experiment_id to result dict
        """
        if not self.log_file.exists():
            logger.info("No existing results log found, starting fresh")
            return {}
        
        try:
            with open(self.log_file, 'r') as f:
                results_list = json.load(f)
            
            # Convert list to dict keyed by experiment_id
            results_dict = {r["experiment_id"]: r for r in results_list}
            logger.info(f"Loaded {len(results_dict)} existing results from log file")
            return results_dict
        except Exception as e:
            logger.warning(f"Failed to load results log: {e}. Starting fresh.")
            return {}
    
    def _save_result_to_log(self, result: dict[str, Any]):
        """
        Save a single result to the log file (append mode).
        
        Args:
            result: Result dictionary to save
        """
        try:
            # Load existing results
            existing_results = self._load_results_log()
            
            # Update with new result
            existing_results[result["experiment_id"]] = result
            
            # Write back to file
            with open(self.log_file, 'w') as f:
                json.dump(list(existing_results.values()), f, indent=2)
            
            logger.debug(f"Saved result for {result['experiment_id']} to log file")
        except Exception as e:
            logger.error(f"Failed to save result to log: {e}")
    
    def _is_model_processed(self, experiment_id: str, existing_results: dict[str, dict[str, Any]]) -> bool:
        """
        Check if a model has already been processed.
        
        Args:
            experiment_id: Experiment ID to check
            existing_results: Dictionary of existing results
        
        Returns:
            True if model has been processed, False otherwise
        """
        return experiment_id in existing_results

    def _process_models(self, model_dirs: list[Path]):
        """
        Process all model directories and collect results.
        
        This method resets self.df before processing to ensure clean results
        for each aggregation method when using 'everything' mode.

        Args:
            model_dirs: List of model directory paths
        """
        # Load existing results from log
        existing_results = self._load_results_log()
        
        # Start with existing results
        results: list[dict[str, Any]] = list(existing_results.values())
        logger.info(f"Starting with {len(results)} existing results")
        
        # Filter out already processed models
        models_to_process = []
        for model_dir in model_dirs:
            # Need to peek at experiment name to check if processed
            try:
                model_storage = ModelStorage.from_directory(model_dir)
                if model_storage.experiment_metadata:
                    experiment_name = model_storage.experiment_metadata.name
                    if self._is_model_processed(experiment_name, existing_results):
                        logger.info(f"Skipping already processed model: {experiment_name}")
                        continue
            except Exception as e:
                logger.warning(f"Could not check if {model_dir.name} is processed: {e}")
            
            models_to_process.append(model_dir)
        
        logger.info(f"Found {len(models_to_process)} models to process (out of {len(model_dirs)} total)")

        def process_single_model(model_dir: Path) -> dict[str, Any] | None:
            """Process a single model directory."""
            try:
                result = self._evaluate_model(model_dir)
                if result is not None:
                    # Save to log immediately after successful processing
                    self._save_result_to_log(result)
                return result
            except Exception as e:
                logger.error(
                    f"Failed to process model {model_dir.name}: {e}\n"
                    f"Traceback:\n{traceback.format_exc()}"
                )
                return None

        if self.use_parallelism:
            # Use ThreadPoolExecutor for parallel processing
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_dir = {
                    executor.submit(process_single_model, model_dir): model_dir
                    for model_dir in models_to_process
                }

                with tqdm(total=len(models_to_process), desc="Evaluating models") as pbar:
                    for future in as_completed(future_to_dir):
                        model_dir = future_to_dir[future]
                        try:
                            # Add timeout of 5 minutes per model
                            result = future.result(timeout=300)
                            if result is not None:
                                results.append(result)
                        except TimeoutError:
                            logger.error(
                                f"Model {model_dir.name} timed out after 300 seconds"
                            )
                        except Exception as e:
                            logger.error(
                                f"Unexpected error for model {model_dir.name}: {e}\n"
                                f"Traceback:\n{traceback.format_exc()}"
                            )
                        pbar.update(1)
        else:
            # Sequential processing
            with tqdm(total=len(models_to_process), desc="Evaluating models") as pbar:
                for model_dir in models_to_process:
                    result = process_single_model(model_dir)
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

        # Load and preprocess metadata for this task (needed for lit_model_creator)
        df = self._load_metadata_df()
        df = preprocess_df(df, components["task"])

        # For final model, create dataset once
        # For ensemble, we'll create dataset per fold with fold-specific transforms
        if self.config.final_model == FinalModel.final:
            # Create dataset for this model configuration with transforms
            dataset = self._create_dataset(
                task=components["task"],
                features=components["features"],
                model=components["model"],
                model_storage=model_storage,
            )

            # Get input dimension from transformed sample
            sample_data = dataset[0]
            if isinstance(dataset, (CellGNNMILDataset, PatchGNNMILDataset)):
                input_dim = sample_data.x.shape[1]  # type: ignore
            else:
                input_dim = sample_data[0].shape[1]

            logger.info(f"Dataset created with input_dim: {input_dim}")

            # Create dataloader based on dataset type
            # Use num_workers=0 to avoid multiprocessing conflicts with ThreadPoolExecutor
            if isinstance(dataset, (CellGNNMILDataset, PatchGNNMILDataset)):
                dataloader = DataLoaderPyG(dataset, batch_size=1, shuffle=False, num_workers=self.dataloader_num_workers)
            else:
                dataloader = DataLoaderTorch(dataset, batch_size=1, shuffle=False, num_workers=self.dataloader_num_workers)
        else:
            # For ensemble, we'll create dataloaders per fold in _predict_ensemble
            dataset = None
            dataloader = None
            input_dim = None

        # Get lit_model_creator with preprocessed df (needed for loss calculation)
        lit_model_creator = get_lit_model_creator(
            model=components["model"],
            task=components["task"],
            n_bins=self.n_bins,
            feature=components["features"],
            df=df,
            regularization=(components["reg"] == "REG"),
        )

        # Run predictions
        predictions_df = self._run_predictions(
            model_storage=model_storage,
            dataloader=dataloader,
            lit_model_creator=lit_model_creator,
            input_dim=input_dim,
            task=components["task"],
            features=components["features"],
            model=components["model"],
        )

        # Calculate metrics
        task_type = self._detect_task_type(predictions_df)
        metrics = self._calculate_metrics(predictions_df, task_type)

        # Save slide-level predictions to CSV
        predictions_file = self._save_predictions(predictions_df, experiment_name)

        # Build result dictionary
        result: dict[str, Any] = {
            "experiment_id": experiment_name,
            "task": components["task"],
            "features": components["features"],
            "model": components["model"],
            "reg": components["reg"],
            "stra": components["stra"],
            "predictions_file": f"predictions/{experiment_name}_predictions.csv",
            **metrics,
        }

        logger.info(f"Completed evaluation for {experiment_name}: {metrics}")
        logger.info(f"Saved predictions to: {predictions_file}")

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
        self, task: str, features: str, model: str, model_storage: ModelStorage
    ) -> CellMILDataset | PatchMILDataset:
        """
        Create external validation dataset with pre-fitted transforms.

        Args:
            task: Task name
            features: Feature type
            model: Model name
            model_storage: Model storage to load transforms from

        Returns:
            MILDataset for external validation
        """

        # Load metadata
        df = self._load_metadata_df()
        df = preprocess_df(df, task)

        # Get extractors
        extractors = get_extractors_from_name(features)

        # Load fitted transforms from model storage
        # Use final model transforms if available, otherwise use fold_0
        if model_storage.has_final_model():
            transforms, label_transforms = model_storage.load_final_transforms()
            logger.info("Loaded transforms from final model")
        else:
            transforms, label_transforms = model_storage.load_fold_transforms(0)
            logger.info("Loaded transforms from fold 0")

        # Create dataset with fitted transforms
        dataset = MILDataset(
            root=self.config.root_dir,
            label=task if task not in ["OS", "PFS"] else ("duration", "event"),
            folder=self.config.dataset_dir,
            data=df,
            extractor=extractors,
            segmentation_model=ModelType.cellvit,
            graph_creator=GraphCreatorType.delaunay_radius,
            cell_type=True if model == "HEAD4TYPE" else False,
            transforms=transforms,  # Apply pre-fitted transforms
            label_transforms=label_transforms,
        )

        return dataset

    def _create_dataset_for_fold(
        self, task: str, features: str, model: str, model_storage: ModelStorage, fold_idx: int
    ) -> CellMILDataset | PatchMILDataset:
        """
        Create external validation dataset with a specific fold's transforms.

        Args:
            task: Task name
            features: Feature type
            model: Model name
            model_storage: Model storage to load transforms from
            fold_idx: Fold index to load transforms from

        Returns:
            MILDataset for external validation with fold-specific transforms
        """

        # Load metadata
        df = self._load_metadata_df()
        df = preprocess_df(df, task)

        # Get extractors
        extractors = get_extractors_from_name(features)

        # Load fitted transforms from specific fold
        transforms, label_transforms = model_storage.load_fold_transforms(fold_idx)
        logger.info(f"Loaded transforms from fold {fold_idx}")

        # Create dataset with fitted transforms
        dataset = MILDataset(
            root=self.config.root_dir,
            label=task if task not in ["OS", "PFS"] else ("duration", "event"),
            folder=self.config.dataset_dir,
            data=df,
            extractor=extractors,
            segmentation_model=ModelType.cellvit,
            graph_creator=GraphCreatorType.delaunay_radius,
            cell_type=True if model == "HEAD4TYPE" else False,
            transforms=transforms,  # Apply pre-fitted transforms from this fold
            label_transforms=label_transforms,
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
        input_dim: int | None,
        task: str,
        features: str,
        model: str,
    ) -> pd.DataFrame:
        """
        Run predictions using either final model or ensemble.

        Args:
            model_storage: Model storage object
            dataloader: Data loader (None for ensemble mode)
            lit_model_creator: Function to create lightning module
            input_dim: Input dimension for model (None for ensemble mode)
            task: Task name (for ensemble mode)
            features: Feature type (for ensemble mode)
            model: Model name (for ensemble mode)

        Returns:
            DataFrame with predictions and labels
        """
        if self.config.final_model == FinalModel.final:
            return self._predict_final(model_storage, dataloader, lit_model_creator, input_dim)
        else:
            return self._predict_ensemble(
                model_storage,
                lit_model_creator,
                self.config.aggregation_method,
                task,
                features,
                model,
            )

    def _predict_final(
        self,
        model_storage: ModelStorage,
        dataloader: Any,
        lit_model_creator: Any,
        input_dim: int,
    ) -> pd.DataFrame:
        """Generate predictions using final model."""
        checkpoint_path = model_storage.load_final_checkpoint()
        
        # Set map_location based on CUDA availability and use_gpu flag
        map_location = None if torch.cuda.is_available() and self.use_gpu else torch.device('cpu')
        
        model = lit_model_creator(input_dim).load_from_checkpoint(
            checkpoint_path, map_location=map_location
        )
        model.eval()

        trainer = Trainer(
            accelerator="auto" if self.use_gpu else "cpu",
            devices=1,
            logger=False,
            enable_progress_bar=False,
            enable_checkpointing=False,
        )

        return self._get_predictions_from_trainer(trainer, model, dataloader)

    def _predict_ensemble(
        self,
        model_storage: ModelStorage,
        lit_model_creator: Any,
        method: AggregationMethod,
        task: str,
        features: str,
        model: str,
    ) -> pd.DataFrame:
        """Generate ensemble predictions from all folds."""
        fold_indices = model_storage.list_folds()
        # Set map_location based on CUDA availability and use_gpu flag
        map_location = None if torch.cuda.is_available() and self.use_gpu else torch.device('cpu')

        if self.parallelize_ensemble_folds and len(fold_indices) > 1:
            # Parallel processing of folds
            logger.info(f"Processing {len(fold_indices)} folds in parallel for ensemble...")
            all_fold_predictions = self._process_folds_parallel(
                fold_indices, model_storage, lit_model_creator, 
                task, features, model, map_location
            )
        else:
            # Sequential processing of folds (original behavior)
            logger.info(f"Processing {len(fold_indices)} folds sequentially for ensemble...")
            all_fold_predictions = self._process_folds_sequential(
                fold_indices, model_storage, lit_model_creator,
                task, features, model, map_location
            )

        return self._aggregate_fold_predictions(all_fold_predictions, method)

    def _process_folds_sequential(
        self,
        fold_indices: list[int],
        model_storage: ModelStorage,
        lit_model_creator: Any,
        task: str,
        features: str,
        model: str,
        map_location: Any,
    ) -> list[pd.DataFrame]:
        """Process folds sequentially (original implementation)."""
        trainer = Trainer(
            accelerator="auto" if self.use_gpu else "cpu",
            devices=1,
            logger=False,
            enable_progress_bar=False,
            enable_checkpointing=False,
        )
        
        all_fold_predictions: list[pd.DataFrame] = []
        
        for fold_idx in fold_indices:
            logger.info(f"Processing fold {fold_idx} for ensemble...")
            fold_preds = self._process_single_fold(
                fold_idx, model_storage, lit_model_creator, trainer,
                task, features, model, map_location, self.dataloader_num_workers
            )
            all_fold_predictions.append(fold_preds)
        
        return all_fold_predictions

    def _process_folds_parallel(
        self,
        fold_indices: list[int],
        model_storage: ModelStorage,
        lit_model_creator: Any,
        task: str,
        features: str,
        model: str,
        map_location: Any,
    ) -> list[pd.DataFrame]:
        """
        Process folds in parallel using ThreadPoolExecutor.
        
        Note: Uses ThreadPoolExecutor (not ProcessPoolExecutor) so instance methods
        are accessible without needing to be static. Threads share memory space.
        """
        all_fold_predictions: list[pd.DataFrame] = []
        
        with ThreadPoolExecutor(max_workers=self.ensemble_max_workers) as executor:
            # Submit all folds - pass all arguments explicitly to avoid closure issues
            future_to_fold = {
                executor.submit(
                    self._process_single_fold_with_trainer_creation,
                    fold_idx,
                    model_storage,
                    lit_model_creator,
                    task,
                    features,
                    model,
                    map_location,
                    self.ensemble_dataloader_workers,
                    self.use_gpu
                ): fold_idx
                for fold_idx in fold_indices
            }
            
            for future in as_completed(future_to_fold):
                fold_idx = future_to_fold[future]
                try:
                    result = future.result(timeout=600)  # 10 min timeout per fold
                    if result is not None:
                        all_fold_predictions.append(result)
                        logger.info(f"Completed fold {fold_idx}")
                    else:
                        logger.error(f"Fold {fold_idx} returned None")
                except TimeoutError:
                    logger.error(f"Fold {fold_idx} timed out after 600 seconds")
                except Exception as e:
                    logger.error(f"Unexpected error for fold {fold_idx}: {e}\n{traceback.format_exc()}")
        
        # Ensure we have predictions from all folds
        if len(all_fold_predictions) != len(fold_indices):
            raise RuntimeError(
                f"Only {len(all_fold_predictions)}/{len(fold_indices)} folds completed successfully"
            )
        
        return all_fold_predictions

    def _process_single_fold_with_trainer_creation(
        self,
        fold_idx: int,
        model_storage: ModelStorage,
        lit_model_creator: Any,
        task: str,
        features: str,
        model: str,
        map_location: Any,
        num_workers: int,
        use_gpu: bool,
    ) -> pd.DataFrame | None:
        """
        Process a single fold with its own Trainer instance (for parallel execution).
        
        This method is called by ThreadPoolExecutor and includes error handling.
        Each thread creates its own Trainer to avoid conflicts.
        """
        try:
            # Create trainer per thread to avoid conflicts
            trainer = Trainer(
                accelerator="auto" if use_gpu else "cpu",
                devices=1,
                logger=False,
                enable_progress_bar=False,
                enable_checkpointing=False,
            )
            
            return self._process_single_fold(
                fold_idx, model_storage, lit_model_creator, trainer,
                task, features, model, map_location, num_workers
            )
        except Exception as e:
            logger.error(f"Failed to process fold {fold_idx}: {e}\n{traceback.format_exc()}")
            return None

    def _process_single_fold(
        self,
        fold_idx: int,
        model_storage: ModelStorage,
        lit_model_creator: Any,
        trainer: Trainer,
        task: str,
        features: str,
        model: str,
        map_location: Any,
        num_workers: int,
    ) -> pd.DataFrame:
        """Process a single fold - shared by both sequential and parallel processing."""
        # Create dataset with this fold's transforms
        dataset = self._create_dataset_for_fold(
            task=task,
            features=features,
            model=model,
            model_storage=model_storage,
            fold_idx=fold_idx,
        )
        
        # Get input dimension from this fold's transformed data
        sample_data = dataset[0]
        if isinstance(dataset, (CellGNNMILDataset, PatchGNNMILDataset)):
            input_dim = sample_data.x.shape[1]  # type: ignore
        else:
            input_dim = sample_data[0].shape[1]
        
        # Create dataloader for this fold
        if isinstance(dataset, (CellGNNMILDataset, PatchGNNMILDataset)):
            dataloader = DataLoaderPyG(dataset, batch_size=1, shuffle=False, num_workers=num_workers)
        else:
            dataloader = DataLoaderTorch(dataset, batch_size=1, shuffle=False, num_workers=num_workers)
        
        # Load this fold's checkpoint
        checkpoint_path = model_storage.load_fold_checkpoint(fold_idx)
        fold_model = lit_model_creator(input_dim).load_from_checkpoint(
            checkpoint_path, map_location=map_location
        )
        fold_model.eval()

        fold_preds = self._get_predictions_from_trainer(trainer, fold_model, dataloader)
        
        del dataloader
        del dataset
        del fold_model
        
        return fold_preds
    
    def _get_predictions_from_trainer(
        self,
        trainer: Trainer,
        model: Any,
        dataloader: Any,
    ) -> pd.DataFrame:
        """Extract predictions using Lightning Trainer."""
        # Check if survival model
        is_surv = is_survival_model(model)
        
        if is_surv:
            # For survival models, use trainer.predict which returns logits
            y_pred = cast(list[Any], trainer.predict(model, dataloader))
            
            # Extract true labels based on dataloader type
            if isinstance(dataloader, DataLoaderPyG):
                y_true = [batch.y for batch in dataloader]
            else:
                y_true = [batch[-1] for batch in dataloader]
            
            return self._process_survival_predictions(y_pred, y_true)
        else:
            # For classification, we need to manually get logits/probs
            return self._get_classification_predictions(model, dataloader)

    def _get_classification_predictions(
        self, model: Any, dataloader: Any
    ) -> pd.DataFrame:
        """Get classification predictions with logits by manually iterating."""
        model.eval()
        
        all_y_true = []
        all_logits = []
        
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(dataloader, DataLoaderPyG):
                    # For graph dataloaders
                    x = batch
                    y_true = batch.y
                    # Ensure batch size is 1 for MIL
                    output = model(x.x.squeeze(0))
                else:
                    
                    # For standard dataloaders
                    # Handle different batch formats (some models include cell types, etc.)
                    if len(batch) == 2:
                        x, y_true = batch
                    elif len(batch) == 3:
                        # Likely (x, cell_types, y_true) for HEAD4TYPE
                        x, cell_types, y_true = batch
                    else:
                        raise ValueError(f"Unexpected batch format with {len(batch)} elements")
                    
                    # Ensure MIL batch size is 1
                    assert x.size(0) == 1, "Batch size must be 1 for MIL"
                    x = x.squeeze(0)  # [n_instances, feat_dim]
                    
                    # Pass appropriate inputs based on batch content
                    if len(batch) == 3:
                        # For HEAD4TYPE, pass cell types as well
                        cell_types = cell_types.squeeze(0)  # [n_instances]
                        output = model(x, cell_types)
                    else:
                        output = model(x)
                
                # Handle models that return (logits, output_dict) or just logits
                if isinstance(output, tuple):
                    logits = output[0]
                else:
                    logits = output
                
                all_y_true.append(y_true.cpu())
                all_logits.append(logits.cpu())
        
        # Flatten all predictions
        y_true_flat = torch.cat([y.flatten() for y in all_y_true]).numpy()
        logits_flat = torch.cat(all_logits, dim=0)  # [N, n_classes]
        
        # Get class predictions from logits
        y_pred_flat = logits_flat.argmax(dim=1).numpy()
        
        # Compute probabilities for storing in DataFrame (useful for analysis/ensemble)
        probs_flat = torch.softmax(logits_flat, dim=1).numpy()
        
        # Build DataFrame with probabilities for each class
        result_dict = {
            "y_true": y_true_flat,
            "y_pred": y_pred_flat,
        }
        
        # Store both logits and probabilities
        # Logits for metric calculation (same as training)
        # Probabilities for human readability and ensemble aggregation
        for i in range(logits_flat.shape[1]):
            result_dict[f"logit_class_{i}"] = logits_flat[:, i].numpy()
            result_dict[f"prob_class_{i}"] = probs_flat[:, i]
        
        return pd.DataFrame(result_dict)

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
            # For classification: check method first, then use available data
            # If majority voting is requested, always use it regardless of available data
            if method == AggregationMethod.majority:
                # Use majority voting
                all_preds = np.stack([df["y_pred"].values for df in fold_predictions])  # type: ignore
                y_pred = stats.mode(all_preds, axis=0, keepdims=False)[0]  # type: ignore

                return pd.DataFrame(
                    {
                        "y_true": fold_predictions[0]["y_true"].values,  # type: ignore
                        "y_pred": y_pred,
                    }
                )
            
            # For other methods, aggregate probabilities (better than logits due to scale differences)
            prob_columns = [
                col for col in fold_predictions[0].columns if col.startswith("prob_class_")
            ]
            
            if prob_columns:
                # Aggregate probabilities (preferred - handles different model scales)
                n_classes = len(prob_columns)
                all_probs = []
                
                for df in fold_predictions:
                    probs = np.stack([df[col].values for col in prob_columns], axis=1)  # [n_samples, n_classes]
                    all_probs.append(probs)
                
                stacked_probs = np.stack(all_probs)  # [n_folds, n_samples, n_classes]
                
                if method == AggregationMethod.mean:
                    aggregated_probs = np.mean(stacked_probs, axis=0)
                elif method == AggregationMethod.median:
                    aggregated_probs = np.median(stacked_probs, axis=0)
                else:
                    raise ValueError(f"Unsupported aggregation method: {method}")
                
                # Get predictions from aggregated probabilities
                y_pred = np.argmax(aggregated_probs, axis=1)
                
                # Compute logits from aggregated probabilities (for consistency)
                aggregated_logits = np.log(aggregated_probs + 1e-12)
                
                # Build result with both probabilities and logits
                result_dict = {
                    "y_true": fold_predictions[0]["y_true"].values,  # type: ignore
                    "y_pred": y_pred,
                }
                
                for i in range(n_classes):
                    result_dict[f"prob_class_{i}"] = aggregated_probs[:, i]
                    result_dict[f"logit_class_{i}"] = aggregated_logits[:, i]
                
                return pd.DataFrame(result_dict)
            else:
                # Fallback to majority voting if no logits/probabilities
                all_preds = np.stack([df["y_pred"].values for df in fold_predictions])  # type: ignore
                y_pred = stats.mode(all_preds, axis=0, keepdims=False)[0]  # type: ignore

                return pd.DataFrame(
                    {
                        "y_true": fold_predictions[0]["y_true"].values,  # type: ignore
                        "y_pred": y_pred,
                    }
                )

    def _save_predictions(
        self, predictions_df: pd.DataFrame, experiment_name: str
    ) -> Path:
        """
        Save slide-level predictions to a CSV file.

        Args:
            predictions_df: DataFrame containing predictions and labels for each slide
            experiment_name: Name of the experiment (used as filename)

        Returns:
            Path to the saved predictions file
        """
        predictions_file = self.predictions_dir / f"{experiment_name}_predictions.csv"
        
        # Save predictions with slide names as index
        predictions_df.to_csv(predictions_file, index=True)
        
        logger.info(
            f"Saved {len(predictions_df)} slide predictions to {predictions_file}"
        )
        
        return predictions_file

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
            # Infer n_classes from the data
            n_classes = self._infer_n_classes(predictions_df)
            return self._calculate_classification_metrics(predictions_df, n_classes)

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

    def _infer_n_classes(self, predictions_df: pd.DataFrame) -> int:
        """Infer number of classes from predictions DataFrame."""
        # Check if logit columns exist (preferred)
        logit_columns = [
            col for col in predictions_df.columns if col.startswith("logit_class_")
        ]
        if logit_columns:
            return len(logit_columns)
        
        # Check if probability columns exist
        prob_columns = [
            col for col in predictions_df.columns if col.startswith("prob_class_")
        ]
        if prob_columns:
            return len(prob_columns)
        
        # Otherwise infer from y_true and y_pred
        y_true = predictions_df["y_true"].values
        y_pred = predictions_df["y_pred"].values
        return int(max(y_true.max(), y_pred.max()) + 1)

    def _calculate_classification_metrics(
        self, predictions_df: pd.DataFrame, n_classes: int
    ) -> dict[str, float]:
        """Calculate classification metrics using torchmetrics (same as training)."""
        y_true = torch.tensor(predictions_df["y_true"].values, dtype=torch.long)
        
        # Get logits (same format as passed to metrics during training)
        logit_columns = [
            col for col in predictions_df.columns if col.startswith("logit_class_")
        ]
        
        if logit_columns and len(logit_columns) == n_classes:
            # Stack logits into tensor
            logit_tensors = []
            for i in range(n_classes):
                col_name = f"logit_class_{i}"
                logit_tensors.append(torch.tensor(predictions_df[col_name].values, dtype=torch.float32))
            
            logits = torch.stack(logit_tensors, dim=1)  # [N, n_classes]
        else:
            # Fallback: if no logits stored, use predictions
            y_pred = torch.tensor(predictions_df["y_pred"].values, dtype=torch.long)
            # Create one-hot encoded "logits" from predictions
            logits = torch.nn.functional.one_hot(y_pred, num_classes=n_classes).float() * 10.0

        # Initialize metrics using torchmetrics (same as in LitGeneral)
        metrics_collection = torchmetrics.MetricCollection({
            "accuracy": torchmetrics.Accuracy(
                task="multiclass", num_classes=n_classes, average="macro"
            ),
            "f1": torchmetrics.F1Score(
                task="multiclass", num_classes=n_classes, average="macro"
            ),
            "precision": torchmetrics.Precision(
                task="multiclass", num_classes=n_classes, average="macro"
            ),
            "recall": torchmetrics.Recall(
                task="multiclass", num_classes=n_classes, average="macro"
            ),
        })

        # Pass logits to metrics (EXACTLY as training: self.train_metrics(logits, y))
        metrics_collection.update(logits, y_true)
        computed_metrics = metrics_collection.compute()

        metrics = {
            "accuracy": float(computed_metrics["accuracy"].item()),
            "f1": float(computed_metrics["f1"].item()),
            "recall": float(computed_metrics["recall"].item()),
            "precision": float(computed_metrics["precision"].item()),
        }

        # Calculate AUROC using logits (torchmetrics will apply softmax internally)
        try:
            auroc_metric = torchmetrics.AUROC(
                task="multiclass", num_classes=n_classes, average="macro"
            )
            # Pass logits, torchmetrics will handle softmax internally (same as training)
            auroc_metric.update(logits, y_true)
            auc = auroc_metric.compute()
            metrics["auroc"] = float(auc.item())
        except Exception as e:
            logger.warning(f"Failed to calculate AUROC: {e}")

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

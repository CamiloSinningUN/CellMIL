import lightning as Pl
import numpy as np
import pandas as pd
import time
import wandb
import torch
from pathlib import Path
from typing import Callable, Union, Any, cast
from cellmil.datamodels.datasets.cell_mil_dataset import CellMILDataset
from sklearn.model_selection import StratifiedKFold  # type: ignore
from cellmil.datamodels.datasets.cell_gnn_mil_dataset import (
    CellGNNMILDataset,
    SubsetCellGNNMILDataset,
)
from cellmil.datamodels.datasets.patch_gnn_mil_dataset import (
    PatchGNNMILDataset,
    SubsetPatchGNNMILDataset,
)
from cellmil.datamodels.datasets.patch_mil_dataset import PatchMILDataset
from cellmil.datamodels.transforms import (
    Transform,
    TransformPipeline,
    LabelTransform,
    LabelTransformPipeline,
)
from cellmil.datamodels.model import ModelStorage, FoldMetadata, ExperimentMetadata
from cellmil.utils import logger
from cellmil.utils.train.evals.utils import (
    compute_slide_cell_counts,
    is_survival_model,
    Report,
)
from lightning import Trainer, seed_everything
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader as DataLoaderTorch
from torch_geometric.loader import DataLoader as DataLoaderPyG  # type: ignore


class KFoldCrossValidation:
    def __init__(
        self,
        k: int = 5,
        random_state: int = 42,
    ):
        self.k = k
        self.random_state = random_state

        seed_everything(self.random_state)

        # Initialize stratified k-fold
        self.skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)

    def evaluate(
        self,
        name: str,
        lit_model_creator: Callable[[int], Pl.LightningModule],
        dataset: Union[
            CellMILDataset, CellGNNMILDataset, PatchGNNMILDataset, PatchMILDataset
        ],
        output_dir: Union[str, Path],
        wandb_project: str,
        transforms: Union[Transform, TransformPipeline, None] = None,
        label_transforms: Union[LabelTransform, LabelTransformPipeline, None] = None,
        balance_cell_counts: bool = False,
        cell_balance_bins: int = 5,
        **kwargs: Any,
    ) -> ModelStorage:
        """
        Perform k-fold cross-validation with comprehensive result storage.

        Args:
            name: Experiment name
            lit_model_creator: Function that creates a Lightning module given fold index
            dataset: Dataset for cross-validation
            output_dir: Directory to store all results
            transforms: Optional feature transforms
            label_transforms: Optional label transforms
            balance_cell_counts: Whether to balance by cell counts
            cell_balance_bins: Number of bins for cell count balancing
            max_epochs: Maximum training epochs per fold
            **kwargs: Additional arguments

        Returns:
            ModelStorage object with all results
        """
        # Initialize model storage
        storage = ModelStorage(output_dir, name)

        # Get stratification targets
        indices = np.arange(len(dataset))
        targets = self._get_targets(
            dataset=dataset,
            balance_cell_counts=balance_cell_counts,
            cell_balance_bins=cell_balance_bins,
        )

        # Store fold reports
        fold_reports: list[dict[str, Any]] = []

        # Login to wandb
        wandb.login()

        logger.info(f"Starting {self.k}-fold cross-validation for '{name}'...")

        for fold_idx, (train_idx, val_idx) in enumerate(
            self.skf.split(indices, targets) # type: ignore
        ):  
            logger.info(f"Training fold {fold_idx + 1}/{self.k}")
            logger.info(
                f"Train indices: {len(train_idx)}, Test indices: {len(val_idx)}"
            )

            # Create dataloaders for this fold
            train_loader, val_loader, train_dataset, _ = self.get_train_val_dataloaders(
                dataset=dataset,
                train_indices=train_idx.tolist(),
                val_indices=val_idx.tolist(),
                transforms=transforms,
                label_transforms=label_transforms,
            )

            # Create fresh model instance for this fold
            if isinstance(
                dataset,
                (
                    CellGNNMILDataset,
                    SubsetCellGNNMILDataset,
                    PatchGNNMILDataset,
                    SubsetPatchGNNMILDataset,
                ),
            ):
                sample_data = dataset[0]
                model = lit_model_creator(sample_data.x.shape[1])  # type: ignore
            else:
                sample_data = dataset[0]
                model = lit_model_creator(sample_data[0].shape[1])

            # Detect if this is a survival model
            is_surv = is_survival_model(model)

            # Setup trainer with checkpoint callback
            if is_surv:
                monitor_metric = "val/c_index"
                mode = "max"
            else:
                monitor_metric = "val/f1"
                mode = "max"

            checkpoint_callback = ModelCheckpoint(
                monitor=monitor_metric,
                mode=mode,
                save_top_k=1,
                dirpath=f"./temp_checkpoints/{name}/fold_{fold_idx}",
                filename="best",
            )

            early_stopping = EarlyStopping(
                monitor=monitor_metric,
                patience=cast(int, kwargs.get("early_stopping_patience", 10)),
                mode=mode,
                verbose=True,
            )

            wandb_logger = WandbLogger(
                project=wandb_project,
                name=f"FOLD_{fold_idx + 1}_{name}_{time.strftime('%Y-%m-%d_%H-%M-%S')}",
                tags=["fold", f"fold-{fold_idx + 1}"],
            )

            trainer = Trainer(
                max_epochs=cast(int, kwargs.get("max_epochs", 100)),
                accelerator="gpu",
                devices=[0],
                log_every_n_steps=1,
                logger=wandb_logger,
                callbacks=[checkpoint_callback, early_stopping],
                enable_progress_bar=False,
            )

            # Train the model
            trainer.fit(model, train_loader, val_loader)

            # Get evaluation report for this fold
            report_gen = Report(trainer, model)
            fold_report = report_gen.generate(val_loader)
            wandb.log(fold_report)
            fold_reports.append(fold_report)

            # Get predictions for validation set
            predictions = self._get_predictions(trainer, model, val_loader, is_surv)
            
            best_epoch = trainer.current_epoch if trainer.current_epoch == kwargs.get("max_epochs", 100) else trainer.current_epoch - early_stopping.patience

            best_metric_value = checkpoint_callback.best_model_score 
            if best_metric_value is None:
                best_metric_value = fold_report.get(monitor_metric.replace("val_", ""), 0.0)

            # Create fold metadata
            fold_meta = FoldMetadata(
                fold_idx=fold_idx,
                train_size=len(train_idx),
                val_size=len(val_idx),
                best_epoch=int(best_epoch)
                if isinstance(best_epoch, (int, float))
                else trainer.current_epoch,
                best_metric_value=float(best_metric_value),
                metric_name=monitor_metric,
                is_survival=is_surv,
                metrics=fold_report,
            )

            # Get the fitted transforms from the datasets
            fitted_transforms = getattr(train_dataset, "transforms", None)
            fitted_label_transforms = getattr(train_dataset, "label_transforms", None)

            # Save fold results
            storage.save_fold_results(
                fold_idx=fold_idx,
                checkpoint_path=checkpoint_callback.best_model_path,  # type: ignore
                train_indices=train_idx.tolist(),
                val_indices=val_idx.tolist(),
                predictions=predictions,
                metadata=fold_meta,
                transforms=fitted_transforms,
                label_transforms=fitted_label_transforms,
            )

            logger.info(
                f"Fold {fold_idx + 1} completed with {monitor_metric}: {best_metric_value:.4f}"
            )
            wandb.finish()

        # Aggregate reports across folds
        logger.info("Aggregating results across folds...")
        aggregated_report = self._aggregate_reports(fold_reports)

        # Determine best fold
        if "c_index" in aggregated_report:
            metric_key = "c_index"
        else:
            metric_key = "macro avg"

        best_fold_idx = int(
            np.argmax(
                [
                    r.get(metric_key, r.get("macro avg", {}).get("f1-score", 0))
                    for r in fold_reports
                ]
            )
        )

        # Calculate average best epoch
        avg_best_epoch = storage.get_average_best_epoch()

        # Extract dataset configuration using the dataset's get_config method
        dataset_config = dataset.get_config()
        
        # Extract model name from first fold model
        model = lit_model_creator(0)
        model_config: dict[str, Any] = {
            "model_class": model.__class__.__name__,
        }

        # Save experiment metadata
        exp_meta = ExperimentMetadata(
            name=name,
            k_folds=self.k,
            random_state=self.random_state,
            balance_cell_counts=balance_cell_counts,
            cell_balance_bins=cell_balance_bins,
            is_survival="c_index" in aggregated_report,
            aggregated_metrics=aggregated_report,
            best_fold_idx=best_fold_idx,
            avg_best_epoch=avg_best_epoch,
            dataset_config=dataset_config,
            model_config=model_config,
        )

        storage.save_experiment_metadata(exp_meta)

        # Train final model with average epochs
        logger.info(f"Training final model with {avg_best_epoch:.1f} average epochs...")
        
        final_checkpoint = self._train_final_model(
            name=name,
            lit_model_creator=lit_model_creator,
            dataset=dataset,
            transforms=transforms,
            label_transforms=label_transforms,
            target_epochs=int(np.round(avg_best_epoch)),
            project_name=wandb_project,
        )

        # Save final model
        storage.save_final_model(
            checkpoint_path=final_checkpoint,
            avg_epochs=avg_best_epoch,
            final_metrics=aggregated_report,
        )

        logger.info("K-fold cross-validation completed successfully!")
        logger.info(f"Results saved to: {storage.output_dir}")

        return storage

    def get_train_val_dataloaders(
        self,
        dataset: Union[
            CellMILDataset, CellGNNMILDataset, PatchGNNMILDataset, PatchMILDataset
        ],
        train_indices: list[int],
        val_indices: list[int],
        transforms: Union[Transform, TransformPipeline, None] = None,
        label_transforms: Union[LabelTransform, LabelTransformPipeline, None] = None,
    ) -> tuple[
        Union[DataLoaderTorch[Any], DataLoaderPyG],
        Union[DataLoaderTorch[Any], DataLoaderPyG],
        Union[
            CellMILDataset,
            CellGNNMILDataset,
            PatchGNNMILDataset,
            PatchMILDataset,
            SubsetCellGNNMILDataset,
            SubsetPatchGNNMILDataset,
        ],
        Union[
            CellMILDataset,
            CellGNNMILDataset,
            PatchGNNMILDataset,
            PatchMILDataset,
            SubsetCellGNNMILDataset,
            SubsetPatchGNNMILDataset,
        ],
    ]:
        """
        Create train and validation dataloaders with proper transforms.

        Args:
            dataset: Full dataset
            train_indices: Indices for training set
            val_indices: Indices for validation set
            transforms: Optional feature transforms
            label_transforms: Optional label transforms

        Returns:
            Tuple of (train_dataloader, val_dataloader, train_dataset, val_dataset)
        """
        train_dataset, val_dataset = dataset.create_train_val_datasets(
            train_indices=train_indices,
            val_indices=val_indices,
            transforms=transforms,
            label_transforms=label_transforms,
        )

        if isinstance(dataset, (CellGNNMILDataset, PatchGNNMILDataset)):
            train_loader = DataLoaderPyG(
                train_dataset,  # type: ignore
                batch_size=1,
                shuffle=True,
                num_workers=0,
            )
            val_loader = DataLoaderPyG(
                val_dataset,  # type: ignore
                batch_size=1,
                shuffle=False,
                num_workers=0,
            )
        else:
            train_loader = DataLoaderTorch(
                train_dataset,  # type: ignore
                batch_size=1,
                shuffle=True,
                num_workers=0,
            )
            val_loader = DataLoaderTorch(
                val_dataset,  # type: ignore
                batch_size=1,
                shuffle=False,
                num_workers=0,
            )

        return train_loader, val_loader, train_dataset, val_dataset

    def _get_targets(
        self,
        dataset: Union[
            CellMILDataset, CellGNNMILDataset, PatchGNNMILDataset, PatchMILDataset
        ],
        balance_cell_counts: bool = False,
        cell_balance_bins: int = 5,
    ) -> np.ndarray[Any, Any]:
        slides = (
            np.array(dataset.slides)
            if hasattr(dataset, "slides")
            else np.arange(len(dataset))
        )

        # Handle both classification labels (single value) and survival labels (tuple)
        raw_labels = [dataset.labels[slide] for slide in slides]

        # Check if this is survival data (labels are tuples of (duration, event))
        is_survival_data = isinstance(raw_labels[0], tuple)

        if is_survival_data:
            # For survival analysis, stratify by event status only
            y = cast(
                np.ndarray[Any, Any],
                np.array([label[1] for label in raw_labels]),  # type: ignore
            )  # Extract event indicators
            logger.info(
                f"Detected survival data - stratifying by event status: {np.bincount(y.astype(int))}"
            )
        else:
            # For classification, use labels directly
            y = np.array(raw_labels)
            logger.info(
                f"Detected classification data - stratifying by class labels: {np.bincount(y.astype(int))}"
            )

        cell_counts = compute_slide_cell_counts(dataset, slides)
        split_targets = y

        if balance_cell_counts:
            try:
                unique_counts = np.unique(cell_counts)
                if unique_counts.size <= 1:
                    logger.warning(
                        "Cell-count balancing disabled: insufficient variability in counts."
                    )
                else:
                    quantile_bins = min(cell_balance_bins, unique_counts.size)
                    bin_array = np.asarray(
                        pd.qcut(  # type: ignore
                            cell_counts,
                            q=quantile_bins,
                            labels=False,
                            duplicates="drop",
                        ),
                        dtype=float,
                    )
                    if np.isnan(bin_array).any():
                        raise ValueError("Quantile binning produced NaNs")

                    cell_bins = bin_array.astype(int)

                    # For both classification and survival, combine event/label with cell bins
                    combined_labels = pd.Series(
                        [f"{label}_{bin_idx}" for label, bin_idx in zip(y, cell_bins)]
                    )
                    combined_codes, _ = pd.factorize(combined_labels, sort=True)  # type: ignore
                    valid_mask = combined_codes >= 0
                    if not valid_mask.any():
                        raise ValueError("Factorization produced all invalid codes")

                    class_counts = np.bincount(combined_codes[valid_mask])
                    if class_counts.size == 0 or class_counts.min() < self.k:
                        logger.warning(
                            "Cell-count balancing fallback: at least one label/bin combo has fewer samples than folds."
                        )
                    else:
                        split_targets = combined_codes
                        if is_survival_data:
                            logger.info(
                                "Enabled joint stratification on event status and %d cell-count bins",
                                len(np.unique(cell_bins)),
                            )
                        else:
                            logger.info(
                                "Enabled joint stratification on labels and %d cell-count bins",
                                len(np.unique(cell_bins)),
                            )
            except Exception as exc:
                logger.warning(
                    f"Failed to apply cell-count-balanced stratification: {exc}. Using event/label-only stratification."
                )

        return split_targets

    def _get_predictions(
        self,
        trainer: Trainer,
        model: Pl.LightningModule,
        dataloader: Union[DataLoaderTorch[Any], DataLoaderPyG],
        is_survival: bool,
    ) -> dict[str, Any]:
        """
        Get predictions and true labels from the model.

        Args:
            trainer: Lightning trainer
            model: Lightning model
            dataloader: DataLoader
            is_survival: Whether this is a survival model

        Returns:
            Dictionary with prediction data
        """
        y_pred = trainer.predict(model, dataloader)

        # Extract true labels
        if isinstance(dataloader, DataLoaderPyG):
            y_true = [data.y for data in dataloader]
        else:
            y_true = [batch[-1] for batch in dataloader]

        if is_survival:
            # For survival: extract durations and events
            durations: list[float] = []
            events: list[float] = []

            for label in y_true:
                if isinstance(label, (tuple, list)) and len(label) == 2: # type: ignore
                    label = cast(tuple[float, float], label)
                    durations.append(float(label[0]))
                    events.append(float(label[1]))
                elif torch.is_tensor(label) and label.numel() == 2:
                    durations.append(float(label[0]))
                    events.append(float(label[1]))
                else:
                    logger.warning(f"Unexpected label format: {label}")
                    durations.append(0.0)
                    events.append(0.0)

            # Get risk scores (predictions)
            if y_pred is not None and isinstance(y_pred[0], torch.Tensor):
                # For survival, predictions are risk from logits
                risk_scores: list[float] = []
                
                for pred in y_pred:  # type: ignore
                    logits = pred.cpu() if pred.is_cuda else pred  # type: ignore
                    # Apply sigmoid to get hazards
                    hazards = torch.sigmoid(logits) # type: ignore
                    # Calculate cumulative survival probability
                    survival = torch.cumprod(1 - hazards, dim=0)
                    # Risk is negative sum of survival probabilities
                    risk = -float(torch.sum(survival))
                    
                    risk_scores.append(risk)
                
                y_pred_flat = risk_scores
            else:
                y_pred_flat = [0.0] * len(durations)

            return {
                "y_true_duration": np.array(durations),
                "y_true_event": np.array(events),
                "y_pred_risk": np.array(y_pred_flat),
            }
        else:
            # For classification: extract predicted and true classes
            if y_pred is not None and isinstance(y_pred[0], torch.Tensor):
                y_pred_flat = [pred.cpu().numpy().flatten()[0] for pred in y_pred]  # type: ignore
            else:
                y_pred_flat = [  # type: ignore
                    pred.flatten()[0] if hasattr(pred, "flatten") else pred  # type: ignore
                    for pred in y_pred  # type: ignore
                ]

            if y_true and isinstance(y_true[0], torch.Tensor):
                y_true_flat = [true.cpu().numpy().flatten()[0] for true in y_true]
            else:
                y_true_flat = [
                    true.flatten()[0] if hasattr(true, "flatten") else true
                    for true in y_true
                ]

            return {
                "y_true": np.array(y_true_flat),
                "y_pred": np.array(y_pred_flat), # type: ignore
            }

    def _aggregate_reports(
        self,
        fold_reports: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Aggregate reports across all folds.

        Args:
            fold_reports: List of fold reports

        Returns:
            Aggregated report dictionary
        """
        if not fold_reports:
            return {}

        # Check if this is survival or classification
        if "c_index" in fold_reports[0]:
            # Survival analysis aggregation
            return {
                "c_index": np.mean(
                    [report.get("c_index", 0) for report in fold_reports]
                ),
                "n_samples": np.sum(
                    [report.get("n_samples", 0) for report in fold_reports]
                ),
                "n_events": np.sum(
                    [report.get("n_events", 0) for report in fold_reports]
                ),
                "c_index_std": np.std(
                    [report.get("c_index", 0) for report in fold_reports]
                )
            }
        else:
            # Classification aggregation
            class_labels: set[str] = set()
            for report in fold_reports:
                for key in report.keys():
                    if key not in ["accuracy", "macro avg", "weighted avg"]:
                        class_labels.add(key)

            aggregated_report: dict[str, Any] = {}

            # Aggregate per-class metrics
            for label in class_labels:
                aggregated_report[label] = {
                    "precision": np.mean(
                        [
                            report.get(label, {}).get("precision", 0)
                            for report in fold_reports
                        ]
                    ),
                    "recall": np.mean(
                        [
                            report.get(label, {}).get("recall", 0)
                            for report in fold_reports
                        ]
                    ),
                    "f1-score": np.mean(
                        [
                            report.get(label, {}).get("f1-score", 0)
                            for report in fold_reports
                        ]
                    ),
                    "support": np.sum(
                        [
                            report.get(label, {}).get("support", 0)
                            for report in fold_reports
                        ]
                    ),
                }

            # Aggregate overall metrics
            aggregated_report["accuracy"] = np.mean(
                [report.get("accuracy", 0) for report in fold_reports]
            )

            aggregated_report["macro avg"] = {
                "precision": np.mean(
                    [
                        report.get("macro avg", {}).get("precision", 0)
                        for report in fold_reports
                    ]
                ),
                "recall": np.mean(
                    [
                        report.get("macro avg", {}).get("recall", 0)
                        for report in fold_reports
                    ]
                ),
                "f1-score": np.mean(
                    [
                        report.get("macro avg", {}).get("f1-score", 0)
                        for report in fold_reports
                    ]
                ),
                "support": np.sum(
                    [
                        report.get("macro avg", {}).get("support", 0)
                        for report in fold_reports
                    ]
                ),
            }

            aggregated_report["weighted avg"] = {
                "precision": np.mean(
                    [
                        report.get("weighted avg", {}).get("precision", 0)
                        for report in fold_reports
                    ]
                ),
                "recall": np.mean(
                    [
                        report.get("weighted avg", {}).get("recall", 0)
                        for report in fold_reports
                    ]
                ),
                "f1-score": np.mean(
                    [
                        report.get("weighted avg", {}).get("f1-score", 0)
                        for report in fold_reports
                    ]
                ),
                "support": np.sum(
                    [
                        report.get("weighted avg", {}).get("support", 0)
                        for report in fold_reports
                    ]
                ),
            }

            return aggregated_report

    def _train_final_model(
        self,
        name: str,
        lit_model_creator: Callable[[int], Pl.LightningModule],
        dataset: Union[
            CellMILDataset, CellGNNMILDataset, PatchGNNMILDataset, PatchMILDataset
        ],
        transforms: Union[Transform, TransformPipeline, None],
        label_transforms: Union[LabelTransform, LabelTransformPipeline, None],
        target_epochs: int,
        project_name: str,
    ) -> str:
        """
        Train final model on full dataset with target number of epochs.

        Args:
            name: Experiment name
            lit_model_creator: Function to create model
            dataset: Full dataset
            transforms: Feature transforms
            label_transforms: Label transforms
            target_epochs: Number of epochs to train
            project_name: Wandb project name

        Returns:
            Path to final checkpoint
        """
        logger.info("Training final model on full dataset...")

        # Use all indices for training the final model
        all_indices = list(range(len(dataset)))
        
        # Use dataset's create_train_val_datasets to properly fit transforms on full data
        # Pass all indices as both train and val (we'll only use train)
        if transforms is not None or label_transforms is not None:
            train_dataset, _ = dataset.create_train_val_datasets(
                train_indices=all_indices,
                val_indices=[],  # Empty validation set
                transforms=transforms,
                label_transforms=label_transforms,
            )
        else:
            # No transforms, use dataset directly
            train_dataset = dataset

        # Create dataloader
        if isinstance(dataset, (CellGNNMILDataset, PatchGNNMILDataset)):
            dataloader = DataLoaderPyG(
                train_dataset,  # type: ignore
                batch_size=1,
                shuffle=True,
                num_workers=0,
            )
            sample_data = dataset[0]
            model = lit_model_creator(sample_data.x.shape[1])  # type: ignore
        else:
            dataloader = DataLoaderTorch(
                train_dataset,  # type: ignore
                batch_size=1,
                shuffle=True,
                num_workers=0,
            )
            sample_data = dataset[0]
            model = lit_model_creator(sample_data[0].shape[1])

        # Setup checkpoint callback
        is_surv = is_survival_model(model)
        monitor_metric = "train/c_index" if is_surv else "train/f1"
        mode = "max"

        checkpoint_callback = ModelCheckpoint(
            monitor=monitor_metric,
            mode=mode,
            save_top_k=1,
            dirpath=f"./temp_checkpoints/{name}/final_model",
            filename="final",
        )

        wandb_logger = WandbLogger(
            project=project_name,
            name=f"FINAL_{name}_{time.strftime('%Y-%m-%d_%H-%M-%S')}",
            tags=["final"],
        )

        trainer = Trainer(
            max_epochs=target_epochs,
            accelerator="gpu",
            devices=[0],
            log_every_n_steps=1,
            logger=wandb_logger,
            callbacks=[checkpoint_callback],
            enable_progress_bar=False,
        )

        # Train without validation (full dataset)
        trainer.fit(model, dataloader)

        wandb.finish()

        return str(checkpoint_callback.best_model_path) # type: ignore



    # def _plot_sample_features(
    #     train_dataset: Union[
    #         CellMILDataset,
    #         CellGNNMILDataset,
    #         PatchGNNMILDataset,
    #         PatchMILDataset,
    #         SubsetCellGNNMILDataset,
    #         SubsetPatchGNNMILDataset,
    #     ],
    #     test_dataset: Union[
    #         CellMILDataset,
    #         CellGNNMILDataset,
    #         PatchGNNMILDataset,
    #         PatchMILDataset,
    #         SubsetCellGNNMILDataset,
    #         SubsetPatchGNNMILDataset,
    #     ],
    #     name: str,
    # ) -> None:
    #     """
    #     Plot random samples from train and test datasets with their labels and feature heatmaps.

    #     Args:
    #         train_dataset: Training dataset for the current fold
    #         test_dataset: Test dataset for the current fold
    #         fold_idx: Current fold index
    #         name: Experiment name for saving plots
    #     """
    #     try:
    #         # Create figure with subplots
    #         fig, axes = plt.subplots(2, 2, figsize=(15, 12))  # type: ignore
    #         fig.suptitle(f"Feature Visualization - {name}", fontsize=16)  # type: ignore

    #         # Get random samples from train and test
    #         train_idx = random.randint(0, len(train_dataset) - 1)
    #         test_idx = random.randint(0, len(test_dataset) - 1)

    #         train_sample = cast(tuple[torch.Tensor, int], train_dataset[train_idx])
    #         test_sample = cast(tuple[torch.Tensor, int], test_dataset[test_idx])

    #         # Extract features and labels based on dataset type
    #         if isinstance(
    #             train_dataset,
    #             (
    #                 CellGNNMILDataset,
    #                 PatchGNNMILDataset,
    #                 SubsetCellGNNMILDataset,
    #                 SubsetPatchGNNMILDataset,
    #             ),
    #         ):
    #             # For graph datasets
    #             train_features = cast(np.ndarray[Any, Any], train_sample.x.cpu().numpy())  # type: ignore
    #             train_label = cast(np.ndarray[Any, Any], train_sample.y.cpu().numpy())  # type: ignore
    #             test_features = cast(np.ndarray[Any, Any], test_sample.x.cpu().numpy())  # type: ignore
    #             test_label = cast(np.ndarray[Any, Any], test_sample.y.cpu().numpy())  # type: ignore
    #         else:
    #             # For regular MIL datasets
    #             train_features, train_label = train_sample
    #             test_features, test_label = test_sample

    #             train_features = cast(np.ndarray[Any, Any], train_features.cpu().numpy())  # type: ignore
    #             train_label = cast(int, train_label)  # type: ignore
    #             test_features = cast(np.ndarray[Any, Any], test_features.cpu().numpy())  # type: ignore
    #             test_label = cast(int, test_label)  # type: ignore

    #         # Ensure features are 2D for heatmap
    #         if hasattr(train_features, "shape") and len(train_features.shape) == 1:
    #             raise Exception("Train features have invalid shape.")
    #         if hasattr(test_features, "shape") and len(test_features.shape) == 1:
    #             raise Exception("Test features have invalid shape.")

    #         # Plot train sample
    #         sns.heatmap(  # type: ignore
    #             train_features[:50],  # Limit to first 50 instances for readability
    #             ax=axes[0, 0],
    #             cmap="viridis",
    #             cbar=True,
    #             xticklabels=False,
    #             yticklabels=False,
    #         )
    #         axes[0, 0].set_title(
    #             f"Train Sample {train_idx}\nLabel: {train_label}\nShape: {train_features.shape}"
    #         )

    #         # Plot feature distribution for train sample
    #         feature_means = np.mean(train_features, axis=0)
    #         axes[0, 1].hist(feature_means, bins=30, alpha=0.7, color="blue")
    #         axes[0, 1].set_title(
    #             f"Train Sample - Feature Mean Distribution\nMean: {np.mean(feature_means):.3f}"
    #         )
    #         axes[0, 1].set_xlabel("Feature Value")
    #         axes[0, 1].set_ylabel("Frequency")

    #         # Plot test sample
    #         sns.heatmap(  # type: ignore
    #             test_features[:50],  # Limit to first 50 instances for readability
    #             ax=axes[1, 0],
    #             cmap="viridis",
    #             cbar=True,
    #             xticklabels=False,
    #             yticklabels=False,
    #         )
    #         axes[1, 0].set_title(
    #             f"Test Sample {test_idx}\nLabel: {test_label}\nShape: {test_features.shape}"
    #         )

    #         # Plot feature distribution for test sample
    #         feature_means = np.mean(test_features, axis=0)
    #         axes[1, 1].hist(feature_means, bins=30, alpha=0.7, color="red")
    #         axes[1, 1].set_title(
    #             f"Test Sample - Feature Mean Distribution\nMean: {np.mean(feature_means):.3f}"
    #         )
    #         axes[1, 1].set_xlabel("Feature Value")
    #         axes[1, 1].set_ylabel("Frequency")

    #         plt.tight_layout()

    #         # Save plot
    #         plot_dir = Path(f"./plots/{name}")
    #         plot_dir.mkdir(parents=True, exist_ok=True)
    #         plot_path = plot_dir / f"{name}_sample_visualization.png"
    #         plt.savefig(plot_path, dpi=300, bbox_inches="tight")  # type: ignore

    #         plt.close(fig)

    #         logger.info(f"Sample visualization saved to: {plot_path}")

    #     except Exception as e:
    #         logger.error(f"Exception occurred: {e}\n{traceback.format_exc()}")
    #         raise RuntimeError(f"Error during sample feature plotting: {e}")
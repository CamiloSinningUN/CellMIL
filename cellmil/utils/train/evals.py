import torch
import copy
import lightning as Pl
import numpy as np
import pandas as pd
import time
import os
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
import random
import shutil
from pathlib import Path
from lightning import Trainer
from typing import cast, Any, Union, Callable
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report  # type: ignore
from sklearn.model_selection import StratifiedKFold  # type: ignore
from torch.utils.data import DataLoader as DataLoaderTorch
from torch_geometric.loader import DataLoader as DataLoaderPyG  # type: ignore
from cellmil.utils import logger
from cellmil.utils.train.metrics import ConcordanceIndex, BrierScore

from cellmil.datamodels.datasets.cell_mil_dataset import CellMILDataset
from cellmil.datamodels.datasets.cell_gnn_mil_dataset import (
    CellGNNMILDataset,
    SubsetCellGNNMILDataset,
)
from cellmil.datamodels.datasets.patch_gnn_mil_dataset import (
    PatchGNNMILDataset,
    SubsetPatchGNNMILDataset,
)
from cellmil.datamodels.datasets.patch_mil_dataset import PatchMILDataset
from cellmil.datamodels.transforms import Transform, TransformPipeline, LabelTransform, LabelTransformPipeline
import traceback

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def _is_survival_model(lit_model: Pl.LightningModule) -> bool:
    """Check if the model is a survival analysis model."""
    # Check if it's a LitSurvAttentionDeepMIL or similar survival model
    model_class_name = lit_model.__class__.__name__
    return "Surv" in model_class_name or hasattr(lit_model, "_train_logits")


def _compute_slide_cell_counts(
    dataset: Union[
        CellMILDataset, CellGNNMILDataset, PatchGNNMILDataset, PatchMILDataset
    ],
    slides: np.ndarray[Any, Any],
):
    """Estimate per-slide cell counts for balancing folds."""
    counts: list[int] = []
    missing: list[str] = []

    feature_cache = getattr(dataset, "features", None)
    index_lookup = cast(dict[int, str] | None, getattr(dataset, "cell_indices", None))

    slide_sequence = list(slides)

    for idx, slide_id in enumerate(slide_sequence):
        count: int | None = None

        if isinstance(feature_cache, dict) and slide_id in feature_cache:
            tensor = cast(torch.Tensor, feature_cache[slide_id])
            count = int(tensor.shape[0])

        if (
            count is None
            and isinstance(index_lookup, dict)
            and slide_id in index_lookup
        ):
            count = int(len(index_lookup[slide_id]))

        if count is None:
            try:
                sample = dataset[idx]
            except Exception:
                sample = None

            if sample is not None:
                if isinstance(sample, tuple) and len(sample) > 0:
                    feature_tensor = sample[0]
                    feature_shape = getattr(feature_tensor, "shape", None)
                    if feature_shape is not None and len(feature_shape) > 0:
                        count = int(feature_shape[0])
                else:
                    data_obj = sample
                    num_nodes = getattr(data_obj, "num_nodes", None)
                    if isinstance(num_nodes, int):
                        count = num_nodes
                    else:
                        node_features = getattr(data_obj, "x", None)
                        node_shape = getattr(node_features, "shape", None)
                        if node_shape is not None and len(node_shape) > 0:
                            count = int(node_shape[0])

        if count is None:
            counts.append(0)
            missing.append(str(slide_id))
        else:
            counts.append(count)

    if missing:
        preview = ", ".join(missing[:5])
        suffix = "..." if len(missing) > 5 else ""
        logger.warning(
            f"Unable to determine cell counts for {len(missing)} slides; assigned 0 for balancing: {preview}{suffix}"
        )

    return np.asarray(counts, dtype=np.int64)


def get_report(
    trainer: Trainer,
    lit_model: Pl.LightningModule,
    dataloader: Union[DataLoaderTorch[Any], DataLoaderPyG],
) -> dict[str, Any]:
    """
    Get evaluation report for classification or survival analysis models.

    Returns a dict with appropriate metrics:
    - Classification: precision, recall, f1-score, accuracy
    - Survival: c_index, brier_score
    """
    try:
        # Try to find ModelCheckpoint callback and load best checkpoint
        for callback in trainer.callbacks:  # type: ignore
            if isinstance(callback, ModelCheckpoint) and callback.best_model_path:  # type: ignore
                logger.info(f"Loading best model from: {callback.best_model_path}")  # type: ignore
                # Load the state dict from the best checkpoint into the current model
                # Set weights_only=False to handle optimizer state and other objects
                checkpoint = torch.load(
                    callback.best_model_path,  # type: ignore
                    map_location=lit_model.device,  # type: ignore
                    weights_only=False,
                )
                lit_model.load_state_dict(checkpoint["state_dict"])
                lit_model.eval()
                break
    except Exception as e:
        logger.warning(
            f"Could not load best checkpoint: {e}. Using current model state."
        )

    # Check if this is a survival model
    is_survival = _is_survival_model(lit_model)

    if is_survival:
        # For survival models, compute C-index and Brier score
        return _get_survival_report(trainer, lit_model, dataloader)
    else:
        # For classification models, use the existing logic
        return _get_classification_report(trainer, lit_model, dataloader)

def _get_classification_report(
    trainer: Trainer,
    lit_model: Pl.LightningModule,
    dataloader: Union[DataLoaderTorch[Any], DataLoaderPyG],
) -> dict[str, Any]:
    """Get classification report with precision, recall, f1-score."""
    y_pred = trainer.predict(lit_model, dataloader)

    if isinstance(dataloader, DataLoaderPyG):
        y_true = [data.y for data in dataloader]
    else:
        # Handle both 2-element (x, y) and 3-element (x, cell_types, y) batches
        # Extract the last element which should always be the label
        y_true = [batch[-1] for batch in dataloader]

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
            true.flatten()[0] if hasattr(true, "flatten") else true for true in y_true
        ]

    report = cast(
        dict[str, Any],
        classification_report(y_true_flat, y_pred_flat, output_dict=True),
    )  # type: ignore
    return report


def _get_survival_report(
    trainer: Trainer,
    lit_model: Pl.LightningModule,
    dataloader: Union[DataLoaderTorch[Any], DataLoaderPyG],
) -> dict[str, Any]:
    """Get survival analysis report with C-index and Brier score."""

    def _extract_survival_tensors(target: Any) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Normalize different target formats to (duration, event) tensors."""

        def _to_tensor(value: Any) -> torch.Tensor:
            tensor = torch.as_tensor(value)
            if tensor.ndim == 0:
                tensor = tensor.unsqueeze(0)
            return tensor

        if isinstance(target, dict):
            key_duration = next((k for k in target if k.lower() in {"duration", "durations", "time"}), None) # type: ignore
            key_event = next((k for k in target if k.lower() in {"event", "events", "status"}), None) # type: ignore
            if key_duration is not None and key_event is not None:
                return _to_tensor(target[key_duration]), _to_tensor(target[key_event])

        if isinstance(target, (list, tuple)) and len(target) == 2: # type: ignore
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

    def _append_target(target: Any, source: str) -> None:
        parsed = _extract_survival_tensors(target)
        if parsed is None:
            logger.warning("Unexpected %s format for survival data", source)
            return
        dur_tensor, evt_tensor = parsed
        durations_list.append(dur_tensor)
        events_list.append(evt_tensor)

    # Get predictions (logits from discrete-time hazard model)
    y_pred = trainer.predict(lit_model, dataloader)

    # Extract true survival data (durations, events) from dataloader
    durations_list: list[torch.Tensor] = []
    events_list: list[torch.Tensor] = []

    if isinstance(dataloader, DataLoaderPyG):
        for data in dataloader:
            _append_target(getattr(data, "y", None), "PyG batch.y")
    else:
        for batch in dataloader:
            # batch is (x, y) where y is (duration, event)
            y = batch[-1]
            _append_target(y, "batch label")

    # Check if we have any data
    if not durations_list or not events_list:
        raise ValueError("No survival data found in dataloader")

    # Convert predictions to tensor
    if y_pred is not None and len(y_pred) > 0:
        if isinstance(y_pred[0], torch.Tensor):
            # Predictions are logits with shape [1, num_bins] per sample
            logits = torch.cat([pred.cpu() for pred in y_pred], dim=0)  # type: ignore [batch_size, num_bins]
        else:
            logger.error("Unexpected prediction format")
            logits = torch.zeros((len(durations_list), 1))  # type: ignore
    else:
        logger.error("No predictions returned")
        logits = torch.zeros((len(durations_list), 1))  # type: ignore

    # Convert durations and events to tensors
    durations = torch.cat([d.cpu().flatten() for d in durations_list])  # type: ignore
    events = torch.cat([e.cpu().flatten() for e in events_list])  # type: ignore

    # Initialize metrics
    c_index_metric = ConcordanceIndex()
    brier_score_metric = BrierScore()

    # Update metrics with logits (not hazards)
    c_index_metric.update(logits, (durations, events))
    brier_score_metric.update(logits, (durations, events))

    report: dict[str, float] = {
        "c_index": float(c_index_metric.compute()),
        "brier_score": float(brier_score_metric.compute()),
        "n_samples": len(durations),
        "n_events": int(events.sum()),
    }

    return report


def k_fold_train_eval(
    name: str,
    lit_model_creator: Callable[[int], Pl.LightningModule],
    dataset: Union[
        CellMILDataset, CellGNNMILDataset, PatchGNNMILDataset, PatchMILDataset
    ],
    transforms: Union[Transform, TransformPipeline, None] = None,
    label_transforms: Union[LabelTransform, LabelTransformPipeline, None] = None,
    k: int = 5,
    random_state: int = 42,
    debug: bool = False,
    split_save_dir: Union[str, Path, None] = None,
    balance_cell_counts: bool = False,
    cell_balance_bins: int = 5,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Perform k-fold cross-validation training and evaluation.

    Args:
        name (str): Name for the experiment run.
        lit_model (Pl.LightningModule): The Lightning model to be trained and evaluated.
        dataset: The dataset to be used for k-fold cross-validation.
        transforms (Union[Transform, TransformPipeline, None]): Optional transforms to apply with proper fitting on training data only.
        label_transforms: Optional label transform (e.g., TimeDiscretizerTransform) to fit on training labels and apply to both train/val.
        k (int): Number of folds for cross-validation.
        random_state (int): Random seed for reproducibility.
        debug (bool): If True, enables debug mode with more verbose logging.
        split_save_dir: Optional base directory to store per-fold train/val splits.
        balance_cell_counts (bool): If True, jointly stratify by label and cell-count quantile bins.
        cell_balance_bins (int): Number of quantile bins to use when balancing cell counts.

    Returns:
        dict[str, Any]: A dictionary containing the aggregated classification report.
    """
    # Extract labels for stratified split

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
            np.ndarray[Any,Any], 
            np.array([label[1] for label in raw_labels]) # type: ignore
        )  # Extract event indicators
        logger.info(f"Detected survival data - stratifying by event status: {np.bincount(y.astype(int))}")
    else:
        # For classification, use labels directly
        y = np.array(raw_labels)
        logger.info(f"Detected classification data - stratifying by class labels: {np.bincount(y.astype(int))}")
    
    indices = np.arange(len(dataset))

    cell_counts = _compute_slide_cell_counts(dataset, slides)
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
                if class_counts.size == 0 or class_counts.min() < k:
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

    # Initialize stratified k-fold
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)

    # Store reports from each fold
    fold_reports: list[dict[str, Any]] = []

    split_output_base = None
    if split_save_dir is not None:
        split_output_base = Path(split_save_dir) / name
        split_output_base.mkdir(parents=True, exist_ok=True)

    # Track best fold information
    best_fold_idx = -1
    best_fold_f1 = -1.0
    best_fold_checkpoint_path: Union[str, None] = None
    best_fold_transforms: Union[Transform, TransformPipeline, None] = None
    best_fold_label_transforms: Union[LabelTransform, LabelTransformPipeline, None] = None

    logger.info(f"Starting {k}-fold cross-validation...")
    # Model training
    wandb.login()

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(indices, split_targets)):  # type: ignore
        logger.info(f"Training fold {fold_idx + 1}/{k}")
        logger.info(f"Train indices: {len(train_idx)}, Test indices: {len(test_idx)}")

        if cell_counts.size:
            train_cells = int(cell_counts[train_idx].sum())
            val_cells = int(cell_counts[test_idx].sum())
            logger.info(
                f"Fold {fold_idx + 1}: total cells -> train {train_cells:,}, val {val_cells:,}"
            )

        # Use the new helper functions for proper transform fitting
        train_dataset, test_dataset = dataset.create_train_val_datasets(  # type: ignore
            train_indices=train_idx.tolist(),
            val_indices=test_idx.tolist(),
            transforms=transforms,
            label_transforms=label_transforms,
        )

        # Create dataloaders based on dataset type
        if isinstance(dataset, (CellGNNMILDataset, PatchGNNMILDataset)):
            train_loader = DataLoaderPyG(
                train_dataset,  # type: ignore
                batch_size=1,
                shuffle=True,
                num_workers=8,
            )
            test_loader = DataLoaderPyG(
                test_dataset,  # type: ignore
                batch_size=1,
                shuffle=False,
                num_workers=8,
            )
        else:
            train_loader = DataLoaderTorch(
                train_dataset,  # type: ignore
                batch_size=1,
                shuffle=True,
                num_workers=8,
            )
            test_loader = DataLoaderTorch(
                test_dataset,  # type: ignore
                batch_size=1,
                shuffle=False,
                num_workers=8,
            )

        if split_output_base is not None:
            fold_dir = split_output_base / f"fold_{fold_idx + 1}"
            fold_dir.mkdir(parents=True, exist_ok=True)

            split_records: list[dict[str, Any]] = []
            for idx in train_idx:
                slide_id = slides[idx] if len(slides) > idx else int(idx)
                split_records.append({"slide_id": slide_id, "split": "train"})
            for idx in test_idx:
                slide_id = slides[idx] if len(slides) > idx else int(idx)
                split_records.append({"slide_id": slide_id, "split": "val"})

            split_df = pd.DataFrame(split_records)
            split_path = fold_dir / "split.csv"
            split_df.to_csv(split_path, index=False)

            if hasattr(train_dataset, "transforms"):  # type: ignore
                fold_transforms = getattr(train_dataset, "transforms", None)  # type: ignore
                if fold_transforms is not None:
                    if isinstance(fold_transforms, TransformPipeline):
                        transforms_path = fold_dir / "transforms"
                        fold_transforms.save(transforms_path)
                    else:
                        transforms_path = fold_dir / "transform.json"
                        fold_transforms.save(transforms_path)
            
            if hasattr(train_dataset, "label_transforms"):  # type: ignore
                fold_label_transforms = getattr(train_dataset, "label_transforms", None)  # type: ignore
                if fold_label_transforms is not None:
                    if isinstance(fold_label_transforms, LabelTransformPipeline):
                        label_transforms_path = fold_dir / "label_transforms"
                        fold_label_transforms.save(label_transforms_path)
                    else:
                        label_transforms_path = fold_dir / "label_transform.json"
                        fold_label_transforms.save(label_transforms_path)

        # Plot sample features if in debug mode
        if debug:
            logger.info("Debug mode enabled - plotting sample features")
            plot_sample_features(
                train_dataset, test_dataset, f"Fold {fold_idx + 1} - {name}"
            )
            input("Press Enter to continue...")

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
            model = lit_model_creator(train_dataset[0].x.shape[-1])  # type: ignore
        else:
            model = lit_model_creator(train_dataset[0][0].shape[-1])  # type: ignore

        # Detect if this is a survival model
        is_survival = _is_survival_model(model)

        # Setup trainer with checkpoint callback
        if is_survival:
            # For survival, monitor c_index
            early_stopping = EarlyStopping(
                monitor="val/total_loss",
                patience=cast(int, kwargs.get("early_stopping_patience", 10)),
                mode="min",
            )
            checkpoint_callback = ModelCheckpoint(
                dirpath=Path(f"./checkpoints/{name}/fold_{fold_idx + 1}"),
                filename="best_model",
                monitor="val/c_index",
                mode="max",
                save_top_k=1,
            )
        else:
            # For classification, monitor f1
            early_stopping = EarlyStopping(
                monitor="val/total_loss",
                patience=cast(int, kwargs.get("early_stopping_patience", 10)),
                mode="min",
            )
            checkpoint_callback = ModelCheckpoint(
                dirpath=Path(f"./checkpoints/{name}/fold_{fold_idx + 1}"),
                filename="best_model",
                monitor="val/f1",
                mode="max",
                save_top_k=1,
            )

        if debug:
            project_name = "(TEST) CELLMIL (K-Fold)"
        else:
            project_name = (
                "CELLMIL (K-Fold)"
                if not balance_cell_counts
                else "CELLMIL (K-Fold Cell Stratified)"
            )

        wandb_logger = WandbLogger(
            project=project_name,
            name=f"FOLD_{fold_idx + 1}_{name}_{time.strftime('%Y-%m-%d_%H-%M-%S')}",
            tags=["fold", f"fold-{fold_idx + 1}"],
        )

        trainer = Trainer(
            max_epochs=100,
            accelerator="gpu",
            devices=[0],
            log_every_n_steps=1,
            logger=wandb_logger,
            callbacks=[checkpoint_callback, early_stopping],
            enable_progress_bar=False,
        )

        # Train the model
        trainer.fit(model, train_loader, test_loader)

        # Get evaluation report for this fold
        fold_report = get_report(trainer, model, test_loader)
        wandb.log(fold_report)

        fold_reports.append(fold_report)

        # Track if this is the best fold
        if is_survival:
            current_metric = fold_report.get("c_index", 0)
            metric_name = "C-index"
        else:
            current_metric = fold_report.get("macro avg", {}).get("f1-score", 0)
            metric_name = "F1"

        if current_metric > best_fold_f1:
            best_fold_f1 = current_metric
            best_fold_idx = fold_idx
            best_fold_checkpoint_path = cast(Path, checkpoint_callback.best_model_path)  # type: ignore

            if hasattr(train_dataset, "transforms") and isinstance( # type: ignore
                train_dataset,
                (CellMILDataset, CellGNNMILDataset, SubsetCellGNNMILDataset),
            ):
                # Save the transforms for this fold
                transforms_obj = getattr(train_dataset, "transforms", None)
                best_fold_transforms = copy.deepcopy(transforms_obj) if transforms_obj is not None else None
            
            if hasattr(train_dataset, "label_transforms"): # type: ignore
                label_transforms_obj = getattr(train_dataset, "label_transforms", None)
                best_fold_label_transforms = (
                    copy.deepcopy(label_transforms_obj) if label_transforms_obj is not None else None
                )

        logger.info(
            f"Fold {fold_idx + 1} completed with {metric_name}: {current_metric:.4f}"
        )
        wandb.finish()

    # Aggregate reports across folds
    logger.info("Aggregating results across folds...")

    # Check if this is survival or classification
    if fold_reports and "c_index" in fold_reports[0]:
        # Survival analysis aggregation
        aggregated_report: dict[str, Any] = {
            "c_index": np.mean([report.get("c_index", 0) for report in fold_reports]),
            "brier_score": np.mean(
                [report.get("brier_score", 0) for report in fold_reports]
            ),
            "n_samples": np.sum(
                [report.get("n_samples", 0) for report in fold_reports]
            ),
            "n_events": np.sum([report.get("n_events", 0) for report in fold_reports]),
            "c_index_std": np.std(
                [report.get("c_index", 0) for report in fold_reports]
            ),
            "brier_score_std": np.std(
                [report.get("brier_score", 0) for report in fold_reports]
            ),
        }
        logger.info(
            f"Aggregated C-index: {aggregated_report['c_index']:.4f} ± {aggregated_report['c_index_std']:.4f}"
        )
        logger.info(
            f"Aggregated Brier Score: {aggregated_report['brier_score']:.4f} ± {aggregated_report['brier_score_std']:.4f}"
        )
    else:
        # Classification aggregation (existing logic)
        # Get unique class labels
        class_labels: set[str] = set()
        for report in fold_reports:
            class_labels.update(
                [
                    k
                    for k in report.keys()
                    if k not in ["accuracy", "macro avg", "weighted avg"]
                ]
            )  # type: ignore

        # Initialize aggregated report structure
        aggregated_report = {}

        # Aggregate per-class metrics
        for label in class_labels:
            if label.isdigit() or label.replace(".", "").isdigit():
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

        # Aggregate macro average
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

        # Aggregate weighted average
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

    if debug:
        project_name = "(TEST) CELLMIL (K-Fold)"
    else:
        project_name = (
            "CELLMIL (K-Fold)"
            if not balance_cell_counts
            else "CELLMIL (K-Fold Cell Stratified)"
        )

    # Initialize wandb run for aggregated results
    wandb.init(
        project=project_name,
        name=f"FINAL_{name}_{time.strftime('%Y-%m-%d_%H-%M-%S')}",
        tags=["final"],
        reinit=True,
    )
    # Log aggregated results to wandb
    wandb.log(aggregated_report)
    # Finish the wandb run
    wandb.finish()

    # Save only the best fold's checkpoint and transforms
    if "c_index" in aggregated_report:
        logger.info(f"Best fold: {best_fold_idx + 1} with C-index: {best_fold_f1:.4f}")
    else:
        logger.info(f"Best fold: {best_fold_idx + 1} with F1: {best_fold_f1:.4f}")

    # Create final checkpoint directory
    final_checkpoint_dir = Path(f"./checkpoints/{name}/best_fold")
    final_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Copy best fold checkpoint to final location
    if best_fold_checkpoint_path and Path(str(best_fold_checkpoint_path)).exists():
        final_checkpoint_path = final_checkpoint_dir / "best_model.ckpt"
        shutil.copy2(str(best_fold_checkpoint_path), final_checkpoint_path)
        logger.info(f"Saved best fold checkpoint to: {final_checkpoint_path}")

        # Save the transforms
        if best_fold_transforms is not None:
            if isinstance(best_fold_transforms, TransformPipeline):
                transforms_path = final_checkpoint_dir / "transforms"
                best_fold_transforms.save(transforms_path)
                logger.info(f"Saved transforms to: {transforms_path}")
            else:
                transforms_path = final_checkpoint_dir / "transform.json"
                best_fold_transforms.save(transforms_path)
                logger.info(f"Saved transform to: {transforms_path}")
        
        # Save the label transforms
        if best_fold_label_transforms is not None:
            if isinstance(best_fold_label_transforms, LabelTransformPipeline):
                label_transforms_path = final_checkpoint_dir / "label_transforms"
                best_fold_label_transforms.save(label_transforms_path)
                logger.info(f"Saved label transforms to: {label_transforms_path}")
            else:
                label_transforms_path = final_checkpoint_dir / "label_transform.json"
                best_fold_label_transforms.save(label_transforms_path)
                logger.info(f"Saved label transform to: {label_transforms_path}")

        # Clean up other fold checkpoints
        for fold_idx in range(k):
            if fold_idx != best_fold_idx:
                fold_checkpoint_dir = Path(f"./checkpoints/{name}/fold_{fold_idx + 1}")
                if fold_checkpoint_dir.exists():
                    shutil.rmtree(fold_checkpoint_dir)
                    logger.info(f"Removed checkpoint for fold {fold_idx + 1}")
    else:
        logger.warning("Best fold checkpoint not found!")

    logger.info("K-fold cross-validation completed successfully")
    return aggregated_report


def plot_sample_features(
    train_dataset: Union[
        CellMILDataset,
        CellGNNMILDataset,
        PatchGNNMILDataset,
        PatchMILDataset,
        SubsetCellGNNMILDataset,
        SubsetPatchGNNMILDataset,
    ],
    test_dataset: Union[
        CellMILDataset,
        CellGNNMILDataset,
        PatchGNNMILDataset,
        PatchMILDataset,
        SubsetCellGNNMILDataset,
        SubsetPatchGNNMILDataset,
    ],
    name: str,
) -> None:
    """
    Plot random samples from train and test datasets with their labels and feature heatmaps.

    Args:
        train_dataset: Training dataset for the current fold
        test_dataset: Test dataset for the current fold
        fold_idx: Current fold index
        name: Experiment name for saving plots
    """
    try:
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))  # type: ignore
        fig.suptitle(f"Feature Visualization - {name}", fontsize=16)  # type: ignore

        # Get random samples from train and test
        train_idx = random.randint(0, len(train_dataset) - 1)
        test_idx = random.randint(0, len(test_dataset) - 1)

        train_sample = cast(tuple[torch.Tensor, int], train_dataset[train_idx])
        test_sample = cast(tuple[torch.Tensor, int], test_dataset[test_idx])

        # Extract features and labels based on dataset type
        if isinstance(
            train_dataset,
            (
                CellGNNMILDataset,
                PatchGNNMILDataset,
                SubsetCellGNNMILDataset,
                SubsetPatchGNNMILDataset,
            ),
        ):
            # For graph datasets
            train_features = cast(np.ndarray[Any, Any], train_sample.x.cpu().numpy())  # type: ignore
            train_label = cast(np.ndarray[Any, Any], train_sample.y.cpu().numpy())  # type: ignore
            test_features = cast(np.ndarray[Any, Any], test_sample.x.cpu().numpy())  # type: ignore
            test_label = cast(np.ndarray[Any, Any], test_sample.y.cpu().numpy())  # type: ignore
        else:
            # For regular MIL datasets
            train_features, train_label = train_sample
            test_features, test_label = test_sample

            train_features = cast(np.ndarray[Any, Any], train_features.cpu().numpy())  # type: ignore
            train_label = cast(int, train_label)  # type: ignore
            test_features = cast(np.ndarray[Any, Any], test_features.cpu().numpy())  # type: ignore
            test_label = cast(int, test_label)  # type: ignore

        # Ensure features are 2D for heatmap
        if hasattr(train_features, "shape") and len(train_features.shape) == 1:
            raise Exception("Train features have invalid shape.")
        if hasattr(test_features, "shape") and len(test_features.shape) == 1:
            raise Exception("Test features have invalid shape.")

        # Plot train sample
        sns.heatmap(  # type: ignore
            train_features[:50],  # Limit to first 50 instances for readability
            ax=axes[0, 0],
            cmap="viridis",
            cbar=True,
            xticklabels=False,
            yticklabels=False,
        )
        axes[0, 0].set_title(
            f"Train Sample {train_idx}\nLabel: {train_label}\nShape: {train_features.shape}"
        )

        # Plot feature distribution for train sample
        feature_means = np.mean(train_features, axis=0)
        axes[0, 1].hist(feature_means, bins=30, alpha=0.7, color="blue")
        axes[0, 1].set_title(
            f"Train Sample - Feature Mean Distribution\nMean: {np.mean(feature_means):.3f}"
        )
        axes[0, 1].set_xlabel("Feature Value")
        axes[0, 1].set_ylabel("Frequency")

        # Plot test sample
        sns.heatmap(  # type: ignore
            test_features[:50],  # Limit to first 50 instances for readability
            ax=axes[1, 0],
            cmap="viridis",
            cbar=True,
            xticklabels=False,
            yticklabels=False,
        )
        axes[1, 0].set_title(
            f"Test Sample {test_idx}\nLabel: {test_label}\nShape: {test_features.shape}"
        )

        # Plot feature distribution for test sample
        feature_means = np.mean(test_features, axis=0)
        axes[1, 1].hist(feature_means, bins=30, alpha=0.7, color="red")
        axes[1, 1].set_title(
            f"Test Sample - Feature Mean Distribution\nMean: {np.mean(feature_means):.3f}"
        )
        axes[1, 1].set_xlabel("Feature Value")
        axes[1, 1].set_ylabel("Frequency")

        plt.tight_layout()

        # Save plot
        plot_dir = Path(f"./plots/{name}")
        plot_dir.mkdir(parents=True, exist_ok=True)
        plot_path = plot_dir / f"{name}_sample_visualization.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")  # type: ignore

        plt.close(fig)

        logger.info(f"Sample visualization saved to: {plot_path}")

    except Exception as e:
        logger.error(f"Exception occurred: {e}\n{traceback.format_exc()}")
        raise RuntimeError(f"Error during sample feature plotting: {e}")
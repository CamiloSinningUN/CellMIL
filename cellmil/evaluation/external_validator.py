import torch
import wandb
import numpy as np
import pandas as pd
from enum import Enum
from typing import Any, Literal, cast
from dataclasses import dataclass

import lightning as Pl
from lightning import Trainer
from torch.utils.data import DataLoader as DataLoaderTorch
from torch_geometric.loader import DataLoader as DataLoaderPyG  # type: ignore
from sklearn.metrics import (  
    accuracy_score, # type: ignore
    precision_recall_fscore_support,  # type: ignore
    roc_auc_score,  # type: ignore
    classification_report,  # type: ignore
)

from cellmil.datamodels.model import ModelStorage
from cellmil.interfaces.EvaluationExternalValidatorConfig import (
    EvaluationExternalValidatorConfig,
)
from cellmil.utils import logger
from cellmil.utils.train.metrics import ConcordanceIndex
from cellmil.utils.train.evals.utils import is_survival_model


class PredictionMode(str, Enum):
    """Prediction mode for external validation."""

    final = "final"
    ensemble = "ensemble"

    @classmethod
    def values(cls):
        return [member.value for member in cls]

    def __str__(self):
        return self.value


@dataclass
class ValidationResults:
    """Results from external validation."""

    mode: str
    task_type: str  # "classification" or "survival"
    metrics: dict[str, Any]
    predictions: pd.DataFrame
    aggregated_report: dict[str, Any] | None = None


class ExternalValidator:
    """
    External validator for evaluating trained models on independent test sets.

    Supports two prediction modes:
    - final: Uses the single final model trained on averaged epochs
    - ensemble: Uses ensemble predictions from all fold models
      - Ensemble methods: mean, median, majority, weighted
      - Weighted method uses fold validation performance for weighting

    Handles both classification and survival analysis tasks.
    
    For classification:
    - Extracts class probabilities from model outputs
    - Enables weighted mean/median aggregation of probabilities
    - Calculates AUROC metrics when probabilities available
    - Falls back to majority voting when probabilities unavailable
    
    For survival:
    - Works with logits from discrete-time hazard models
    - Aggregates logits before converting to risk scores
    - Uses ConcordanceIndex metric for evaluation
    """

    def __init__(self, config: EvaluationExternalValidatorConfig):
        """
        Initialize external validator.

        Args:
            config: Configuration containing model path
        """
        self.config = config
        self.model_storage = ModelStorage.from_directory(self.config.model_path)

        # Validate that model storage has necessary data
        if not self.model_storage.experiment_metadata:
            raise ValueError("Model storage does not contain experiment metadata")

        logger.info(
            f"Loaded model: {self.model_storage.experiment_name} "
            f"with {len(self.model_storage.list_folds())} folds"
        )

    def validate(
        self,
        test_dataloader: DataLoaderTorch[Any] | DataLoaderPyG,
        lit_model_class: type[Pl.LightningModule],
        mode: PredictionMode = PredictionMode.final,
        ensemble_method: Literal["mean", "median", "majority", "weighted"] = "mean",
        wandb_project: str | None = None,
        save_predictions: bool = True,
    ) -> ValidationResults:
        """
        Perform external validation on test data.

        Args:
            test_dataloader: DataLoader containing test samples
            lit_model_class: Lightning module class to instantiate models
            mode: Prediction mode ("final" or "ensemble")
            ensemble_method: Method for ensemble aggregation ("mean", "median", "majority")
            wandb_project: Optional wandb project name for logging
            save_predictions: Whether to save predictions to disk

        Returns:
            ValidationResults containing metrics and predictions
        """
        logger.info(f"Starting external validation with mode: {mode}")

        # Initialize wandb if requested
        if wandb_project:
            wandb.login()
            wandb.init(
                project=wandb_project,
                name=f"{self.model_storage.experiment_name}_external_val_{mode}",
                config={
                    "experiment": self.model_storage.experiment_name,
                    "mode": str(mode),
                    "ensemble_method": ensemble_method if mode == PredictionMode.ensemble else None,
                },
            )

        # Generate predictions based on mode
        if mode == PredictionMode.final:
            predictions_df = self._predict_final(
                test_dataloader, lit_model_class
            )
        elif mode == PredictionMode.ensemble:
            predictions_df = self._predict_ensemble(
                test_dataloader, lit_model_class, method=ensemble_method
            )
        else:
            raise ValueError(f"Unknown prediction mode: {mode}")

        # Detect task type
        task_type = self._detect_task_type(predictions_df)
        logger.info(f"Detected task type: {task_type}")

        # Calculate metrics
        metrics, report = self._calculate_metrics(predictions_df, task_type)

        # Log to wandb if enabled
        if wandb_project:
            wandb.log(metrics)
            if report:
                wandb.log({"classification_report": wandb.Table(dataframe=pd.DataFrame(report))})

        # Save predictions to disk
        if save_predictions:
            self._save_predictions(predictions_df, metrics, mode, task_type)

        # Create results object
        results = ValidationResults(
            mode=str(mode),
            task_type=task_type,
            metrics=metrics,
            predictions=predictions_df,
            aggregated_report=report,
        )

        logger.info(f"External validation completed. Metrics: {metrics}")

        if wandb_project:
            wandb.finish()

        return results

    def _predict_final(
        self,
        dataloader: DataLoaderTorch[Any] | DataLoaderPyG,
        lit_model_class: type[Pl.LightningModule],
    ) -> pd.DataFrame:
        """
        Generate predictions using the final model.

        Args:
            dataloader: Test data loader
            lit_model_class: Lightning module class

        Returns:
            DataFrame with predictions
        """
        logger.info("Generating predictions with final model...")

        # Load final checkpoint
        checkpoint_path = self.model_storage.load_final_checkpoint()
        model = lit_model_class.load_from_checkpoint(checkpoint_path) # type: ignore
        model.eval()

        # Use trainer for predictions (handles device management automatically)
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
        dataloader: DataLoaderTorch[Any] | DataLoaderPyG,
        lit_model_class: type[Pl.LightningModule],
        method: Literal["mean", "median", "majority", "weighted"] = "mean",
    ) -> pd.DataFrame:
        """
        Generate ensemble predictions from all fold models.

        Args:
            dataloader: Test data loader
            lit_model_class: Lightning module class
            method: Ensemble method ("mean", "median", or "majority")

        Returns:
            DataFrame with ensemble predictions
        """
        logger.info(f"Generating ensemble predictions with method: {method}...")

        fold_indices = self.model_storage.list_folds()
        
        # Create trainer once
        trainer = Trainer(
            accelerator="auto",
            devices=1,
            logger=False,
            enable_progress_bar=False,
            enable_checkpointing=False,
        )

        # Collect predictions from all folds
        all_fold_predictions: list[pd.DataFrame] = []

        for fold_idx in fold_indices:
            logger.info(f"  Processing fold {fold_idx}...")
            checkpoint_path = self.model_storage.load_fold_checkpoint(fold_idx)
            model = lit_model_class.load_from_checkpoint(checkpoint_path)  # type: ignore
            model.eval()

            fold_preds = self._get_predictions_from_trainer(trainer, model, dataloader)
            all_fold_predictions.append(fold_preds)

        # Combine predictions using specified method
        return self._aggregate_fold_predictions(all_fold_predictions, method)

    def _get_predictions_from_trainer(
        self,
        trainer: Trainer,
        model: Pl.LightningModule,
        dataloader: DataLoaderTorch[Any] | DataLoaderPyG,
    ) -> pd.DataFrame:
        """
        Extract predictions using Lightning Trainer.

        Args:
            trainer: Lightning trainer
            model: Lightning module
            dataloader: Data loader

        Returns:
            DataFrame with predictions and true labels
        """
        # Get predictions using trainer
        y_pred = trainer.predict(model, dataloader)

        # Extract true labels from dataloader
        if isinstance(dataloader, DataLoaderPyG):
            y_true = [data.y for data in dataloader]
        else:
            y_true = [batch[-1] for batch in dataloader]

        # Detect if this is a survival model
        is_surv = is_survival_model(model)

        if is_surv:
            # For survival, use trainer.predict as before
            if y_pred is None:
                raise ValueError("Predictions returned by trainer.predict() should not be None")
            return self._process_survival_predictions(y_pred, y_true)
        else:
            # For classification, manually run inference to get probabilities
            return self._get_classification_predictions(model, dataloader)

    def _get_classification_predictions(
        self,
        model: Pl.LightningModule,
        dataloader: DataLoaderTorch[Any] | DataLoaderPyG,
    ) -> pd.DataFrame:
        """
        Extract classification predictions with probabilities by running manual inference.
        
        Args:
            model: Lightning module
            dataloader: Data loader
            
        Returns:
            DataFrame with predictions, probabilities, and true labels
        """
        model.eval()
        device = next(model.parameters()).device
        
        all_predictions: list[int] = []
        all_probabilities: list[np.ndarray[Any, Any]] = []
        all_labels: list[int] = []
        
        with torch.no_grad():
            for batch in dataloader:
                # Handle different data types (graph vs tensor)
                if isinstance(dataloader, DataLoaderPyG):
                    # Graph data
                    data = batch.to(device)
                    labels = data.y
                    
                    # Forward pass
                    logits, _ = model(data)
                else:
                    # Tensor data - MIL batch
                    x, y = batch
                    labels = y
                    
                    # Ensure MIL batch size is 1 and squeeze
                    assert x.size(0) == 1, "Batch size must be 1 for MIL"
                    x = x.squeeze(0).to(device)  # [n_instances, feat_dim]
                    
                    # Forward pass
                    logits, _ = model(x)
                
                # Get probabilities from softmax
                probabilities = torch.softmax(logits, dim=-1)
                predictions = torch.argmax(probabilities, dim=-1)
                
                # Convert to numpy
                probs_np = cast(np.ndarray[Any, Any], probabilities.cpu().numpy()) # type: ignore
                preds_np = cast(np.ndarray[Any, Any], predictions.cpu().numpy()) # type: ignore
                
                # Handle labels
                if isinstance(labels, torch.Tensor):
                    labels_np = cast(np.ndarray[Any, Any], labels.cpu().numpy()) # type: ignore
                    if labels_np.ndim > 0:
                        labels_np = labels_np.flatten()
                    else:
                        labels_np = np.array([labels_np])
                else:
                    labels_np = np.array([labels])
                
                # Store results
                all_predictions.extend(preds_np.flatten().tolist())
                all_labels.extend(labels_np.tolist())
                
                # Store probabilities (handle both multi-class and binary)
                if probs_np.ndim == 1:
                    all_probabilities.append(probs_np)
                else:
                    # Squeeze batch dimension if present
                    probs_squeezed = probs_np.squeeze()
                    if probs_squeezed.ndim == 1:
                        all_probabilities.append(probs_squeezed)
                    else:
                        # Should not happen with batch_size=1, but handle anyway
                        all_probabilities.extend(probs_squeezed)
        
        # Create DataFrame
        df = pd.DataFrame({
            "y_true": all_labels,
            "y_pred": all_predictions,
        })
        
        # Add probability columns
        probs_array = np.array(all_probabilities)
        if probs_array.ndim == 2:
            # Multi-class: add column for each class
            n_classes = probs_array.shape[1]
            for i in range(n_classes):
                df[f"prob_class_{i}"] = probs_array[:, i]
        else:
            # Binary or single dimension
            df["probability"] = probs_array
        
        return df

    def _process_survival_predictions(
        self, y_pred: list[Any], y_true: list[Any]
    ) -> pd.DataFrame:
        """
        Process survival predictions following k_fold approach.

        Args:
            y_pred: List of prediction tensors (logits from discrete-time hazard model)
            y_true: List of true labels (duration, event) tuples or tensors

        Returns:
            DataFrame with survival predictions
        """
        # Extract durations and events
        durations: list[float] = []
        events: list[float] = []

        for label in y_true:
            if isinstance(label, (tuple, list)) and len(label) == 2:  # type: ignore
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

        # Process predictions (logits)
        # Store raw logits for metric computation
        logits_list: list[torch.Tensor] = []
        risk_scores: list[float] = []

        if isinstance(y_pred[0], torch.Tensor):
            for pred in y_pred:  # type: ignore
                logits = pred.cpu() if pred.is_cuda else pred  # type: ignore
                logits_list.append(logits)
                
                # Calculate risk score for display (same as k_fold)
                hazards = torch.sigmoid(logits)  # type: ignore
                survival = torch.cumprod(1 - hazards, dim=0)
                risk = -float(torch.sum(survival))
                risk_scores.append(risk)
        else:
            risk_scores = [0.0] * len(durations)
            # Create zero logits as fallback
            for _ in range(len(durations)):
                logits_list.append(torch.zeros(1))

        # Create DataFrame
        df = pd.DataFrame({
            "y_true_duration": durations,
            "y_true_event": events,
            "risk_score": risk_scores,
        })

        # Store logits as a column (for ensemble averaging)
        # We'll store them as numpy arrays
        df["logits"] = [logits.cpu().numpy() for logits in logits_list] # type: ignore

        return df

    def _process_classification_predictions(
        self, y_pred: list[Any], y_true: list[Any]
    ) -> pd.DataFrame:
        """
        Process classification predictions following k_fold approach.

        Args:
            y_pred: List of predictions (class indices)
            y_true: List of true labels

        Returns:
            DataFrame with classification predictions
        """
        # Extract predicted classes
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

        df = pd.DataFrame({
            "y_true": y_true_flat,
            "y_pred": y_pred_flat,
        })

        return df

    def _aggregate_fold_predictions(
        self,
        fold_predictions: list[pd.DataFrame],
        method: Literal["mean", "median", "majority", "weighted"],
    ) -> pd.DataFrame:
        """
        Aggregate predictions from multiple folds.

        Args:
            fold_predictions: List of prediction DataFrames from each fold
            method: Aggregation method

        Returns:
            Aggregated predictions DataFrame
        """
        # Use first fold as template
        result_df = fold_predictions[0].copy()

        # Determine task type
        is_survival = "y_true_duration" in result_df.columns
        
        # Get fold weights if using weighted method
        fold_weights = None
        if method == "weighted":
            fold_weights = self._get_fold_weights(is_survival)
            logger.info(f"Fold weights: {fold_weights}")

        if is_survival:
            # Aggregate logits for survival (before converting to risk)
            if "logits" in result_df.columns:
                # Stack all logits
                all_logits = [df["logits"].tolist() for df in fold_predictions]
                
                # Aggregate logits
                aggregated_logits: list[np.ndarray[Any, Any]] = []
                for sample_idx in range(len(all_logits[0])):
                    sample_logits = [
                        fold_logits[sample_idx] for fold_logits in all_logits
                    ]
                    # Stack into array [n_folds, n_bins]
                    stacked = np.stack(sample_logits, axis=0)
                    
                    if method == "mean":
                        agg = np.mean(stacked, axis=0)
                    elif method == "median":
                        agg = np.median(stacked, axis=0)
                    elif method == "weighted" and fold_weights is not None:
                        # Weighted average using fold performance
                        agg = np.average(stacked, axis=0, weights=fold_weights)
                    else:
                        # For survival, mean is most appropriate
                        if method != "weighted":
                            logger.warning(f"Method '{method}' not ideal for survival, using mean")
                        agg = np.mean(stacked, axis=0)
                    
                    aggregated_logits.append(agg)
                
                # Convert aggregated logits back to risk scores
                risk_scores: list[float] = []
                for logits_np in aggregated_logits:
                    logits_tensor = torch.from_numpy(logits_np) # type: ignore
                    hazards = torch.sigmoid(logits_tensor)
                    survival = torch.cumprod(1 - hazards, dim=0)
                    risk = -float(torch.sum(survival))
                    risk_scores.append(risk)
                
                result_df["risk_score"] = risk_scores
                result_df["logits"] = aggregated_logits
            else:
                # Fallback: average risk scores directly (less accurate)
                risk_scores = cast(np.ndarray[Any, Any], np.stack([df["risk_score"].values for df in fold_predictions])) # type: ignore
                if method == "mean":
                    result_df["risk_score"] = np.mean(risk_scores, axis=0)
                elif method == "median":
                    result_df["risk_score"] = np.median(risk_scores, axis=0)                
                elif method == "weighted" and fold_weights is not None:
                    result_df["risk_score"] = np.average(risk_scores, axis=0, weights=fold_weights)                
                else:
                    result_df["risk_score"] = np.mean(risk_scores, axis=0)

        else:
            # Classification task
            # Check if we have probabilities
            prob_cols = [col for col in result_df.columns if col.startswith("prob_")]
            has_probs = len(prob_cols) > 0 or "probability" in result_df.columns
            
            if has_probs:
                # Aggregate probabilities, then derive predictions
                if prob_cols:
                    # Multi-class probabilities
                    for col in prob_cols:
                        probs = cast(np.ndarray[Any, Any], np.stack([df[col].values for df in fold_predictions])) # type: ignore
                        
                        if method == "mean":
                            result_df[col] = np.mean(probs, axis=0)
                        elif method == "median":
                            result_df[col] = np.median(probs, axis=0)
                        elif method == "weighted" and fold_weights is not None:
                            result_df[col] = np.average(probs, axis=0, weights=fold_weights)
                        else:  # majority
                            # For majority, still do voting on predictions below
                            pass
                    
                    # Update predictions from aggregated probabilities (unless using majority)
                    if method != "majority":
                        prob_values = result_df[prob_cols].values
                        result_df["y_pred"] = np.argmax(prob_values, axis=1)
                    else:
                        # Majority voting on predictions
                        predictions = cast(np.ndarray[Any, Any], np.stack([df["y_pred"].values for df in fold_predictions])) # type: ignore
                        result_df["y_pred"] = np.apply_along_axis(
                            lambda x: np.bincount(x.astype(int)).argmax(), axis=0, arr=predictions
                        )
                
                elif "probability" in result_df.columns:
                    # Binary classification
                    probs = cast(np.ndarray[Any, Any], np.stack([df["probability"].values for df in fold_predictions])) # type: ignore
                    
                    if method == "mean":
                        result_df["probability"] = np.mean(probs, axis=0)
                    elif method == "median":
                        result_df["probability"] = np.median(probs, axis=0)
                    elif method == "weighted" and fold_weights is not None:
                        result_df["probability"] = np.average(probs, axis=0, weights=fold_weights)
                    else:  # majority
                        # For majority, do voting on predictions
                        predictions = cast(np.ndarray[Any, Any], np.stack([df["y_pred"].values for df in fold_predictions])) # type: ignore
                        result_df["y_pred"] = np.apply_along_axis(
                            lambda x: np.bincount(x.astype(int)).argmax(), axis=0, arr=predictions
                        )
                        return result_df
                    
                    # Update predictions from aggregated probability
                    result_df["y_pred"] = (result_df["probability"] > 0.5).astype(int)
            
            else:
                # No probabilities available, use voting methods
                if method == "majority":
                    # Majority voting for predictions
                    predictions = cast(np.ndarray[Any, Any], np.stack([df["y_pred"].values for df in fold_predictions])) # type: ignore
                    result_df["y_pred"] = np.apply_along_axis(
                        lambda x: np.bincount(x.astype(int)).argmax(), axis=0, arr=predictions
                    )
                else:
                    # For mean/median, we need probabilities which we don't have
                    # Fall back to majority voting
                    logger.warning(
                        f"Method '{method}' requires probabilities but none available. Using majority voting."
                    )
                    predictions = cast(np.ndarray[Any, Any], np.stack([df["y_pred"].values for df in fold_predictions])) # type: ignore
                    result_df["y_pred"] = np.apply_along_axis(
                        lambda x: np.bincount(x.astype(int)).argmax(), axis=0, arr=predictions
                    )

        return result_df

    def _get_fold_weights(self, is_survival: bool) -> np.ndarray[Any, Any]:
        """
        Calculate weights for each fold based on their validation performance.
        
        For survival tasks, uses C-index from each fold.
        For classification tasks, uses F1 score from each fold.
        
        Args:
            is_survival: Whether this is a survival task
            
        Returns:
            Array of weights (normalized to sum to 1) for each fold
        """
        fold_indices = self.model_storage.list_folds()
        performances: list[float] = []
        
        for fold_idx in fold_indices:
            fold_metadata = self.model_storage.fold_metadata.get(fold_idx)
            
            if fold_metadata is None:
                logger.warning(f"No metadata found for fold {fold_idx}, using default weight")
                performances.append(0.5)  # Default to neutral performance
                continue
            
            # Extract performance metric
            if is_survival:
                # Use C-index for survival
                metric_value = fold_metadata.metrics.get("c_index", 0.5)
            else:
                # Use F1 score for classification (from macro avg)
                if "macro avg" in fold_metadata.metrics:
                    metric_value = fold_metadata.metrics["macro avg"].get("f1-score", 0.0)
                else:
                    # Fallback to weighted avg or any F1 metric
                    metric_value = fold_metadata.metrics.get("f1", 0.0)
            
            performances.append(float(metric_value))
        
        # Convert to numpy array
        performances_array = np.array(performances)
        
        # Handle edge case: all performances are zero
        if np.sum(performances_array) == 0:
            logger.warning("All fold performances are zero, using uniform weights")
            return np.ones(len(fold_indices)) / len(fold_indices)
        
        # Normalize to sum to 1
        weights = performances_array / np.sum(performances_array)
        
        return weights

    def _detect_task_type(self, predictions_df: pd.DataFrame) -> str:
        """
        Detect whether this is a classification or survival task.

        Args:
            predictions_df: Predictions DataFrame

        Returns:
            "classification" or "survival"
        """
        if "y_true_duration" in predictions_df.columns and "y_true_event" in predictions_df.columns:
            return "survival"
        elif "y_true" in predictions_df.columns:
            return "classification"
        else:
            raise ValueError(
                f"Cannot determine task type from columns: {predictions_df.columns.tolist()}"
            )

    def _calculate_metrics(
        self, predictions_df: pd.DataFrame, task_type: str
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        """
        Calculate appropriate metrics based on task type.

        Args:
            predictions_df: Predictions DataFrame
            task_type: "classification" or "survival"

        Returns:
            Tuple of (metrics dict, optional report dict)
        """
        if task_type == "survival":
            return self._calculate_survival_metrics(predictions_df), None
        else:
            return self._calculate_classification_metrics(predictions_df)

    def _calculate_survival_metrics(self, predictions_df: pd.DataFrame) -> dict[str, Any]:
        """
        Calculate survival analysis metrics using ConcordanceIndex metric.
        """
        durations = torch.tensor(predictions_df["y_true_duration"].values, dtype=torch.float32) # type: ignore
        events = torch.tensor(predictions_df["y_true_event"].values, dtype=torch.float32) # type: ignore
        
        # Get logits if available, otherwise reconstruct from risk scores
        if "logits" in predictions_df.columns:
            # Stack all logits into a tensor [batch_size, num_bins]
            logits_list = predictions_df["logits"].tolist()
            logits = torch.stack([torch.from_numpy(np.array(logits)) for logits in logits_list]) # type: ignore
        else:
            raise ValueError("Logits are required for survival metric calculation")

        # Initialize and compute C-index using the ConcordanceIndex metric
        c_index_metric = ConcordanceIndex()
        c_index_metric.update(logits, (durations, events))
        c_index = float(c_index_metric.compute())

        metrics: dict[str, Any] = {
            "c_index": c_index,
            "n_samples": len(predictions_df),
            "n_events": int(events.sum()),
            "event_rate": float(events.mean()),
        }

        logger.info(f"  C-index: {c_index:.4f}")
        logger.info(f"  Events: {metrics['n_events']}/{metrics['n_samples']} ({metrics['event_rate']:.2%})")

        return metrics

    def _calculate_classification_metrics(
        self, predictions_df: pd.DataFrame
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Calculate classification metrics including AUROC if probabilities available."""
        y_true = cast(np.ndarray[Any, Any], predictions_df["y_true"].values) # type: ignore
        y_pred = cast(np.ndarray[Any, Any], predictions_df["y_pred"].values) # type: ignore

        # Basic metrics
        accuracy = float(accuracy_score(y_true, y_pred))
        precision, recall, f1, _ = cast(
            tuple[float, float, float, np.ndarray[Any, Any]], 
            precision_recall_fscore_support(
                y_true, y_pred, average="weighted", zero_division=0
            )
        )

        metrics: dict[str, Any] = {
            "accuracy": accuracy,
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "n_samples": len(predictions_df),
        }

        # ROC AUC if probabilities available
        prob_cols = [col for col in predictions_df.columns if col.startswith("prob_")]
        if prob_cols and len(prob_cols) > 1:
            # Multi-class AUROC
            try:
                probs = predictions_df[prob_cols].values
                auc = float(roc_auc_score(y_true, probs, multi_class="ovr", average="weighted"))
                metrics["roc_auc_weighted"] = auc
                
                # Also compute macro average
                auc_macro = float(roc_auc_score(y_true, probs, multi_class="ovr", average="macro"))
                metrics["roc_auc_macro"] = auc_macro
                
                logger.info(f"  ROC AUC (weighted): {auc:.4f}")
                logger.info(f"  ROC AUC (macro): {auc_macro:.4f}")
            except Exception as e:
                logger.warning(f"Could not calculate ROC AUC: {e}")
        elif "probability" in predictions_df.columns:
            # Binary AUROC
            try:
                probs = cast(np.ndarray[Any, Any], predictions_df["probability"].values) # type: ignore
                auc = float(roc_auc_score(y_true, probs))
                metrics["roc_auc"] = auc
                logger.info(f"  ROC AUC: {auc:.4f}")
            except Exception as e:
                logger.warning(f"Could not calculate ROC AUC: {e}")

        # Classification report
        report_dict = cast(
            dict[str, Any], 
            classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        )

        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  F1 Score: {f1:.4f}")

        return metrics, report_dict

    def _save_predictions(
        self,
        predictions_df: pd.DataFrame,
        metrics: dict[str, Any],
        mode: PredictionMode,
        task_type: str,
    ) -> None:
        """
        Save predictions and metrics to disk.

        Args:
            predictions_df: Predictions DataFrame
            metrics: Metrics dictionary
            mode: Prediction mode
            task_type: Task type
        """
        output_dir = self.model_storage.output_dir / "external_validation"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Remove logits column before saving (too large and not human-readable)
        df_to_save = predictions_df.copy()
        if "logits" in df_to_save.columns:
            df_to_save = df_to_save.drop(columns=["logits"])

        # Save predictions
        pred_path = output_dir / f"predictions_{mode}.csv"
        df_to_save.to_csv(pred_path, index=False)
        logger.info(f"Predictions saved to: {pred_path}")

        # Save metrics
        metrics_path = output_dir / f"metrics_{mode}.json"
        import json

        with open(metrics_path, "w") as f:
            json.dump(
                {
                    "mode": str(mode),
                    "task_type": task_type,
                    "experiment": self.model_storage.experiment_name,
                    "metrics": metrics,
                },
                f,
                indent=2,
            )
        logger.info(f"Metrics saved to: {metrics_path}")

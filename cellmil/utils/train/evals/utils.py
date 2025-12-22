import lightning as Pl
from lightning import Trainer
import numpy as np
import torch
from torch.utils.data import DataLoader as DataLoaderTorch
from lightning.pytorch.callbacks import ModelCheckpoint
from sklearn.metrics import classification_report  # type: ignore
from torch_geometric.loader import DataLoader as DataLoaderPyG  # type: ignore
from cellmil.datamodels.datasets.cell_mil_dataset import CellMILDataset
from cellmil.utils.train.metrics import ConcordanceIndex
from cellmil.datamodels.datasets.cell_gnn_mil_dataset import (
    CellGNNMILDataset
)
from cellmil.datamodels.datasets.patch_gnn_mil_dataset import (
    PatchGNNMILDataset
)
from cellmil.datamodels.datasets.patch_mil_dataset import PatchMILDataset
from typing import Any, Union, cast
from cellmil.utils import logger

def is_survival_model(lit_model: Pl.LightningModule) -> bool:
    """Check if the model is a survival analysis model."""
    # Check if it's a LitSurv or similar survival model
    model_class_name = lit_model.__class__.__name__
    return "Surv" in model_class_name

def compute_slide_cell_counts(
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

class Report:
    """Base class for evaluation reports."""


    def __init__(
        self, 
        trainer: Trainer,
        lit_model: Pl.LightningModule,  
    ) -> None:
        self.trainer = trainer
        self.lit_model = lit_model
        
        self._load_best_model()
    
    def _load_best_model(
        self
    ) -> None:
        """Load the best model checkpoint if available."""
        
        try:
            # Try to find ModelCheckpoint callback and load best checkpoint
            for callback in self.trainer.callbacks:  # type: ignore
                if isinstance(callback, ModelCheckpoint) and callback.best_model_path:  # type: ignore
                    logger.info(f"Loading best model from: {callback.best_model_path}")  # type: ignore
                    # Load the state dict from the best checkpoint into the current model
                    # Set weights_only=False to handle optimizer state and other objects
                    checkpoint = torch.load(
                        callback.best_model_path,  # type: ignore
                        map_location=self.lit_model.device,  # type: ignore
                        weights_only=False,
                    )
                    self.lit_model.load_state_dict(checkpoint["state_dict"])
                    self.lit_model.eval()
                    break
        except Exception as e:
            logger.warning(
                f"Could not load best checkpoint: {e}. Using current model state."
            )
            
    def generate(self, dataloader: Union[DataLoaderTorch[Any], DataLoaderPyG]) -> dict[str, Any]:
        """Generate the report."""
        
        # Check if this is a survival model
        is_survival = is_survival_model(self.lit_model)

        if is_survival:
            # For survival models, compute C-index and Brier score
            return self._get_survival_report(dataloader)
        else:
            # For classification models, use the existing logic
            return self._get_classification_report(dataloader)
        
    
    def _get_classification_report(
        self,
        dataloader: Union[DataLoaderTorch[Any], DataLoaderPyG],
    ) -> dict[str, Any]:
        """Get classification report with precision, recall, f1-score."""
        y_pred = self.trainer.predict(self.lit_model, dataloader)
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
        self,
        dataloader: Union[DataLoaderTorch[Any], DataLoaderPyG],
    ) -> dict[str, Any]:
        """Get survival analysis report with C-index"""

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
        y_pred = self.trainer.predict(self.lit_model, dataloader)

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

        # Update metrics with logits (not hazards)
        c_index_metric.update(logits, (durations, events))

        report: dict[str, float] = {
            "c_index": float(c_index_metric.compute()),
            "n_samples": len(durations),
            "n_events": int(events.sum()),
        }

        return report
        
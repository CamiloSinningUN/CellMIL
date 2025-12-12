# -*- coding: utf-8 -*-
# Utils for CLAM Model Training
#
# References:
# Data-efficient and weakly supervised computational pathology on whole-slide images
# Lu, Ming Y et al., Nature Biomedical Engineering, 2021
# DOI: https://doi.org/10.1038/s41551-021-00707-9

import math
import numpy as np
import torch
from torch import nn
import lightning as Pl
import torchmetrics
from typing import cast, Any
from torch.optim.lr_scheduler import LRScheduler

class Accuracy_Logger(object):
    """Accuracy logger"""
    def __init__(self, n_classes: int):
        super().__init__()
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for _ in range(self.n_classes)]
    
    def log(self, Y_hat: torch.Tensor, Y: torch.Tensor):
        _Y_hat = int(Y_hat)
        _Y = int(Y)
        self.data[_Y]["count"] += 1
        self.data[_Y]["correct"] += (_Y_hat == _Y)
    
    def log_batch(
        self, 
        Y_hat: torch.Tensor, 
        Y: torch.Tensor
    ):
        _Y_hat = np.array(Y_hat).astype(int)
        _Y = np.array(Y).astype(int)
        for label_class in np.unique(_Y):
            cls_mask = _Y == label_class
            self.data[label_class]["count"] += cls_mask.sum()
            self.data[label_class]["correct"] += (_Y_hat[cls_mask] == _Y[cls_mask]).sum()
    
    def get_summary(
        self, 
        c: int
    ):
        count = self.data[c]["count"] 
        correct = self.data[c]["correct"]
        
        if count == 0: 
            acc = None
        else:
            acc = float(correct) / count
        
        return acc, correct, count

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(
        self,
        patience: int = 20, 
        stop_epoch: int = 50, 
        verbose: bool = False
    ):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(
        self, 
        epoch: int, 
        val_loss: float, 
        model: nn.Module, 
        ckpt_name: str = 'checkpoint.pt'
    ):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(
        self, 
        val_loss: float, 
        model: nn.Module, 
        ckpt_name: str
    ):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss
        
class LitGeneral(Pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss: nn.Module = nn.CrossEntropyLoss(),
        lr_scheduler: LRScheduler | None = None,
        n_classes: int = 2
    ):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.n_classes = n_classes
        self.lr_scheduler = lr_scheduler
        
        self.save_hyperparameters({
            "model": model.__class__.__name__,
            "optimizer": optimizer,
            "loss": loss
        })
        self._setup_metrics()
    
    def _setup_metrics(self):
        metrics = torchmetrics.MetricCollection({
            "accuracy": torchmetrics.Accuracy(
                task="multiclass", num_classes=self.n_classes, average="none"
            ),
            "f1": torchmetrics.F1Score(
                task="multiclass", num_classes=self.n_classes, average="macro"
            ),
            "precision": torchmetrics.Precision(
                task="multiclass", num_classes=self.n_classes, average="macro"
            ),
            "recall": torchmetrics.Recall(
                task="multiclass", num_classes=self.n_classes, average="macro"
            ),
            "auroc": torchmetrics.AUROC(
                task="multiclass", num_classes=self.n_classes, average="macro"
            ),
        })
        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def configure_optimizers(self): # type: ignore
        if self.lr_scheduler:
            scheduler: dict[str, Any] = {
                "scheduler": self.lr_scheduler,
                "interval": "epoch",
                "monitor": "val/total_loss",
                "frequency": 1,
                "strict": True,
                "name": "learning_rate",
            }
            return [self.optimizer], [scheduler]
        return [self.optimizer]
    
    def _shared_step(
        self, 
        batch: tuple[torch.Tensor, torch.Tensor],
        stage: str, 
        log: bool = True
    ):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)

        # Convert soft labels to hard labels for metrics if needed
        if y.dim() == 2:
            y = y.argmax(dim=1).long()
        else:
            y = y.long()

        if log:
            self.log(f"{stage}/total_loss", loss, prog_bar=(stage != "train"), on_step=(stage=="train"), on_epoch=True)

        return loss, logits, y

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int ):
        loss, logits, y =  self._shared_step(batch, stage="train", log=True)
        self.train_metrics(logits, y)
        return loss
        

    def on_train_epoch_end(self) -> None:
        self._flatten_and_log_metrics(
            self.train_metrics.compute(), prefix="train"
        )
        self.train_metrics.reset()
        
    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        loss, logits, y =  self._shared_step(batch, stage="val", log=True)
        self.val_metrics(logits, y)
        return loss

    def on_validation_epoch_end(self) -> None:
        self._flatten_and_log_metrics(
            self.val_metrics.compute(), prefix="val"
        )
        self.val_metrics.reset()
        
    def _flatten_and_log_metrics(self, computed: dict[str, torch.Tensor], prefix: str) -> None:
        """
        Convert metric dictionary produced by torchmetrics into a flat dict of
        scalar values and log it with `self.log_dict`.

        - Vector/tensor metrics (e.g. per-class accuracy) are expanded into
          keys like `{prefix}/class_{i}_acc`.
        - Scalar tensors are converted to floats.
        - None values are converted to NaN to satisfy loggers that expect
          numeric scalars.
        """
        flat: dict[str, float] = {}
        for key, val in computed.items():
            # Normalize key: some metrics come as 'train/accuracy' etc.; keep full key
            try:
                if val.dim() == 0:
                    flat[key] = float(val.item())
                else:
                    vals = cast(list[torch.Tensor], val.cpu().tolist()) # type: ignore
                    for i, v in enumerate(vals):
                        # Special-case accuracy to use *_acc suffix
                        if key.endswith("/accuracy"):
                            base = key.rsplit("/", 1)[0]
                            flat[f"{base}/class_{i}_acc"] = float(v)
                        else:
                            flat[f"{key}_class_{i}"] = float(v)
            except Exception:
                # Fallback: set NaN so logging doesn't fail
                flat[key] = float("nan")

        # Finally log flattened scalars
        self.log_dict(flat, prog_bar=True, batch_size=1)

    def predict_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        _, logits, _ =  self._shared_step(batch, stage="test", log=False)
        preds = logits.argmax(dim=1)
        return preds
    
class AEM():
    """
    Attention Entropy Maximization (AEM) to encourage diversity in attention weights.
    
    This class provides methods to compute the AEM weight based on cosine annealing
    and to calculate the attention entropy loss.
    """
    def __init__(
        self,
        weight_initial: float = 0.1,
        weight_final: float = 0.01,
        annealing_epochs: int = 25
    ):
        """
        Args:
            use_aem (bool): Whether to use AEM loss
            aem_weight_initial (float): Initial weight for AEM loss
            aem_weight_final (float): Final weight for AEM loss after annealing
            aem_annealing_epochs (int): Number of epochs over which to anneal the AEM weight
        """
        self.weight_initial = weight_initial
        self.weight_final = weight_final
        self.annealing_epochs = annealing_epochs
        
    def get_negative_entropy(self, attention_weights: torch.Tensor) -> torch.Tensor:
        """
        Calculate the negative entropy of attention weights to encourage diversity.

        Args:
            attention_weights (torch.Tensor): Attention weights tensor of shape (K, N) or (1, N)
                where K is number of classes/branches and N is number of instances
                
        Returns:
            torch.Tensor: Negative entropy (to maximize entropy, we minimize negative entropy)
        """
        # Add small epsilon to avoid log(0), choosing appropriate precision based on dtype
        if attention_weights.dtype == torch.float16:
            eps = 1e-7  # Safe for half precision
        else:
            eps = 1e-12
            
        attention_weights = attention_weights + eps
        
        if torch.isnan(attention_weights).any():
            print(f"Nan in attention weights: {torch.isnan(attention_weights).any().item()}")
            input("Press Enter to continue...")
        
        # Calculate entropy: -sum(p * log(p))
        entropy = -torch.sum(attention_weights * torch.log(attention_weights), dim=-1)
        
        if torch.isnan(entropy).any():
            print(f"Nan in entropy: {torch.isnan(entropy).any().item()}")
            input("Press Enter to continue...")

        return -entropy.mean()
    
    def get_weight(self, current_epoch: int) -> float:
        """
        Calculate the current AEM weight using cosine annealing.
        
        Args:
            current_epoch (int): Current training epoch
            
        Returns:
            float: Current AEM weight
        """
        if current_epoch >= self.annealing_epochs:
            return self.weight_final
            
        # Cosine annealing: weight decreases from initial to final over annealing_epochs
        progress = current_epoch / self.annealing_epochs
        
        if math.isnan(progress):
            print(f"Progress: {progress}")
            input("Press Enter to continue...")
        
        cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
        
        if math.isnan(cosine_factor):
            print(f"Cosine factor: {cosine_factor}")
            input("Press Enter to continue...")

        return self.weight_final + (self.weight_initial - self.weight_final) * cosine_factor

    def get_aem(self, current_epoch: int, attention_weights: torch.Tensor) -> torch.Tensor:
        """
        Get the term to sum to the loss function for AEM.
        Args:
            current_epoch (int): Current training epoch
            attention_weights (torch.Tensor): Attention weights tensor
        Returns:
            float: Term to add to the loss function for AEM
        """
        weight = self.get_weight(current_epoch)
        neg_entropy = self.get_negative_entropy(attention_weights)
        return weight * neg_entropy

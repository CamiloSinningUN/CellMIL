# -*- coding: utf-8 -*-
# CLAM Model Implementation
#
# References:
# Data-efficient and weakly supervised computational pathology on whole-slide images
# Lu, Ming Y et al., Nature Biomedical Engineering, 2021
# DOI: https://doi.org/10.1038/s41551-021-00707-9

from typing_extensions import Self
import torch
import wandb
import torchmetrics
import numpy as np
import torch.nn as nn
import lightning as Pl
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
from typing import IO, Any, Callable, Literal, cast
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LRScheduler
from sklearn.preprocessing import label_binarize  # type: ignore
from sklearn.metrics import roc_auc_score, roc_curve  # type: ignore
from sklearn.metrics import auc as calc_auc
from .utils import EarlyStopping, Accuracy_Logger, AEM
from cellmil.utils import logger
from cellmil.utils.train.losses import NegativeLogLikelihoodSurvLoss
from cellmil.utils.train.metrics import ConcordanceIndex, BrierScore
from topk.svm import SmoothTop1SVM  # type: ignore


class Attn_Net(nn.Module):
    """
    Attention Network without Gating.

    This class implements a basic attention mechanism using fully connected layers
    followed by a tanh activation. It is used to compute attention weights for
    Multiple Instance Learning (MIL).

    Args:
        L (int, optional): Input feature dimension. Defaults to 1024.
        D (int, optional): Hidden layer dimension. Defaults to 256.
        dropout (bool, optional): Whether to use dropout (p = 0.25). Defaults to False.
        n_classes (int, optional): Number of classes (determines output dimension). Defaults to 1.
    """

    def __init__(
        self, L: int = 1024, D: int = 256, dropout: bool = False, n_classes: int = 1
    ):
        super(Attn_Net, self).__init__()  # type: ignore
        _module: list[nn.Module] = [nn.Linear(L, D), nn.Tanh()]

        if dropout:
            _module.append(nn.Dropout(0.25))

        _module.append(nn.Linear(D, n_classes))

        self.module = nn.Sequential(*_module)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the Attention Network.

        Args:
            x (torch.Tensor): Input tensor of shape (N, L), where N is batch size and
                             L is the input feature dimension.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - The attention scores after processing (shape: N x n_classes)
                - The original input tensor (shape: N x L)
        """
        return self.module(x), x  # N x n_classes


class Attn_Net_Gated(nn.Module):
    """
    Attention Network with Sigmoid Gating.

    This class implements a gated attention mechanism using two parallel pathways:
    - One path with linear layer followed by tanh activation
    - Another path with linear layer followed by sigmoid activation

    These paths are combined via element-wise multiplication (gating mechanism)
    and passed through a final linear layer to compute attention weights.

    Args:
        L (int, optional): Input feature dimension. Defaults to 1024.
        D (int, optional): Hidden layer dimension. Defaults to 256.
        dropout (bool, optional): Whether to use dropout (p = 0.25) in both pathways. Defaults to False.
        n_classes (int, optional): Number of classes (determines output dimension). Defaults to 1.
    """

    def __init__(
        self, L: int = 1024, D: int = 256, dropout: bool = False, n_classes: int = 1
    ):
        super(Attn_Net_Gated, self).__init__()  # type: ignore
        _attention_a: list[nn.Module] = [nn.Linear(L, D), nn.Tanh()]

        _attention_b: list[nn.Module] = [nn.Linear(L, D), nn.Sigmoid()]
        if dropout:
            _attention_a.append(nn.Dropout(0.25))
            _attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*_attention_a)
        self.attention_b = nn.Sequential(*_attention_b)

        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the Gated Attention Network.

        This method implements a gated attention mechanism where two parallel paths process the input:
        - Path A: Linear -> Tanh activation
        - Path B: Linear -> Sigmoid activation

        These paths are then combined via element-wise multiplication and passed through
        a final linear layer to produce attention scores.

        Args:
            x (torch.Tensor): Input tensor of shape (N, L), where N is batch size and
                             L is the input feature dimension.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - The attention scores after the final linear layer (shape: N x n_classes)
                - The original input tensor (shape: N x L)
        """
        a = self.attention_a(x)
        b = self.attention_b(x)
        c = self.attention_c(a.mul(b))  # N x n_classes
        return c, x


class CLAM_SB(nn.Module):
    """
    CLAM Single Branch (SB) - Clustering-constrained Attention Multiple Instance Learning model.

    This model uses attention mechanisms to aggregate features from multiple instances (patches)
    in a bag for classification. It supports instance-level evaluation and can handle both
    binary and multi-class classification problems.

    Args:
        gate (bool, optional): Whether to use gated attention network. If True, uses Attn_Net_Gated,
            otherwise uses Attn_Net. Defaults to True.
        size_arg (Literal['small', 'big'], list, optional): Configuration for network size.
            'small': [embed_dim, 512, 256], 'big': [embed_dim, 512, 384]. Defaults to "small".
        dropout (bool, optional): Whether to use dropout (p = 0.25) in attention networks and
            feature layers. Defaults to False.
        k_sample (int, optional): Number of positive/negative patches to sample for instance-level
            training. Used in inst_eval methods. Defaults to 8.
        n_classes (int, optional): Number of classes for classification. Defaults to 2.
        instance_loss_fn (nn.Module, optional): Loss function to supervise instance-level training.
            Defaults to nn.CrossEntropyLoss().
        subtyping (bool, optional): Whether this is a subtyping problem. Affects instance-level
            evaluation for out-of-class samples. Defaults to False.
        embed_dim (int, optional): Input embedding dimension. Defaults to 1024.
        temperature (float, optional): Temperature parameter for softmax. Defaults to 1.0.
    """

    def __init__(
        self,
        gate: bool = True,
        size_arg: Literal["small", "big"] | list[int] = "small",
        dropout: bool = False,
        k_sample: int = 8,
        n_classes: int = 2,
        instance_loss_fn: nn.Module = SmoothTop1SVM(n_classes=2).cuda()
        if torch.cuda.is_available()
        else SmoothTop1SVM(n_classes=2),
        subtyping: bool = False,
        embed_dim: int = 1024,
        temperature: float = 1.0,
    ):
        super().__init__()  # type: ignore
        self.size_dict = {"small": [embed_dim, 512, 256], "big": [embed_dim, 512, 384]}
        if isinstance(size_arg, list):
            if len(size_arg) != 2:
                raise ValueError("size_arg must be a list of length 2")
            size = [embed_dim, *size_arg]
        else:
            size = self.size_dict[size_arg]
        fc: list[nn.Module] = [
            nn.Linear(size[0], size[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
        ]
        if gate:
            attention_net = Attn_Net_Gated(
                L=size[1], D=size[2], dropout=dropout, n_classes=1
            )
        else:
            attention_net = Attn_Net(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifiers = nn.Linear(size[1], n_classes)
        instance_classifiers = [nn.Linear(size[1], 2) for _ in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping
        self.temperature = temperature

    def __str__(self) -> str:
        return "<CLAM_SB>"

    @staticmethod
    def create_positive_targets(length: int, device: torch.device) -> torch.Tensor:
        """
        Create a tensor of positive targets (all ones).

        Args:
            length (int): The length of the tensor to create.
            device (torch.device): The device to create the tensor on.

        Returns:
            torch.Tensor: A tensor of ones of the specified length.
        """
        return torch.full((length,), 1, device=device).long()

    @staticmethod
    def create_negative_targets(length: int, device: torch.device) -> torch.Tensor:
        """
        Create a tensor of negative targets (all zeros).

        Args:
            length (int): The length of the tensor to create.
            device (torch.device): The device to create the tensor on.

        Returns:
            torch.Tensor: A tensor of zeros of the specified length.
        """
        return torch.full((length,), 0, device=device).long()

    # instance-level evaluation for in-the-class attention branch
    def inst_eval(
        self, a: torch.Tensor, h: torch.Tensor, classifier: nn.Module
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Instance-level evaluation for in-the-class attention branch.

        This method evaluates the model at the instance level by:
        1. Selecting the top k instances with highest attention scores (positive)
        2. Selecting the top k instances with lowest attention scores (negative)
        3. Creating targets for these instances
        4. Computing loss and predictions using the classifier

        Args:
            a (torch.Tensor): Attention scores tensor.
            h (torch.Tensor): Features tensor.
            classifier (nn.Module): Instance-level classifier.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - instance_loss: The loss for instance-level classification
                - all_preds: The predicted labels for all selected instances
                - all_targets: The target labels for all selected instances
        """
        device = h.device
        if len(a.shape) == 1:
            a = a.view(1, -1)
        top_p_ids = torch.topk(a, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        top_n_ids = torch.topk(-a, self.k_sample, dim=1)[1][-1]
        top_n = torch.index_select(h, dim=0, index=top_n_ids)
        p_targets = self.create_positive_targets(self.k_sample, device)
        n_targets = self.create_negative_targets(self.k_sample, device)

        all_targets = torch.cat([p_targets, n_targets], dim=0)
        all_instances = torch.cat([top_p, top_n], dim=0)
        logits = classifier(all_instances)
        all_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, all_targets)
        return instance_loss, all_preds, all_targets

    # instance-level evaluation for out-of-the-class attention branch
    def inst_eval_out(
        self, a: torch.Tensor, h: torch.Tensor, classifier: nn.Module
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Instance-level evaluation for out-of-the-class attention branch.

        This method evaluates the model at the instance level for out-of-class samples by:
        1. Selecting the top k instances with highest attention scores
        2. Creating negative targets for these instances (since they should be negative for out-of-class)
        3. Computing loss and predictions using the classifier

        Args:
            a (torch.Tensor): Attention scores tensor.
            h (torch.Tensor): Features tensor.
            classifier (nn.Module): Instance-level classifier.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - instance_loss: The loss for instance-level classification
                - p_preds: The predicted labels for the selected instances
                - p_targets: The target labels for the selected instances
        """
        device = h.device
        if len(a.shape) == 1:
            a = a.view(1, -1)
        top_p_ids = torch.topk(a, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        p_targets = self.create_negative_targets(self.k_sample, device)
        logits = classifier(top_p)
        p_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, p_targets)
        return instance_loss, p_preds, p_targets

    def forward(
        self,
        h: torch.Tensor,
        label: torch.Tensor | None = None,
        instance_eval: bool = False,
        return_features: bool = False,
        attention_only: bool = False,
    ) -> (
        torch.Tensor
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]
    ):
        """
        Forward pass of the CLAM Single Branch model.

        Args:
            h (torch.Tensor): Input feature tensor of shape (N, embed_dim), where N is the number
                of instances (patches) and embed_dim is the feature dimension.
            label (torch.Tensor | None, optional): Ground truth labels for instance-level evaluation.
                Required when instance_eval=True. Should be of shape (1,) for single class or
                (n_classes,) for multi-class. Defaults to None.
            instance_eval (bool, optional): Whether to perform instance-level evaluation and compute
                instance loss. Requires label to be provided. Defaults to False.
            return_features (bool, optional): Whether to return aggregated features (M) in the
                results dictionary. Defaults to False.
            attention_only (bool, optional): If True, returns only attention weights without
                classification. Defaults to False.

        Returns:
            torch.Tensor | tuple:
                - If attention_only=True: Returns attention weights tensor of shape (K, N)
                - Otherwise: Returns tuple of (logits, Y_prob, Y_hat, a_raw, results_dict) where:
                    - logits (torch.Tensor): Raw classification logits of shape (1, n_classes)
                    - Y_prob (torch.Tensor): Softmax probabilities of shape (1, n_classes)
                    - Y_hat (torch.Tensor): Predicted class indices of shape (1, 1)
                    - a_raw (torch.Tensor): Raw attention weights before softmax of shape (K, N)
                    - results_dict (dict): Dictionary containing:
                        - 'instance_loss': Instance-level loss (if instance_eval=True)
                        - 'inst_labels': Instance-level target labels (if instance_eval=True)
                        - 'inst_preds': Instance-level predictions (if instance_eval=True)
                        - 'features': Aggregated features M (if return_features=True)
        """
        a, h = self.attention_net(h)  # Nx1
        a = torch.transpose(a, 1, 0)  # 1xN
        a = F.softmax(a / self.temperature, dim=1)  # softmax over N
        if attention_only:
            return a

        if instance_eval and label is not None:
            total_inst_loss = 0.0
            all_preds: list[np.ndarray[Any, Any]] = []
            all_targets: list[np.ndarray[Any, Any]] = []
            inst_labels = F.one_hot(
                label, num_classes=self.n_classes
            ).squeeze()  # binarize label
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1:  # in-the-class:
                    instance_loss, preds, targets = self.inst_eval(a, h, classifier)
                    all_preds.extend(preds.cpu().numpy())  # type: ignore
                    all_targets.extend(targets.cpu().numpy())  # type: ignore
                else:  # out-of-the-class
                    if self.subtyping:
                        instance_loss, preds, targets = self.inst_eval_out(
                            a, h, classifier
                        )
                        all_preds.extend(preds.cpu().numpy())  # type: ignore
                        all_targets.extend(targets.cpu().numpy())  # type: ignore
                    else:
                        continue
                total_inst_loss += instance_loss

            if self.subtyping:
                total_inst_loss /= len(self.instance_classifiers)

            results_dict: dict[str, Any] = {
                "instance_loss": total_inst_loss,
                "inst_labels": np.array(all_targets),
                "inst_preds": np.array(all_preds),
            }
        else:
            results_dict = {}

        M = torch.mm(a, h)
        logits = self.classifiers(M)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        Y_prob = F.softmax(logits, dim=1)

        if return_features:
            results_dict.update({"features": M})
        return logits, Y_prob, Y_hat, a, results_dict


class CLAM_MB(CLAM_SB):
    """
    CLAM Multi-Branch (MB) - Clustering-constrained Attention Multiple Instance Learning model.

    This class extends CLAM_SB by using a multi-branch architecture where each class has
    its own attention branch and classifier. This architecture is more suitable for
    multi-class classification problems.

    Args:
        gate (bool, optional): Whether to use gated attention network. Defaults to True.
        size_arg (Literal["small", "big"], list, optional): Configuration for network size. Defaults to "small".
        dropout (bool, optional): Whether to use dropout. Defaults to False.
        k_sample (int, optional): Number of positive/negative patches to sample for instance-level
            training. Defaults to 8.
        n_classes (int, optional): Number of classes. Defaults to 2.
        instance_loss_fn (nn.Module, optional): Loss function for instance-level training.
            Defaults to nn.CrossEntropyLoss().
        subtyping (bool, optional): Whether it's a subtyping problem. Defaults to False.
        embed_dim (int, optional): Input embedding dimension. Defaults to 1024.
        temperature (float, optional): Temperature parameter for softmax. Defaults to 1.0.
    """

    def __init__(
        self,
        gate: bool = True,
        size_arg: Literal["small", "big"] | list[int] = "small",
        dropout: bool = False,
        k_sample: int = 8,
        n_classes: int = 2,
        instance_loss_fn: nn.Module = SmoothTop1SVM(n_classes=2),
        subtyping: bool = False,
        embed_dim: int = 1024,
        temperature: float = 1.0,
    ):
        nn.Module.__init__(self)  # type: ignore
        self.size_dict = {"small": [embed_dim, 512, 256], "big": [embed_dim, 512, 384]}
        if isinstance(size_arg, list):
            if len(size_arg) != 2:
                raise ValueError("size_arg must be a list of length 2")
            size = [embed_dim, *size_arg]
        else:
            size = self.size_dict[size_arg]
        fc: list[nn.Module] = [
            nn.Linear(size[0], size[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
        ]
        if gate:
            attention_net = Attn_Net_Gated(
                L=size[1], D=size[2], dropout=dropout, n_classes=n_classes
            )
        else:
            attention_net = Attn_Net(
                L=size[1], D=size[2], dropout=dropout, n_classes=n_classes
            )
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        bag_classifiers = [
            nn.Linear(size[1], 1) for _ in range(n_classes)
        ]  # use an indepdent linear layer to predict each class
        self.classifiers = nn.ModuleList(bag_classifiers)
        instance_classifiers = [nn.Linear(size[1], 2) for _ in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping
        self.temperature = temperature

    def __str__(self) -> str:
        return "<CLAM_MB>"

    def forward(
        self,
        h: torch.Tensor,
        label: torch.Tensor | None = None,
        instance_eval: bool = False,
        return_features: bool = False,
        attention_only: bool = False,
    ) -> (
        torch.Tensor
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]
    ):
        """
        Forward pass of the CLAM Multi-Branch model.

        This method extends the CLAM_SB forward pass by using multiple attention branches,
        one for each class. Each branch computes its own attention weights and features,
        which are then processed by class-specific classifiers.

        Args:
            h (torch.Tensor): Input feature tensor of shape (N, embed_dim), where N is
                the number of instances (patches) and embed_dim is the feature dimension.
            label (torch.Tensor | None, optional): Ground truth labels for instance-level
                evaluation. Required when instance_eval=True. Defaults to None.
            instance_eval (bool, optional): Whether to perform instance-level evaluation
                and compute instance loss. Defaults to False.
            return_features (bool, optional): Whether to return aggregated features in
                the results dictionary. Defaults to False.
            attention_only (bool, optional): If True, returns only attention weights without
                classification. Defaults to False.

        Returns:
            torch.Tensor | tuple:
                - If attention_only=True: Returns attention weights tensor of shape (K, N)
                - Otherwise: Returns tuple of (logits, Y_prob, Y_hat, a_raw, results_dict) where:
                    - logits (torch.Tensor): Raw classification logits of shape (1, n_classes)
                    - Y_prob (torch.Tensor): Softmax probabilities of shape (1, n_classes)
                    - Y_hat (torch.Tensor): Predicted class indices of shape (1, 1)
                    - a_raw (torch.Tensor): Raw attention weights before softmax of shape (K, N)
                    - results_dict (dict): Dictionary containing:
                        - 'instance_loss': Instance-level loss (if instance_eval=True)
                        - 'inst_labels': Instance-level target labels (if instance_eval=True)
                        - 'inst_preds': Instance-level predictions (if instance_eval=True)
                        - 'features': Aggregated features M (if return_features=True)
        """
        a, h = self.attention_net(h)  # NxK
        a = torch.transpose(a, 1, 0)  # KxN
        a = F.softmax(a / self.temperature, dim=1)  # softmax over N
        if attention_only:
            return a

        if instance_eval and label is not None:
            total_inst_loss = 0.0
            all_preds: list[np.ndarray[Any, Any]] = []
            all_targets: list[np.ndarray[Any, Any]] = []
            inst_labels = F.one_hot(
                label, num_classes=self.n_classes
            ).squeeze()  # binarize label
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1:  # in-the-class:
                    instance_loss, preds, targets = self.inst_eval(a[i], h, classifier)
                    all_preds.extend(preds.cpu().numpy())  # type: ignore
                    all_targets.extend(targets.cpu().numpy())  # type: ignore
                else:  # out-of-the-class
                    if self.subtyping:
                        instance_loss, preds, targets = self.inst_eval_out(
                            a[i], h, classifier
                        )
                        all_preds.extend(preds.cpu().numpy())  # type: ignore
                        all_targets.extend(targets.cpu().numpy())  # type: ignore
                    else:
                        continue
                total_inst_loss += instance_loss

            if self.subtyping:
                total_inst_loss /= len(self.instance_classifiers)

            results_dict: dict[str, Any] = {
                "instance_loss": total_inst_loss,
                "inst_labels": np.array(all_targets),
                "inst_preds": np.array(all_preds),
            }
        else:
            results_dict = {}

        M = torch.mm(a, h)

        logits = torch.empty(1, self.n_classes).float().to(M.device)
        for c in range(self.n_classes):
            logits[0, c] = self.classifiers[c](M[c])

        Y_hat = torch.topk(logits, 1, dim=1)[1]
        Y_prob = F.softmax(logits, dim=1)

        if return_features:
            results_dict.update({"features": M})
        return logits, Y_prob, Y_hat, a, results_dict


class CLAMTrainerLegacy:
    """
    Trainer class for CLAM models.

    This class handles the training loop, validation, and evaluation of CLAM models.
    It supports early stopping, metrics logging, and checkpointing.

    Args:
        model (CLAM_MB | CLAM_SB): The CLAM model to train.
        optimizer (torch.optim.Optimizer): Optimizer for model training.
        device (str): Device to use for training ("cuda", "cpu", etc.)
        ckpt_path (Path): Path to save checkpoints.
        weight_loss_slide (float, optional): Weight for slide-level loss. Defaults to 0.7.
        loss_slide (nn.Module, optional): Loss function for slide-level classification. Defaults to nn.CrossEntropyLoss().
        early_stopping (EarlyStopping | None, optional): Early stopping controller. Defaults to EarlyStopping with patience=20.
        use_wandb (bool, optional): Whether to log metrics to Weights & Biases. Defaults to True.
    """

    def __init__(
        self,
        model: CLAM_MB | CLAM_SB,
        optimizer: torch.optim.Optimizer,
        device: str,
        ckpt_path: Path,
        weight_loss_slide: float = 0.7,
        loss_slide: nn.Module = nn.CrossEntropyLoss(),
        early_stopping: EarlyStopping | None = EarlyStopping(
            patience=20, stop_epoch=50, verbose=True
        ),
        use_wandb: bool = True,
        scale_attention_grads_by_bag: bool = False,
        attn_ref_bag_size: int = 100000,
        attn_alpha: float = 0.5,
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = torch.device(device)
        self.weight_loss_slide = weight_loss_slide
        self.loss_slide = loss_slide
        self.early_stopping = early_stopping
        self.ckpt_path = ckpt_path
        self.use_wandb = use_wandb
        self.model.to(self.device)
        self.scale_attention_grads_by_bag = scale_attention_grads_by_bag
        self.attn_ref_bag_size = max(1, int(attn_ref_bag_size))
        self.attn_alpha = float(attn_alpha)

    def fit(
        self,
        train_loader: DataLoader[tuple[torch.Tensor, int]],
        val_loader: DataLoader[tuple[torch.Tensor, int]],
        epochs: int,
    ):
        """
        Train the CLAM model.

        This method runs the training loop for a specified number of epochs,
        with validation after each epoch. It supports early stopping and
        saves the best model based on validation loss.

        Args:
            train_loader (DataLoader[tuple[torch.Tensor, int]]): DataLoader for training data.
            val_loader (DataLoader[tuple[torch.Tensor, int]]): DataLoader for validation data.
            epochs (int): Number of epochs to train for.
        """
        for epoch in range(epochs):
            logger.info(f"Starting epoch {epoch}")
            train_metrics = self._train_epoch(epoch, train_loader)
            val_metrics = self._val(epoch, val_loader)

            # Log metrics for the epoch (both to logger and wandb if enabled)
            self._log_epoch_metrics(epoch, train_metrics, val_metrics)

            if val_metrics.get("early_stop", False):
                logger.info("Early stopping triggered")
                break

        if self.early_stopping:
            logger.info(f"Loading best model from {self.ckpt_path}")
            self.model.load_state_dict(torch.load(self.ckpt_path))
        else:
            logger.info(f"Saving final model to {self.ckpt_path}")
            torch.save(self.model.state_dict(), self.ckpt_path)

    def _log_epoch_metrics(
        self, epoch: int, train_metrics: dict[str, Any], val_metrics: dict[str, Any]
    ) -> None:
        """
        Log metrics for the current epoch.

        This method logs training and validation metrics to the logger and optionally
        to Weights & Biases if enabled.

        Args:
            epoch (int): Current epoch number.
            train_metrics (dict[str, Any]): Dictionary of training metrics.
            val_metrics (dict[str, Any]): Dictionary of validation metrics.
        """
        # Combine metrics for logging
        metrics: dict[str, int | float | None] = {
            "epoch": epoch,
            "train/loss": train_metrics.get("loss", 0.0),
            "train/error": train_metrics.get("error", 0.0),
            "train/inst_loss": train_metrics.get("inst_loss", 0.0),
            "val/loss": val_metrics.get("loss", 0.0),
            "val/error": val_metrics.get("error", 0.0),
            "val/inst_loss": val_metrics.get("inst_loss", 0.0),
            "val/auc": val_metrics.get("auc", 0.0),
        }

        # Add class accuracy metrics
        for i in range(self.model.n_classes):
            if f"class_{i}_acc" in train_metrics:
                metrics[f"train/class_{i}_acc"] = train_metrics[f"class_{i}_acc"]
            if f"class_{i}_acc" in val_metrics:
                metrics[f"val/class_{i}_acc"] = val_metrics[f"class_{i}_acc"]

        # Add clustering accuracy metrics if available
        for i in range(min(2, self.model.n_classes)):  # Usually just binary
            if f"cluster_{i}_acc" in train_metrics:
                metrics[f"train/cluster_{i}_acc"] = train_metrics[f"cluster_{i}_acc"]
            if f"cluster_{i}_acc" in val_metrics:
                metrics[f"val/cluster_{i}_acc"] = val_metrics[f"cluster_{i}_acc"]

        # Log to wandb if enabled
        if self.use_wandb:
            wandb.log(metrics)

    def _train_epoch(
        self,
        epoch: int,
        train_loader: DataLoader[tuple[torch.Tensor, int]],
    ) -> dict[str, float | int | None]:
        """
        Train the model for one epoch.

        This method processes all batches in the training data loader for one epoch.
        It computes loss, performs backpropagation, and collects training metrics.

        Args:
            epoch (int): Current epoch number.
            train_loader (DataLoader[tuple[torch.Tensor, int]]): DataLoader for training data.

        Returns:
            dict[str, float | int | None]: Dictionary of training metrics for the epoch.
        """
        self.model.train()

        acc_logger = Accuracy_Logger(self.model.n_classes)
        inst_logger = Accuracy_Logger(self.model.n_classes)

        train_loss: float = 0.0
        train_error: float = 0.0
        train_inst_loss: float = 0.0
        inst_count: int = 0

        # Create progress bar for training
        train_pbar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch} - Training",
            leave=False,
        )

        for batch_idx, (data, label) in train_pbar:
            data, label = data.to(self.device), label.to(self.device)
            logits, _, Y_hat, _, instance_dict = self.model(
                data, label=label, instance_eval=True
            )

            acc_logger.log(Y_hat, label)

            loss = self.loss_slide(logits, label)
            loss_value = loss.item()

            instance_loss = instance_dict["instance_loss"]
            inst_count += 1
            instance_loss_value = instance_loss.item()
            train_inst_loss += instance_loss_value

            total_loss = (
                self.weight_loss_slide * loss
                + (1 - self.weight_loss_slide) * instance_loss
            )

            inst_preds = instance_dict["inst_preds"]
            inst_labels = instance_dict["inst_labels"]
            inst_logger.log_batch(inst_preds, inst_labels)

            train_loss += loss_value
            if (batch_idx + 1) % 20 == 0:
                logger.info(
                    "Batch {}, loss: {:.4f}, instance_loss: {:.4f}, weighted_loss: {:.4f}, "
                    "label: {}, bag_size: {}".format(
                        batch_idx,
                        loss_value,
                        instance_loss_value,
                        total_loss.item(),
                        label.item(),
                        data.size(0),
                    )
                )

                # Log batch metrics to wandb if enabled
                if self.use_wandb:
                    wandb.log(
                        {
                            "batch/loss": loss_value,
                            "batch/instance_loss": instance_loss_value,
                            "batch/total_loss": total_loss.item(),
                            "batch/bag_size": data.size(0),
                        }
                    )

            error = self.calculate_error(Y_hat, label)
            train_error += error

            # Update progress bar with current metrics
            train_pbar.set_postfix(
                {  # type: ignore
                    "Loss": f"{loss_value:.4f}",
                    "Inst_Loss": f"{instance_loss_value:.4f}",
                    "Avg_Loss": f"{train_loss / (batch_idx + 1):.4f}",
                    "Error": f"{error:.4f}",
                }
            )

            # backward pass
            total_loss.backward()
            # scale attention gradients directly based on current bag size
            if self.scale_attention_grads_by_bag:
                bag_size = max(1, int(data.size(0)))
                # factor = clamp((Nref / N)^alpha, [min_scale, max_scale])
                factor = (self.attn_ref_bag_size / float(bag_size)) ** self.attn_alpha
                for p in self.model.attention_net.parameters():
                    if p.grad is not None:
                        p.grad.mul_(factor)
            # step
            self.optimizer.step()
            self.optimizer.zero_grad()

        # calculate loss and error for epoch
        train_loss /= len(train_loader)
        train_error /= len(train_loader)

        # Collect metrics to return
        metrics: dict[str, float | int | None] = {
            "loss": train_loss,
            "error": train_error,
        }

        if inst_count > 0:
            train_inst_loss /= inst_count
            metrics["inst_loss"] = train_inst_loss

            logger.info("Instance-level clustering metrics:")
            for i in range(2):
                acc, correct, count = inst_logger.get_summary(i)
                metrics[f"cluster_{i}_acc"] = acc
                logger.info(
                    f"Class {i} clustering accuracy: {acc:.4f}, correct: {correct}/{count}"
                )

        logger.info(
            f"Epoch: {epoch}, train_loss: {train_loss:.4f}, "
            f"train_clustering_loss: {train_inst_loss:.4f}, train_error: {train_error:.4f}"
        )

        # Log class-specific accuracy
        for i in range(self.model.n_classes):
            acc, correct, count = acc_logger.get_summary(i)
            metrics[f"class_{i}_acc"] = acc
            logger.info(f"Class {i}: accuracy {acc:.4f}, correct {correct}/{count}")

        return metrics

    def _val(
        self,
        epoch: int,
        val_loader: DataLoader[tuple[torch.Tensor, int]],
    ) -> dict[str, float | int | None]:
        """
        Validate the model on the validation set.

        This method evaluates the model on the validation data and computes various
        metrics including loss, error rate, AUC, and class-specific accuracy.

        Args:
            epoch (int): Current epoch number.
            val_loader (DataLoader[tuple[torch.Tensor, int]]): DataLoader for validation data.

        Returns:
            dict[str, float | int | None]: Dictionary of validation metrics for the epoch.
        """
        self.model.eval()
        acc_logger = Accuracy_Logger(self.model.n_classes)
        inst_logger = Accuracy_Logger(self.model.n_classes)
        val_loss = 0.0
        val_error = 0.0

        val_inst_loss: float = 0.0
        inst_count: int = 0

        prob = np.zeros((len(val_loader), self.model.n_classes))
        labels = np.zeros(len(val_loader))

        # Create progress bar for validation
        val_pbar = tqdm(
            enumerate(val_loader),
            total=len(val_loader),
            desc=f"Epoch {epoch} - Validation",
            leave=False,
        )

        with torch.inference_mode():
            for batch_idx, (data, label) in val_pbar:
                data, label = data.to(self.device), label.to(self.device)
                logits, Y_prob, Y_hat, _, instance_dict = self.model(
                    data, label=label, instance_eval=True
                )

                acc_logger.log(Y_hat, label)

                loss = self.loss_slide(logits, label)

                val_loss += loss.item()

                instance_loss = instance_dict["instance_loss"]

                inst_count += 1
                instance_loss_value = instance_loss.item()
                val_inst_loss += instance_loss_value

                inst_preds = instance_dict["inst_preds"]
                inst_labels = instance_dict["inst_labels"]

                inst_logger.log_batch(inst_preds, inst_labels)

                prob[batch_idx] = Y_prob.cpu().numpy()
                labels[batch_idx] = label.item()

                error = self.calculate_error(Y_hat, label)
                val_error += error

                # Update progress bar with current metrics
                val_pbar.set_postfix(
                    {  # type: ignore
                        "Loss": f"{loss.item():.4f}",
                        "Inst_Loss": f"{instance_loss_value:.4f}",
                        "Avg_Loss": f"{val_loss / (batch_idx + 1):.4f}",
                        "Error": f"{error:.4f}",
                    }
                )

        val_error /= len(val_loader)
        val_loss /= len(val_loader)

        # Calculate AUC
        if self.model.n_classes == 2:
            auc = roc_auc_score(labels, prob[:, 1])
            aucs = []
        else:
            aucs: list[float] = []
            binary_labels = cast(
                np.ndarray[Any, Any],
                label_binarize(
                    labels, classes=[i for i in range(self.model.n_classes)]
                ),
            )
            for class_idx in range(self.model.n_classes):
                if class_idx in labels:
                    fpr, tpr, _ = cast(
                        tuple[
                            np.ndarray[Any, Any],
                            np.ndarray[Any, Any],
                            np.ndarray[Any, Any],
                        ],
                        roc_curve(binary_labels[:, class_idx], prob[:, class_idx]),
                    )
                    aucs.append(float(calc_auc(fpr, tpr)))
                else:
                    aucs.append(float("nan"))

            auc = np.nanmean(np.array(aucs))

        # Collect metrics to return
        metrics: dict[str, float | int | None] = {
            "loss": val_loss,
            "error": val_error,
            "auc": float(auc),
        }

        logger.info(
            f"Validation Set, val_loss: {val_loss:.4f}, val_error: {val_error:.4f}, auc: {auc:.4f}"
        )

        if inst_count > 0:
            val_inst_loss /= inst_count
            metrics["inst_loss"] = val_inst_loss

            logger.info("Instance-level clustering metrics:")
            for i in range(2):
                acc, correct, count = inst_logger.get_summary(i)
                metrics[f"cluster_{i}_acc"] = acc
                logger.info(
                    f"Class {i} clustering accuracy: {acc:.4f}, correct: {correct}/{count}"
                )

        # Log class-specific accuracy
        for i in range(self.model.n_classes):
            acc, correct, count = acc_logger.get_summary(i)
            metrics[f"class_{i}_acc"] = acc
            logger.info(f"Class {i}: accuracy {acc:.4f}, correct {correct}/{count}")

        # Handle early stopping
        early_stop = False
        if self.early_stopping:
            self.early_stopping(epoch, val_loss, self.model, str(self.ckpt_path))
            if self.early_stopping.early_stop:
                logger.info("Early stopping")
                early_stop = True

        metrics["early_stop"] = early_stop
        return metrics

    @staticmethod
    def calculate_error(Y_hat: torch.Tensor, Y: torch.Tensor):
        """
        Calculate classification error.

        Args:
            Y_hat (torch.Tensor): Predicted labels.
            Y (torch.Tensor): Ground truth labels.

        Returns:
            float: Error rate (1 - accuracy).
        """
        return 1.0 - Y_hat.float().eq(Y.float()).float().mean().item()

    def eval(
        self,
        test_loader: DataLoader[tuple[torch.Tensor, int]],
    ):
        """
        Evaluate the model on a test set.

        Args:
            test_loader (DataLoader[tuple[torch.Tensor, int]]): DataLoader for test data.

        Note:
            This is a placeholder method for model evaluation. Implementation is not provided.
        """
        pass


class LitCLAM(Pl.LightningModule):
    @staticmethod
    def _is_gated_attention(model: CLAM_MB | CLAM_SB) -> bool:
        """Check if model uses gated attention."""
        if hasattr(model, "attention_net"):
            return any(
                isinstance(m, Attn_Net_Gated) for m in model.attention_net.modules()
            )
        return True

    @staticmethod
    def _get_size_args(model: CLAM_MB | CLAM_SB) -> list[int]:
        """Extract L and D parameters from attention network (size[1] and size[2])."""
        if hasattr(model, "attention_net"):
            for module in model.attention_net.modules():
                if isinstance(module, Attn_Net_Gated):
                    # For gated attention: attention_a and attention_b are Sequential with Linear layers
                    linear_layer = module.attention_a[0]  # First layer is Linear
                    if isinstance(linear_layer, nn.Linear):
                        l_dim = linear_layer.in_features  # size[1]
                        d_dim = linear_layer.out_features  # size[2]
                        return [l_dim, d_dim]
                elif isinstance(module, Attn_Net):
                    # For regular attention: module is Sequential with Linear layers
                    linear_layer = module.module[0]  # First layer is Linear
                    if isinstance(linear_layer, nn.Linear):
                        l_dim = linear_layer.in_features  # size[1]
                        d_dim = linear_layer.out_features  # size[2]
                        return [l_dim, d_dim]
        return [512, 256]  # default

    def __init__(
        self,
        model: CLAM_MB | CLAM_SB,
        optimizer: torch.optim.Optimizer,
        loss_slide: nn.Module = nn.CrossEntropyLoss(),
        weight_loss_slide: float = 0.7,
        lr_scheduler: LRScheduler | None = None,
        subsampling: float = 1.0,
        use_aem: bool = False,
        aem_weight_initial: float = 0.0001,
        aem_weight_final: float = 0.0,
        aem_annealing_epochs: int = 50,
    ):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.loss_slide = loss_slide
        self.weight_loss_slide = weight_loss_slide
        self.lr_scheduler = lr_scheduler
        self.subsampling = subsampling

        self.use_aem = use_aem

        if self.use_aem:
            self.aem = AEM(
                weight_initial=aem_weight_initial,
                weight_final=aem_weight_final,
                annealing_epochs=aem_annealing_epochs,
            )

        # Save all hyperparameters including model config
        model_config: dict[str, Any] = {
            "model_class": model.__class__.__name__,
            "gated": self._is_gated_attention(model),
            "size_arg": self._get_size_args(model),
            "n_classes": model.n_classes,
            "instance_loss_fn": model.instance_loss_fn.__class__.__name__,
            "k_sample": model.k_sample,
            "subtyping": model.subtyping,
            "embed_dim": model.size_dict.get("small", [1024])[0]
            if hasattr(model, "size_dict")
            else 1024,
            "dropout": any(
                isinstance(m, nn.Dropout) for m in model.attention_net.modules()
            )
            if hasattr(model, "attention_net")
            else False,
            "temperature": model.temperature if hasattr(model, "temperature") else 1.0,
        }

        self.save_hyperparameters(
            {
                **model_config,
                "optimizer_class": optimizer.__class__.__name__,
                "optimizer_lr": optimizer.param_groups[0]["lr"],
                "loss_slide": loss_slide,
                "weight_loss_slide": weight_loss_slide,
                "lr_scheduler": lr_scheduler.__class__.__name__
                if lr_scheduler
                else None,
                "subsampling": subsampling,
                "use_aem": use_aem,
                "aem_weight_initial": aem_weight_initial,
                "aem_weight_final": aem_weight_final,
                "aem_annealing_epochs": aem_annealing_epochs,
            }
        )
        self._setup_metrics()

        self.bag_size: int = 0

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str | Path | IO[bytes],
        map_location: torch.device
        | str
        | int
        | Callable[[torch.UntypedStorage, str], torch.UntypedStorage | None]
        | dict[torch.device | str | int, torch.device | str | int]
        | None = None,
        hparams_file: str | Path | None = None,
        strict: bool | None = None,
        **kwargs: Any,
    ) -> Self:
        """
        Load a LitCLAM model from a checkpoint file.

        Args:
            checkpoint_path (str | Path | IO[bytes]): Path to the checkpoint file.
            map_location: Device mapping for loading the model.
            hparams_file (str | Path | None): Optional path to a YAML file with hyperparameters.
            strict (bool | None): Whether to strictly enforce that the keys in state_dict match the keys returned by the model's state_dict function.
            **kwargs: Additional keyword arguments.
        Returns:
            LitCLAM: The loaded LitCLAM model.
        """

        checkpoint = torch.load(
            checkpoint_path, map_location=map_location, weights_only=False
        )  # type: ignore
        hparams = checkpoint.get("hyper_parameters", {})

        # Reconstruct model
        model_class = CLAM_MB if hparams.get("model_class") == "CLAM_MB" else CLAM_SB
        model = model_class(
            gate=hparams.get("gated", True),
            size_arg=hparams.get("size_arg", "small"),
            dropout=hparams.get("dropout", False),
            k_sample=hparams.get("k_sample", 8),
            n_classes=hparams.get("n_classes", 2),
            instance_loss_fn=SmoothTop1SVM(n_classes=2)
            if hparams.get("instance_loss_fn") == "SmoothTop1SVM"
            else nn.CrossEntropyLoss(),
            subtyping=hparams.get("subtyping", False),
            embed_dim=hparams.get("embed_dim", 1024),
            temperature=hparams.get("temperature", 1.0),
        )

        # Reconstruct optimizer
        optimizer_class = getattr(torch.optim, hparams.get("optimizer_class", "Adam"))
        optimizer = optimizer_class(
            model.parameters(), lr=hparams.get("optimizer_lr", 1e-4)
        )

        # Reconstruct loss function
        loss_slide_param = hparams.get("loss_slide", nn.CrossEntropyLoss())
        
        # Handle both string names (old checkpoints) and loss objects (new checkpoints)
        if isinstance(loss_slide_param, str):
            # Old checkpoint format - reconstruct from name
            from cellmil.utils.train.losses import FocalLoss
            if loss_slide_param == "FocalLoss":
                loss_slide = FocalLoss()
            elif loss_slide_param == "CrossEntropyLoss":
                loss_slide = nn.CrossEntropyLoss()
            else:
                # Default fallback
                loss_slide = nn.CrossEntropyLoss()
        else:
            # New checkpoint format - already a loss object
            loss_slide = loss_slide_param

        # Note: lr_scheduler is not reconstructed from checkpoint as it's typically
        # created fresh when loading a model for further training or evaluation

        lit_model = cls(
            model=model,
            optimizer=optimizer,
            loss_slide=loss_slide,
            weight_loss_slide=hparams.get("weight_loss_slide", 0.7),
            lr_scheduler=None,  # Scheduler not restored from checkpoint
            subsampling=hparams.get("subsampling", 1.0),
            use_aem=hparams.get("use_aem", False),
            aem_weight_initial=hparams.get("aem_weight_initial", 0.001),
            aem_weight_final=hparams.get("aem_weight_final", 0.0),
            aem_annealing_epochs=hparams.get("aem_annealing_epochs", 50),
        )

        lit_model.load_state_dict(
            checkpoint["state_dict"], strict=strict if strict is not None else True
        )
        return lit_model

    def _setup_metrics(self):
        metrics = torchmetrics.MetricCollection(
            {
                "accuracy": torchmetrics.Accuracy(
                    task="multiclass", num_classes=self.model.n_classes, average="none"
                ),
                "f1": torchmetrics.F1Score(
                    task="multiclass", num_classes=self.model.n_classes, average="macro"
                ),
                "precision": torchmetrics.Precision(
                    task="multiclass", num_classes=self.model.n_classes, average="macro"
                ),
                "recall": torchmetrics.Recall(
                    task="multiclass", num_classes=self.model.n_classes, average="macro"
                ),
                "auroc": torchmetrics.AUROC(
                    task="multiclass", num_classes=self.model.n_classes, average="macro"
                ),
            }
        )
        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

    def forward(
        self,
        x: torch.Tensor,
        label: torch.Tensor | None = None,
        instance_eval: bool = True,
    ):
        return self.model(x, label, instance_eval=instance_eval)

    def _shared_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], stage: str, log: bool = True
    ):
        data, label = batch

        # Ensure MIL batch size is 1
        assert data.size(0) == 1, "Batch size must be 1 for MIL"
        data = data.squeeze(0)  # [n_instances, feat_dim]

        # Apply subsampling during training
        if stage == "train" and self.subsampling != 1.0:
            # Calculate the number of samples to keep
            if 0 < self.subsampling < 1.0:
                # Treat as percentage
                num_samples = int(self.subsampling * data.shape[0])
            elif self.subsampling >= 1.0:
                # Treat as absolute count
                num_samples = min(int(self.subsampling), data.shape[0])
            else:
                raise ValueError(f"Invalid subsampling value: {self.subsampling}")

            # Generate random permutation of indices
            indices = torch.randperm(data.shape[0], device=data.device)
            # Select the first N samples from the permuted indices
            sampled_indices = indices[:num_samples]
            # Use the sampled indices to select instances
            data = data[sampled_indices]

        self.bag_size = data.size(0)

        logits, Y_prob, Y_hat, attention_weights, instance_dict = self(
            data, label=label, instance_eval=True
        )

        slide_loss = self.loss_slide(logits, label)
        instance_loss = instance_dict["instance_loss"]

        # Calculate total loss starting with slide and instance losses
        total_loss = (
            self.weight_loss_slide * slide_loss
            + (1 - self.weight_loss_slide) * instance_loss
        )

        # AEM (Attention Entropy Maximization)
        current_epoch = self.current_epoch if hasattr(self, "current_epoch") else 0
        aem: torch.Tensor | None = None
        if self.use_aem and stage == "train":
            aem = self.aem.get_aem(current_epoch, attention_weights)
            total_loss = total_loss + aem

        error = self.calculate_error(Y_hat, label)

        if log:
            self.log(
                f"{stage}/slide_loss",
                slide_loss,
                prog_bar=(stage != "train"),
                on_step=(stage == "train"),
                on_epoch=True,
                batch_size=1,
            )
            self.log(
                f"{stage}/instance_loss",
                instance_loss,
                prog_bar=(stage != "train"),
                on_step=(stage == "train"),
                on_epoch=True,
                batch_size=1,
            )
            self.log(
                f"{stage}/total_loss",
                total_loss,
                prog_bar=(stage != "train"),
                on_step=(stage == "train"),
                on_epoch=True,
                batch_size=1,
            )
            self.log(
                f"{stage}/error",
                error,
                prog_bar=(stage != "train"),
                on_step=(stage == "train"),
                on_epoch=True,
                batch_size=1,
            )

            if current_epoch == 0 and stage in ["train", "val"]:
                self.log(
                    f"{stage}/num_instances",
                    self.bag_size,
                    prog_bar=False,
                    on_step=True,
                    on_epoch=False,
                    batch_size=1,
                )

            if self.use_aem and stage == "train" and aem is not None:
                self.log(
                    f"{stage}/aem",
                    aem,
                    prog_bar=True,
                    on_step=False,
                    on_epoch=True,
                    batch_size=1,
                )

        return total_loss, Y_prob, label

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        loss, Y_prob, label = self._shared_step(batch, stage="train")

        self.train_metrics(Y_prob, label)
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        loss, Y_prob, label = self._shared_step(batch, stage="val")
        self.val_metrics(Y_prob, label)
        return loss

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        loss, Y_prob, label = self._shared_step(batch, stage="test")
        self.test_metrics(Y_prob, label)
        return loss

    def on_train_epoch_end(self) -> None:
        computed = self.train_metrics.compute()
        self._flatten_and_log_metrics(computed, prefix="train")
        self.train_metrics.reset()

    def on_validation_epoch_end(self):
        computed = self.val_metrics.compute()
        self._flatten_and_log_metrics(computed, prefix="val")
        self.val_metrics.reset()

    def on_test_epoch_end(self):
        computed = self.test_metrics.compute()
        self._flatten_and_log_metrics(computed, prefix="test")
        self.test_metrics.reset()

    def _flatten_and_log_metrics(
        self, computed: dict[str, torch.Tensor], prefix: str
    ) -> None:
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
                    vals = cast(list[torch.Tensor], val.cpu().tolist())  # type: ignore
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

    def configure_optimizers(self):  # type: ignore
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

    def predict_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Any:
        _, Y_prob, _ = self._shared_step(batch, stage="test", log=False)
        return Y_prob.argmax(dim=-1)

    @staticmethod
    def calculate_error(Y_hat: torch.Tensor, Y: torch.Tensor):
        """Classification error = 1 - accuracy."""
        return 1.0 - Y_hat.float().eq(Y.float()).float().mean().item()

    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get attention weights for a bag of instances.

        Args:
            x (torch.Tensor): Input tensor of shape [n_instances, feat_dim].
        Returns:
            torch.Tensor: Attention weights of shape [n_classes, n_instances].
        """
        self.model.eval()
        with torch.inference_mode():
            a = self.model(x, attention_only=True)  # [n_classes, n_instances]
        return a


class LitSurvCLAM(LitCLAM):
    """
    Lightning wrapper for CLAM models adapted for survival analysis.

    This class extends LitCLAM to support survival analysis tasks using
    discrete-time survival models with logistic hazard parameterization.
    Only overrides the metrics setup to use survival-specific metrics.

    Args:
        model (CLAM_MB | CLAM_SB): The CLAM model instance (SB or MB).
        optimizer (torch.optim.Optimizer): Optimizer for training.
        loss_slide (nn.Module, optional): Loss function for survival. Defaults to NegativeLogLikelihoodSurvLoss().
        weight_loss_slide (float, optional): Weight for slide-level loss. Defaults to 0.7.
        lr_scheduler (LRScheduler | None, optional): Learning rate scheduler. Defaults to None.
        use_aem (bool, optional): Whether to use AEM regularization. Defaults to False.
        aem_weight_initial (float, optional): Initial weight for AEM loss. Defaults to 0.001.
        aem_weight_final (float, optional): Final weight for AEM loss after annealing. Defaults to 0.0.
        aem_annealing_epochs (int, optional): Number of epochs to anneal AEM weight. Defaults to 50.
    """

    def __init__(
        self,
        model: CLAM_MB | CLAM_SB,
        optimizer: torch.optim.Optimizer,
        loss_slide: nn.Module = NegativeLogLikelihoodSurvLoss(),
        weight_loss_slide: float = 0.7,
        lr_scheduler: LRScheduler | None = None,
        subsampling: float = 1.0,
        use_aem: bool = False,
        aem_weight_initial: float = 0.0001,
        aem_weight_final: float = 0.0,
        aem_annealing_epochs: int = 50,
    ):
        super().__init__(
            model,
            optimizer,
            loss_slide,
            weight_loss_slide,
            lr_scheduler,
            subsampling,
            use_aem,
            aem_weight_initial,
            aem_weight_final,
            aem_annealing_epochs,
        )

        # For logistic hazard, n_classes should equal num_bins
        self.num_bins = model.n_classes

        # Override with survival-specific metrics
        self._setup_metrics()

    def _setup_metrics(self):
        """Setup C-index and Brier score metrics for survival analysis."""

        metrics = torchmetrics.MetricCollection(
            {
                "c_index": ConcordanceIndex(),
                "brier_score": BrierScore(),
            }
        )

        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

    def _shared_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], stage: str, log: bool = True
    ):
        data, label = batch

        # Ensure MIL batch size is 1
        assert data.size(0) == 1, "Batch size must be 1 for MIL"
        data = data.squeeze(0)  # [n_instances, feat_dim]

        # Apply subsampling during training
        if stage == "train" and self.subsampling != 1.0:
            # Calculate the number of samples to keep
            if 0 < self.subsampling < 1.0:
                # Treat as percentage
                num_samples = int(self.subsampling * data.shape[0])
            elif self.subsampling >= 1.0:
                # Treat as absolute count
                num_samples = min(int(self.subsampling), data.shape[0])
            else:
                raise ValueError(f"Invalid subsampling value: {self.subsampling}")

            # Generate random permutation of indices
            indices = torch.randperm(data.shape[0], device=data.device)
            # Select the first N samples from the permuted indices
            sampled_indices = indices[:num_samples]
            # Use the sampled indices to select instances
            data = data[sampled_indices]

        self.bag_size = data.size(0)

        logits, Y_prob, _, attention_weights, instance_dict = self(
            data, label=label[0], instance_eval=True
        )

        slide_loss = self.loss_slide(logits, label)
        instance_loss = instance_dict["instance_loss"]

        # Calculate total loss starting with slide and instance losses
        total_loss = (
            self.weight_loss_slide * slide_loss
            + (1 - self.weight_loss_slide) * instance_loss
        )

        # AEM (Attention Entropy Maximization)
        current_epoch = self.current_epoch if hasattr(self, "current_epoch") else 0
        aem: torch.Tensor | None = None
        if self.use_aem and stage == "train":
            aem = self.aem.get_aem(current_epoch, attention_weights)
            total_loss = total_loss + aem

        # error = self.calculate_error(Y_hat, label)

        if log:
            self.log(
                f"{stage}/slide_loss",
                slide_loss,
                prog_bar=(stage != "train"),
                on_step=(stage == "train"),
                on_epoch=True,
                batch_size=1,
            )
            self.log(
                f"{stage}/instance_loss",
                instance_loss,
                prog_bar=(stage != "train"),
                on_step=(stage == "train"),
                on_epoch=True,
                batch_size=1,
            )
            self.log(
                f"{stage}/total_loss",
                total_loss,
                prog_bar=(stage != "train"),
                on_step=(stage == "train"),
                on_epoch=True,
                batch_size=1,
            )
            # self.log(f"{stage}/error", error, prog_bar=(stage != "train"), on_step=(stage=="train"), on_epoch=True, batch_size=1)

            if current_epoch == 0 and stage in ["train", "val"]:
                self.log(
                    f"{stage}/num_instances",
                    self.bag_size,
                    prog_bar=False,
                    on_step=True,
                    on_epoch=False,
                    batch_size=1,
                )

            if self.use_aem and stage == "train" and aem is not None:
                self.log(
                    f"{stage}/aem",
                    aem,
                    prog_bar=True,
                    on_step=False,
                    on_epoch=True,
                    batch_size=1,
                )

        return total_loss, Y_prob, label

    def predict_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        """Prediction step returns logits for discrete-time hazard intervals."""
        x, _ = batch

        # Ensure MIL batch size is 1
        assert x.size(0) == 1, "Batch size must be 1 for MIL"
        x = x.squeeze(0)  # [n_instances, feat_dim]

        logits, _, _, _, _ = self.model(x, instance_eval=False)
        return logits  # Return logits, not hazards

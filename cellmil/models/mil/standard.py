# -*- coding: utf-8 -*-
# Standard Model Implementation
#
# References:
# Data-efficient and weakly supervised computational pathology on whole-slide images
# Lu, Ming Y et al., Nature Biomedical Engineering, 2021
# DOI: https://doi.org/10.1038/s41551-021-00707-9

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Literal
from .utils import LitGeneral


class MIL_fc(nn.Module):
    """
    Multiple Instance Learning model with fully connected layers for binary classification.

    This model processes a bag of instances, applies a feature extractor (FC layers),
    and performs binary classification. It selects the top k instances based on their
    probability scores for the positive class.

    Args:
        size_arg: Size configuration for the network architecture ('small' is the only option currently).
        dropout: Dropout rate for regularization.
        n_classes: Number of classes (must be 2 for binary classification).
        top_k: Number of top instances to select based on positive class probability.
        embed_dim: Dimension of the input feature embeddings.
    """

    def __init__(
        self,
        size_arg: Literal["small"] | list[int] = "small",
        dropout: float = 0.0,
        n_classes: int = 2,
        top_k: int = 1,
        embed_dim: int = 1024,
    ):
        super().__init__()  # type: ignore
        assert n_classes == 2
        self.n_classes = n_classes
        self.size_dict = {"small": [embed_dim, 512]}
        if isinstance(size_arg, list):
            if len(size_arg) != 1:
                raise ValueError("size_arg must be a list of length 1")
            size = [embed_dim, *size_arg]
        else:
            size = self.size_dict[size_arg]
        
        fc: list[nn.Module] = [
            nn.Linear(size[0], size[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
        ]
        self.fc = nn.Sequential(*fc)
        self.classifier = nn.Linear(size[1], n_classes)
        self.top_k = top_k

    def forward(
        self, h: torch.Tensor, return_features: bool = False
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]
    ]:
        """
        Forward pass of the MIL_fc model.

        Args:
            h: Input tensor of shape [n_instances, embed_dim] containing instance embeddings.
            return_features: If True, returns the feature representations of top instances.

        Returns:
            A tuple containing:
            - top_instance: Logits of the top instance(s).
            - Y_prob: Softmax probabilities for the top instance(s).
            - Y_hat: Predicted class labels for the top instance(s).
            - y_probs: Softmax probabilities for all instances.
            - results_dict: Additional results, may contain feature representations if return_features is True.
        """
        h = self.fc(h)
        logits = self.classifier(h)  # K x 2

        y_probs = F.softmax(logits, dim=1)
        top_instance_idx = torch.topk(y_probs[:, 1], self.top_k, dim=0)[1].view(
            1,
        )
        top_instance = torch.index_select(logits, dim=0, index=top_instance_idx)
        Y_hat = torch.topk(top_instance, 1, dim=1)[1]
        Y_prob = F.softmax(top_instance, dim=1)
        results_dict: dict[str, torch.Tensor] = {}

        if return_features:
            top_features = torch.index_select(h, dim=0, index=top_instance_idx)
            results_dict.update({"features": top_features})
        return top_instance, Y_prob, Y_hat, y_probs, results_dict


class MIL_fc_mc(nn.Module):
    """
    Multiple Instance Learning model with fully connected layers for multi-class classification.

    This model processes a bag of instances, applies a feature extractor (FC layers),
    and performs multi-class classification. It selects the top instance based on
    the highest probability across all classes.

    Args:
        size_arg: Size configuration for the network architecture ('small' is the only option currently).
        dropout: Dropout rate for regularization.
        n_classes: Number of classes (must be > 2 for multi-class classification).
        top_k: Number of top instances to select (must be 1 for this implementation).
        embed_dim: Dimension of the input feature embeddings.
    """

    def __init__(
        self,
        size_arg: Literal["small"] | list[int] = "small",
        dropout: float = 0.0,
        n_classes: int = 2,
        top_k: int = 1,
        embed_dim: int = 1024,
    ):
        super().__init__()  # type: ignore
        assert n_classes > 2
        self.size_dict = {"small": [embed_dim, 512]}
        if isinstance(size_arg, list):
            if len(size_arg) != 1:
                raise ValueError("size_arg must be a list of length 1")
            size = [embed_dim, *size_arg]
        else:
            size = self.size_dict[size_arg]
        fc: list[nn.Module] = [
            nn.Linear(size[0], size[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
        ]
        self.fc = nn.Sequential(*fc)
        self.classifiers = nn.Linear(size[1], n_classes)
        self.top_k = top_k
        self.n_classes = n_classes
        assert self.top_k == 1

    def forward(
        self, h: torch.Tensor, return_features: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        """
        Forward pass of the MIL_fc_mc model for multi-class classification.

        Args:
            h: Input tensor of shape [n_instances, embed_dim] containing instance embeddings.
            return_features: If True, returns the feature representations of top instances.

        Returns:
            A tuple containing:
            - top_instance: Logits of the top instance.
            - Y_prob: Softmax probabilities for the top instance.
            - Y_hat: Predicted class label for the top instance.
            - y_probs: Softmax probabilities for all instances.
            - results_dict: Additional results, may contain feature representations if return_features is True.
        """
        h = self.fc(h)
        logits = self.classifiers(h)

        y_probs = F.softmax(logits, dim=1)
        m = y_probs.view(1, -1).argmax(1)
        top_indices = torch.cat(
            ((m // self.n_classes).view(-1, 1), (m % self.n_classes).view(-1, 1)), dim=1
        ).view(-1, 1)
        top_instance = logits[top_indices[0]]

        Y_hat = top_indices[1]
        Y_prob = y_probs[top_indices[0]]

        results_dict: dict[str, torch.Tensor] = {}

        if return_features:
            top_features = torch.index_select(h, dim=0, index=top_indices[0])
            results_dict.update({"features": top_features})
        return top_instance, Y_prob, Y_hat, y_probs, results_dict

class LitStandard(LitGeneral):
    def forward(self, x: torch.Tensor):
        return self.model(x)[0]

    def _shared_step(
        self, 
        batch: tuple[torch.Tensor, torch.Tensor],
        stage: str, 
        log: bool = True
    ):
        x, y = batch

        # Ensure MIL batch size is 1
        assert x.size(0) == 1, "Batch size must be 1 for MIL"
        x = x.squeeze(0)   # [n_instances, feat_dim]

        logits = self(x)
        loss = self.loss(logits, y)

        if log:
            self.log(f"{stage}/loss", loss, prog_bar=(stage != "train"), on_step=(stage=="train"), on_epoch=True)

        return loss, logits, y
        

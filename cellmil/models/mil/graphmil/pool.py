import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any
from topk.svm import SmoothTop1SVM  # type: ignore
from torch_geometric.nn import global_mean_pool  # type: ignore
from ..clam import CLAM_SB, CLAM_MB
from ..standard import MIL_fc, MIL_fc_mc
from ..attentiondeepmil import AttentionDeepMIL
from abc import ABC, abstractmethod

from cellmil.utils import logger


class GlobalPooling_Classifier(nn.Module, ABC):
    """Abstract base class for global pooling classifiers in GraphMIL.
    
    This class defines the interface for pooling classifiers that aggregate
    node features into graph-level predictions. Each subclass handles its
    own specific arguments and validates them appropriately.
    """
    def __init__(
        self,
        input_dim: int,
        dropout: float,
        n_classes: int,
        size_arg: list[int] | str,
        **kwargs: Any,
    ):
        super().__init__() # type: ignore
        self.input_dim = input_dim
        self.dropout = dropout
        self.kwargs = kwargs
        self.n_classes = n_classes
        self.size_arg = size_arg

    def get_hyperparameters(self) -> dict[str, Any]:
        """Get all hyperparameters for this pooling classifier."""
        return {
            "type": self.__class__.__name__,
            "input_dim": self.input_dim,
            "dropout": self.dropout,
            "n_classes": self.n_classes,
            "size_arg": self.size_arg,
            **self.kwargs,
            **self._get_specific_hyperparameters()
        }
    
    def _get_specific_hyperparameters(self) -> dict[str, Any]:
        """Override in subclasses to add specific hyperparameters."""
        return {}

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        batch: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Forward pass for the pooling classifier.
        
        Args:
            x: Node features tensor of shape (num_nodes, input_dim)
            batch: Batch assignment tensor for global pooling (if applicable)
            **kwargs: Additional arguments specific to each pooling classifier
            
        Returns:
            tuple containing:
                - logits: Raw model outputs of shape (1, n_classes)
                - output_dict: Dictionary containing instance-level information
        """
        pass

    def get_attention_weights(
        self, 
        x: torch.Tensor, 
        batch: torch.Tensor | None = None
    ) -> torch.Tensor | None:
        """
        Extract attention weights from the pooling classifier.
        
        Args:
            x: Node features tensor
            batch: Batch assignment tensor (if applicable)
            
        Returns:
            Attention weights tensor or None if not available
        """
        return None


class Mean_MLP(GlobalPooling_Classifier):
    """Mean pooling followed by MLP classifier."""
    
    def __init__(
        self,
        input_dim: int,
        dropout: float,
        n_classes: int,
        size_arg: list[int],
        **kwargs: Any,
    ):
        super().__init__(
            input_dim=input_dim,
            dropout=dropout,
            n_classes=n_classes,
            size_arg=size_arg,
            **kwargs,
        )
        
        if isinstance(self.size_arg, str):
            raise ValueError("Mean_MLP requires size_arg to be a list of integers")

        layers: list[nn.Module] = []
        prev_dim = input_dim
        if len(self.size_arg) > 0:
            for hidden_dim in self.size_arg:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(nn.ReLU())
                prev_dim = hidden_dim
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(prev_dim, n_classes))
        self.classifier = nn.Sequential(*layers)

    def forward(
        self,
        x: torch.Tensor,
        batch: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Forward pass for Mean_MLP pooling classifier.
        
        Args:
            x: Node features tensor of shape (num_nodes, input_dim)
            batch: Batch assignment tensor for global pooling
            **kwargs: Additional arguments (ignored by Mean_MLP)
            
        Returns:
            tuple containing:
                - logits: Raw model outputs of shape (1, n_classes)
                - output_dict: Dictionary containing instance-level information
        """
        # Use global_mean_pool with batch assignment if provided, otherwise assume single graph
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        x = global_mean_pool(x, batch)
        logits = self.classifier(x)
        Y_prob = (
            torch.sigmoid(logits) if self.n_classes == 1 else F.softmax(logits, dim=1)
        )
        Y_hat = Y_prob.argmax(dim=1, keepdim=True)
        output_dict: dict[str, Any] = {
            "y_hat": Y_hat,
            "y_prob": Y_prob,
        }
        return logits, output_dict


class CLAM(GlobalPooling_Classifier):
    """CLAM pooling classifier with attention-based multiple instance learning."""
    
    def __init__(
        self,
        input_dim: int,
        dropout: float,
        n_classes: int,
        size_arg: list[int] | str,
        gate: bool = True,
        k_sample: int = 8,
        instance_loss_fn: nn.Module = SmoothTop1SVM(n_classes=2).cuda() if torch.cuda.is_available() else SmoothTop1SVM(n_classes=2),
        subtyping: bool = False,
        clam_type: str = "SB",
        temperature: float = 1.0,
        **kwargs: Any,
    ):
        super().__init__(
            input_dim=input_dim,
            dropout=dropout,
            n_classes=n_classes,
            size_arg=size_arg,
            **kwargs
        )
            
        self.gate = gate
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.subtyping = subtyping
        self.clam_type = clam_type
        self.temperature = temperature

        # Validate size_arg
        if isinstance(self.size_arg, str):
            if self.size_arg not in ["small", "big"]:
                raise ValueError("size_arg must be 'small' or 'big' or a list")
        else:
            if len(self.size_arg) == 0 or len(self.size_arg) > 2:
                raise ValueError(
                    "size_arg list must not be empty and must have at most 2 elements"
                )

        # Validate CLAM type
        if self.clam_type not in ["SB", "MB"]:
            raise ValueError(f"Unknown CLAM type: {self.clam_type}")

        # Create the appropriate CLAM network
        if self.clam_type == "SB":
            self.net = CLAM_SB(
                self.gate,
                self.size_arg,  # type: ignore
                self.dropout > 0,
                self.k_sample,
                self.n_classes,
                self.instance_loss_fn,
                self.subtyping,
                self.input_dim,
                self.temperature,
            )
        elif self.clam_type == "MB":
            self.net = CLAM_MB(
                self.gate,
                self.size_arg,  # type: ignore
                self.dropout > 0,
                self.k_sample,
                self.n_classes,
                self.instance_loss_fn,
                self.subtyping,
                self.input_dim,
                self.temperature,
            )

    def _get_specific_hyperparameters(self) -> dict[str, Any]:
        return {
            "gate": self.gate,
            "k_sample": self.k_sample,
            "instance_loss_fn": self.instance_loss_fn.__class__.__name__,
            "subtyping": self.subtyping,
            "clam_type": self.clam_type,
            "temperature": self.temperature
        }

    def forward(
        self,
        x: torch.Tensor,
        batch: torch.Tensor | None = None,
        label: torch.Tensor | None = None,
        instance_eval: bool = False,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Forward pass for CLAM pooling classifier.
        
        Args:
            x: Node features tensor of shape (num_nodes, input_dim)
            batch: Batch assignment tensor (not used by CLAM, must be single graph)
            label: Ground truth labels for instance evaluation
            instance_eval: Whether to perform instance-level evaluation
            **kwargs: Additional arguments (ignored by CLAM)
            
        Returns:
            tuple containing:
                - logits: Raw model outputs of shape (1, n_classes)
                - output_dict: Dictionary containing instance-level information
        """
        # CLAM doesn't use batch assignment since it handles MIL internally
        # For batch_size > 1, we'd need to handle this differently, but we assert batch_size=1
        if batch is not None and batch.max().item() > 0:
            raise ValueError(
                "CLAM pooling classifier requires batch_size=1. Found multiple graphs in batch."
            )
        logits, Y_prob, Y_hat, a, instance_dict = self.net(x, label, instance_eval)
        output_dict: dict[str, Any] = {
            "y_prob": Y_prob,
            "y_hat": Y_hat,
            "attention": a, 
            **instance_dict,
        }
        return logits, output_dict

    def get_attention_weights(
        self, 
        x: torch.Tensor, 
        batch: torch.Tensor | None = None
    ) -> torch.Tensor | None:
        """
        Extract attention weights from CLAM.
        
        Args:
            x: Node features tensor
            batch: Batch assignment tensor (should be single graph)
            
        Returns:
            Attention weights tensor of shape [1, num_nodes] or [num_classes, num_nodes]
        """
        try:
            result = self.net.forward(x, attention_only=True, instance_eval=False)  # type: ignore
            if isinstance(result, torch.Tensor):
                return result
        except Exception:
            pass
            
        return None


class Standard(GlobalPooling_Classifier):
    """Standard MIL pooling classifier."""
    
    def __init__(
        self,
        input_dim: int,
        dropout: float,
        n_classes: int,
        size_arg: list[int] | str = "small",
        standard_type: str = "fc",
        **kwargs: Any,
    ):
        super().__init__(
            input_dim=input_dim,
            dropout=dropout,
            n_classes=n_classes,
            size_arg=size_arg,
            **kwargs
        )

        self.standard_type = standard_type

        # Validate size_arg
        if isinstance(self.size_arg, str):
            if self.size_arg not in ["small"]:
                raise ValueError("size_arg must be 'small' or 'big' or a list")
        else:
            if len(self.size_arg) == 0 or len(self.size_arg) > 2:
                raise ValueError(
                    "size_arg list must not be empty and must have at most 2 elements"
                )

        # Validate standard_type
        if self.standard_type not in ["fc", "fc_mc"]:
            raise ValueError(f"Unknown Standard type: {self.standard_type}")
            
        # Create the appropriate Standard network
        if self.standard_type == "fc":
            self.net = MIL_fc(
                size_arg=self.size_arg,  # type: ignore
                dropout=self.dropout,
                n_classes=self.n_classes,
                embed_dim=self.input_dim,
            )
        elif self.standard_type == "fc_mc":
            self.net = MIL_fc_mc(
                size_arg=self.size_arg,  # type: ignore
                dropout=self.dropout,
                n_classes=self.n_classes,
                embed_dim=self.input_dim,
            )

    def _get_specific_hyperparameters(self) -> dict[str, Any]:
        return {"standard_type": self.standard_type}

    def forward(
        self,
        x: torch.Tensor,
        batch: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Forward pass for Standard MIL pooling classifier.
        
        Args:
            x: Node features tensor of shape (num_nodes, input_dim)
            batch: Batch assignment tensor (not used by Standard, must be single graph)
            **kwargs: Additional arguments (ignored by Standard)
            
        Returns:
            tuple containing:
                - logits: Raw model outputs of shape (1, n_classes)
                - output_dict: Dictionary containing instance-level information
        """
        # For batch_size > 1, we'd need to handle this differently, but we assert batch_size=1
        if batch is not None and batch.max().item() > 0:
            raise ValueError(
                "Standard pooling classifier requires batch_size=1. Found multiple graphs in batch."
            )
        top_instance, Y_prob, Y_hat, y_probs, results_dict = self.net(x)
        
        # Add consistent attention fields (Standard doesn't use attention)
        output_dict: dict[str, Any] = {
            "y_prob": Y_prob,
            "y_hat": Y_hat,
            "y_probs": y_probs,
            **results_dict,
        }
        
        return top_instance, output_dict

class Attention(GlobalPooling_Classifier):
    """AttentionDeepMIL pooling classifier."""
    
    def __init__(
        self,
        input_dim: int,
        dropout: float,
        n_classes: int,
        size_arg: list[int],
        attention_branches: int = 1,
        temperature: float = 1.0,
        **kwargs: Any,
    ):
        super().__init__(
            input_dim=input_dim,
            dropout=dropout,
            n_classes=n_classes,
            size_arg=size_arg,
            **kwargs
        )
        
        if isinstance(self.size_arg, str):
            raise ValueError("AttentionDeepMIL requires size_arg to be a list of integers")
        
        if len(size_arg) != 2:
            raise ValueError("size_arg must be a list with exactly 2 elements for AttentionDeepMIL")
            
        if attention_branches < 1:
            raise ValueError("attention_branches must be a positive integer")
        
        self.attention_branches = attention_branches
        self.temperature = temperature

        self.net = AttentionDeepMIL(
            size_arg=self.size_arg,
            n_classes=self.n_classes,
            embed_dim=self.input_dim,
            attention_branches=self.attention_branches,
            temperature=self.temperature,
            dropout=self.dropout
        )

    def _get_specific_hyperparameters(self) -> dict[str, Any]:
        return {
            "attention_branches": self.attention_branches,
            "temperature": self.temperature
        }
        
    def forward(
        self,
        x: torch.Tensor,
        batch: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Forward pass for AttentionDeepMIL pooling classifier.
        
        Args:
            x: Node features tensor of shape (num_nodes, input_dim)
            batch: Batch assignment tensor (not used by AttentionDeepMIL, must be single graph)
            **kwargs: Additional arguments (ignored by AttentionDeepMIL)
            
        Returns:
            tuple containing:
                - logits: Raw model outputs of shape (1, n_classes)
                - output_dict: Dictionary containing instance-level information and attention weights
        """
        # For batch_size > 1, we'd need to handle this differently, but we assert batch_size=1
        if batch is not None and batch.max().item() > 0:
            raise ValueError(
                "AttentionDeepMIL pooling classifier requires batch_size=1. Found multiple graphs in batch."
            )
        logits, output_dict = self.net(x)
        
        return logits, output_dict

    def get_attention_weights(
        self, 
        x: torch.Tensor, 
        batch: torch.Tensor | None = None
    ) -> torch.Tensor | None:
        """
        Extract attention weights from AttentionDeepMIL.
        
        Args:
            x: Node features tensor
            batch: Batch assignment tensor (must be single graph)
            
        Returns:
            Attention weights tensor of shape [attention_branches, num_nodes]
        """
        if batch is not None and batch.max().item() > 0:
            raise ValueError(
                "AttentionDeepMIL requires batch_size=1. Found multiple graphs in batch."
            )
        
        try:
            logits, output_dict = self.net(x)
            
            logger.info(f"Extracted logits: {logits}")
            
            return output_dict.get('attention')
        except Exception:
            return None
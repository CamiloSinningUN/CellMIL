import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List, Optional, cast, Literal


class FocalLoss(nn.Module):
    def __init__(
        self,
        alpha: Optional[Union[float, List[float], torch.Tensor]] = None,
        gamma: float = 2.0,
        label_smoothing: float = 0.0,
    ):
        super().__init__()  # type: ignore
        self.gamma = gamma
        self.label_smoothing = label_smoothing

        # Handle alpha parameter for multi-class support
        if alpha is None:
            self.alpha = None
        elif isinstance(alpha, (float, int)):
            self.alpha = alpha
        else:
            # Convert list or tensor to tensor and register as buffer
            self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # inputs: [batch_size, num_classes] (logits)
        # targets: [batch_size] (class indices) or [batch_size, num_classes] (soft labels)

        # Check if targets are soft labels (2D) or hard labels (1D)
        if targets.dim() == 2:
            # Soft labels - get hard class indices for alpha weighting
            target_indices = targets.argmax(dim=1)
        else:
            # Hard labels - convert to long for indexing
            target_indices = targets.long()

        # Compute cross entropy loss (supports both soft and hard labels)
        CE_loss = F.cross_entropy(inputs, targets, reduction="none", label_smoothing=self.label_smoothing)

        # Compute probabilities
        pt = torch.exp(-CE_loss)

        # Handle alpha weighting
        if self.alpha is None:
            # No class weighting
            alpha_t = 1.0
        elif isinstance(self.alpha, float):
            # Binary classification case (backward compatibility)
            if inputs.size(1) == 2:
                alpha_t = self.alpha * target_indices.float() + (1 - self.alpha) * (1 - target_indices.float())
            else:
                # Multi-class with single alpha value (uniform weighting)
                alpha_t = self.alpha
        else:
            # Multi-class with per-class alpha values
            alpha_t = cast(torch.Tensor, self.alpha[target_indices])  # type: ignore

        loss = alpha_t * (1 - pt) ** self.gamma * CE_loss

        return loss.mean()
    
class NegativeLogLikelihoodSurvLoss(nn.Module):
    """
    Negative Log-Likelihood Loss for Discrete-Time Survival Analysis using Logistic Hazard.

    This loss function is designed for survival analysis tasks where the model predicts
    the hazard probabilities for discrete time intervals. It computes the negative log-likelihood
    based on the predicted hazards and the observed survival times and event indicators.

    Args:
        alpha (Optional[float]): Weighting factor for the loss. Default is None.
        epsilon (float): Small value to avoid log(0). Default is 1e-8.
        reduction (str): Specifies the reduction to apply to the output: 'mean' | 'sum'. Default is 'sum'.
    """
    def __init__(
        self,
        alpha: Optional[float] = None,
        epsilon: float = 1e-8,
        reduction: Literal["sum", "mean"] = "sum",
    ):
        super().__init__()  # type: ignore
        self.alpha = alpha
        self.epsilon = epsilon
        self.reduction = reduction
        
    def __call__(self, inputs: torch.Tensor, target: tuple[torch.Tensor, torch.Tensor] | tuple[int, int]) -> torch.Tensor:
        """
        Compute the Negative Log-Likelihood Loss for Discrete-Time Survival Analysis.

        Args:
            inputs (torch.Tensor): Predicted hazard probabilities for each time bin. Shape: [batch_size, n_bins]
            target: Either:
                - tuple[torch.Tensor, torch.Tensor]: (durations, events) where both are tensors of shape [batch_size]
                - tuple[int, int]: (duration, event) for single sample (will be converted to tensors)
            
        Returns:
            torch.Tensor: Computed loss value.
        """
        # Handle both tensor and int inputs
        if isinstance(target[0], int):
            durations = torch.tensor([target[0]], dtype=torch.int64, device=inputs.device)
            events = torch.tensor([target[1]], dtype=torch.int64, device=inputs.device)
        else:
            durations = target[0]
            events = target[1]
            
        assert isinstance(durations, torch.Tensor)
        assert isinstance(events, torch.Tensor)
        assert inputs.dim() == 2, "Inputs should be of shape [batch_size, n_bins]"
        assert durations.dim() == 1, "Durations should be of shape [batch_size]"
        assert events.dim() == 1, "Events should be of shape [batch_size]"
        assert inputs.size(0) == durations.size(0) == events.size(0) == 1, "Batch size must be 1"
        
        durations = durations.type(torch.int64).unsqueeze(1)
        events = events.type(torch.int64).unsqueeze(1)
        
        hazards = torch.sigmoid(inputs)
        
        survival = torch.cumprod(1 - hazards, dim=1)
        survival = torch.cat([torch.ones_like(events), survival], dim=1)  # Add S(0) = 1 at the beginning
        
        survival_prev = torch.gather(survival, dim=1, index=durations).clamp(min=self.epsilon)
        hazard_at_event = torch.gather(hazards, dim=1, index=durations).clamp(min=self.epsilon)
        survival_at_event = torch.gather(survival, dim=1, index=durations + 1).clamp(min=self.epsilon)
        
        uncensored_loss = - events * (torch.log(survival_prev) + torch.log(hazard_at_event))
        censored_loss = - (1 - events)  * torch.log(survival_at_event)
        
        negative_log_likelihood = uncensored_loss + censored_loss
        
        if self.alpha is not None:
            negative_log_likelihood = (1 - self.alpha) * negative_log_likelihood + self.alpha * uncensored_loss
            
        if self.reduction == "sum":
            return negative_log_likelihood.sum()
        elif self.reduction == "mean":
            return negative_log_likelihood.mean()
        else:
            raise ValueError(f"Invalid reduction type: {self.reduction}. Supported types: 'sum', 'mean'.")
        
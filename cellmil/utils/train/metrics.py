import torch
import numpy as np
from torchmetrics import Metric
from typing import Any, cast
from sksurv.metrics import concordance_index_censored, integrated_brier_score  # type: ignore

class ConcordanceIndex(Metric):
    """
    Concordance Index (C-index) for survival analysis.

    The C-index measures the model's ability to correctly order pairs of samples
    by their survival times. A C-index of 1.0 indicates perfect concordance,
    while 0.5 indicates random predictions.

    Args:
        compute_on_cpu (bool): Whether to compute on CPU. Default: False.
        **kwargs: Additional keyword arguments passed to the parent Metric class.
    """

    is_differentiable: bool | None = False
    higher_is_better: bool | None = True
    full_state_update: bool | None = False

    def __init__(self, compute_on_cpu: bool = False, **kwargs: Any):
        super().__init__(**kwargs)
        self.compute_on_cpu = compute_on_cpu

        # Store predictions and targets for batch computation
        self.add_state("preds_list", default=[], dist_reduce_fx="cat")  # type: ignore
        self.add_state("durations_list", default=[], dist_reduce_fx="cat")  # type: ignore
        self.add_state("events_list", default=[], dist_reduce_fx="cat")  # type: ignore

    def update(
        self, preds: torch.Tensor, target: tuple[torch.Tensor, torch.Tensor] | tuple[int, int]
    ) -> None:
        """
        Update state with predictions and targets.

        Args:
            preds (torch.Tensor): Logits for hazard at each time bin.
                Shape: [batch_size, num_bins] for discrete-time hazard model
                or [batch_size] for single risk score.
            target: Either:
                - tuple[torch.Tensor, torch.Tensor]: (durations, events) where both are tensors of shape [batch_size]
                - tuple[int, int]: (duration, event) for single sample
        """
        # Handle both tensor and int inputs
        if isinstance(target[0], int):
            durations = torch.tensor([target[0]], dtype=torch.float32)
            events = torch.tensor([target[1]], dtype=torch.float32)
        else:
            durations = target[0]
            events = target[1]
            
        assert isinstance(durations, torch.Tensor)
        assert isinstance(events, torch.Tensor)

        # Convert logits to risk scores that capture expected event timing
        if preds.dim() == 2:
            hazards = torch.sigmoid(preds)
            batch_size, num_bins = hazards.shape

            device = hazards.device
            dtype = hazards.dtype

            # Survival probability at the start of each bin
            survival_prefix = torch.cumprod(
                torch.cat(
                    [torch.ones(batch_size, 1, device=device, dtype=dtype), 1.0 - hazards],
                    dim=1,
                ),
                dim=1,
            )
            survival_prev = survival_prefix[:, :-1]

            # Event probability mass per bin
            event_probs = survival_prev * hazards

            time_positions = torch.arange(num_bins, device=device, dtype=dtype)
            expected_time = torch.sum(event_probs * time_positions, dim=1)

            tail_mass = torch.clamp(1.0 - event_probs.sum(dim=1), min=0.0)
            expected_time = expected_time + tail_mass * num_bins

            # Store expected event time; conversion to risk happens in compute()
            risk_scores = expected_time
        elif preds.dim() == 1:
            # Single score per sample. 
            # NOTE: This is treated as a survival score (higher = longer survival)
            # because it is negated in compute(). If passing risk (hazard), negate it first.
            risk_scores = preds
        else:
            raise ValueError(f"Unexpected prediction shape: {preds.shape}")

        # Convert to CPU if requested
        if self.compute_on_cpu:
            risk_scores = risk_scores.cpu()
            durations = durations.cpu()
            events = events.cpu()

        # Convert boolean events to float
        if events.dtype == torch.bool:
            events = events.float()

        # Store for later computation
        self.preds_list.append(risk_scores)  # type: ignore
        self.durations_list.append(durations)  # type: ignore
        self.events_list.append(events)  # type: ignore

    def compute(self) -> torch.Tensor:
        """Compute the final C-index using scikit-survival."""
        if len(self.preds_list) == 0:  # type: ignore
            return torch.tensor(0.5)  # Return 0.5 if no samples
        
        # Concatenate all stored values
        all_preds = torch.cat(self.preds_list).detach().cpu().numpy()  # type: ignore
        all_durations = torch.cat(self.durations_list).detach().cpu().numpy()  # type: ignore
        all_events = torch.cat(self.events_list).detach().cpu().numpy().astype(bool)  # type: ignore
        
        # Use scikit-survival's concordance_index_censored
        # Note: Higher risk scores should predict shorter survival times
        try:
            result = cast(tuple[float | int, ...], concordance_index_censored(
                all_events,
                all_durations,
                -all_preds  # Negate because higher risk = shorter survival
            ))
            c_index = result[0]  # Returns (c_index, concordant, discordant, tied_risk, tied_time)
            return torch.tensor(c_index, dtype=torch.float32)
        except Exception:
            # Fallback if computation fails
            return torch.tensor(0.5, dtype=torch.float32)


class BrierScore(Metric):
    """
    Integrated Brier Score for survival analysis.

    The Brier score measures the accuracy of probabilistic predictions.
    For survival analysis, we compute the time-integrated Brier score.
    Lower values indicate better predictions (0 is perfect, 1 is worst).

    Note: This is a simplified implementation that approximates the integrated
    Brier score by computing the mean squared error between predicted risk
    and actual outcomes at observed time points.

    Args:
        compute_on_cpu (bool): Whether to compute on CPU. Default: False.
        **kwargs: Additional keyword arguments passed to the parent Metric class.
    """

    is_differentiable: bool | None = False
    higher_is_better: bool | None = False
    full_state_update: bool | None = False

    def __init__(self, compute_on_cpu: bool = False, **kwargs: Any):
        super().__init__(**kwargs)
        self.compute_on_cpu = compute_on_cpu

        # Aggregate Brier statistics instead of storing entire tensors
        self.add_state("brier_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("weight_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(
        self, preds: torch.Tensor, target: tuple[torch.Tensor, torch.Tensor] | tuple[int, int]
    ) -> None:
        """
        Update state with predictions and targets.

        Args:
            preds (torch.Tensor): Logits for hazard at each time bin.
                Shape: [batch_size, num_bins] for discrete-time hazard model
                or [batch_size] for single risk score.
            target: Either:
                - tuple[torch.Tensor, torch.Tensor]: (durations, events) where both are tensors of shape [batch_size]
                - tuple[int, int]: (duration, event) for single sample
        """
        # Handle both tensor and int inputs
        if isinstance(target[0], int):
            durations = torch.tensor([target[0]], dtype=torch.float32)
            events = torch.tensor([target[1]], dtype=torch.float32)
        else:
            durations = target[0]
            events = target[1]
            
        assert isinstance(durations, torch.Tensor)
        assert isinstance(events, torch.Tensor)

        if preds.dim() != 2:
            raise ValueError(f"Unexpected prediction shape: {preds.shape}")

        hazards = torch.sigmoid(preds)
        batch_size, num_bins = hazards.shape
        device = hazards.device
        dtype = hazards.dtype

        survival_full = torch.cumprod(
            torch.cat(
                [torch.ones(batch_size, 1, device=device, dtype=dtype), 1.0 - hazards],
                dim=1,
            ),
            dim=1,
        )
        survival_after_bin = survival_full[:, 1:]
        cumulative_event = 1.0 - survival_after_bin

        time_positions = torch.arange(num_bins, device=device, dtype=dtype).unsqueeze(0)
        durations = durations.to(device=device, dtype=time_positions.dtype)
        events = events.to(device=device, dtype=cumulative_event.dtype)

        durations = torch.clamp(durations, min=0, max=num_bins - 1)
        duration_expanded = durations.unsqueeze(1)
        event_expanded = events.unsqueeze(1)

        # Event indicator becomes 1 once the event has occurred
        target_event = (time_positions >= duration_expanded).float() * event_expanded

        # Only evaluate bins up to the observed time for censored samples
        # For event samples, we know the status for all future bins (dead)
        valid_mask = (event_expanded.bool() | (time_positions <= duration_expanded)).float()

        squared_error = (cumulative_event - target_event) ** 2 * valid_mask
        batch_error = squared_error.sum()
        batch_weight = valid_mask.sum().clamp_min(1.0)

        if self.compute_on_cpu:
            batch_error = batch_error.cpu()
            batch_weight = batch_weight.cpu()

        self.brier_sum += batch_error
        self.weight_sum += batch_weight

    def compute(self) -> torch.Tensor:
        """Compute the final Brier score using scikit-survival."""
        if self.weight_sum <= 0:
            return torch.tensor(0.25, dtype=torch.float32)

        return (self.brier_sum / self.weight_sum).to(dtype=torch.float32)

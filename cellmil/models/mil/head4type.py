from typing_extensions import Self
import torch
import torchmetrics
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LRScheduler
from .utils import LitGeneral, AEM
from typing import IO, Any, Callable, Literal
from pathlib import Path
from cellmil.utils.train.losses import NegativeLogLikelihoodSurvLoss
from cellmil.utils.train.metrics import ConcordanceIndex, BrierScore


class Head4Type(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        n_classes: int = 2,
        size_arg: list[int] = [512, 128],
        temperature: float = 1.0,
        cell_types: int = 5,
        heads_aggregation: Literal[
            "weighted_mean", "attention", "mean", "concatenation", "custom"
        ] = "custom",
        dropout: float = 0.0,
        custom_aggregation_weights: list[float] | None = [3.0, 2.0, 1.0, 0.0, 0.0],
    ):
        super().__init__()  # type: ignore
        self.size_arg = size_arg
        self.embed_dim = embed_dim
        self.temperature = temperature
        self.n_classes = n_classes
        self.cell_types = cell_types
        self.heads_aggregation = heads_aggregation
        self.dropout = dropout

        if heads_aggregation not in [
            "weighted_mean",
            "attention",
            "mean",
            "concatenation",
            "custom",
        ]:
            raise ValueError(
                f"heads_aggregation must be one of ['weighted_mean', 'attention', 'mean', 'concatenation', 'custom'], got '{heads_aggregation}'"
            )

        # Validate custom weights if using custom aggregation
        if heads_aggregation == "custom":
            if custom_aggregation_weights is None:
                raise ValueError(
                    "custom_aggregation_weights must be provided when heads_aggregation is 'custom'"
                )
            if len(custom_aggregation_weights) != cell_types:
                raise ValueError(
                    f"custom_aggregation_weights must have length {cell_types}, got {len(custom_aggregation_weights)}"
                )
            # Normalize weights to sum to 1
            total = sum(custom_aggregation_weights)
            self.custom_weights = torch.tensor(
                [w / total for w in custom_aggregation_weights], dtype=torch.float32
            )
        else:
            self.custom_weights = None

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(self.embed_dim, self.size_arg[0]),
            nn.ReLU(),
            nn.Dropout(self.dropout),
        )

        self.attention = nn.Sequential(
            nn.Linear(self.size_arg[0], self.size_arg[1]),  # matrix V
            nn.Tanh(),
            nn.Dropout(self.dropout),
            nn.Linear(self.size_arg[1], self.cell_types),
        )

        # Classifier input size depends on aggregation mode
        classifier_input_size = (
            self.size_arg[0] * self.cell_types
            if heads_aggregation == "concatenation"
            else self.size_arg[0]
        )
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_size, self.n_classes)
        )

        if self.heads_aggregation == "attention":
            self.aggregation_attention = nn.Sequential(
                nn.Linear(self.size_arg[0], self.size_arg[1]),
                nn.Tanh(),
                nn.Dropout(self.dropout),
                nn.Linear(self.size_arg[1], 1),
            )

    def forward(
        self,
        x: torch.Tensor,  # NxD
        cell_types: torch.Tensor,  # NxC
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if len(x.shape) != 2:
            raise ValueError("Input tensor must be 2D (KxD)")

        if x.shape[0] != cell_types.shape[0]:
            raise ValueError(
                "Input tensor and cell_types must have the same first dimension (K)"
            )

        if cell_types.shape[-1] != self.cell_types:
            raise ValueError(
                f"cell_types tensor must have last dimension of size {self.cell_types}"
            )

        h = self.feature_extractor_part2(x)  # NxM

        a = self.attention(h)  # NxATTENTION_BRANCHES
        a = torch.transpose(a, 1, 0)  # ATTENTION_BRANCHESxN

        # Mask attention scores: set to -inf where cell type doesn't match the branch
        # cell_types is NxC with one-hot or probability distribution
        # We want branch i to only attend to cells of type i
        cell_type_mask = torch.transpose(cell_types, 1, 0)  # CxN (same shape as a)
        # Set attention to -inf where mask is 0 (before softmax)
        a = a.masked_fill(cell_type_mask == 0, float("-inf"))

        a = F.softmax(a / self.temperature, dim=1)  # softmax over N

        # Replace any NaN values with 0
        a = torch.where(torch.isnan(a), torch.zeros_like(a), a)

        m = torch.mm(a, h)  # ATTENTION_BRANCHESxM

        # Aggregate branch representations
        if self.heads_aggregation == "weighted_mean":
            # Weighted average over branches based on cell type proportions
            # Count cells of each type: sum over N dimension of cell_types
            cell_type_counts = torch.sum(cell_types, dim=0)  # C
            # Normalize to get proportions
            cell_type_proportions = cell_type_counts / torch.sum(cell_type_counts)  # C
            # Weight each branch representation by its cell type proportion
            weighted_m = cell_type_proportions.unsqueeze(1) * m  # CxM
            # Sum over branches to get final representation
            aggregated_m = torch.sum(weighted_m, dim=0, keepdim=True)  # 1xM

        elif self.heads_aggregation == "attention":
            # Use attention mechanism to aggregate branches
            # m is CxM, we want to learn which branches are more important
            agg_scores = self.aggregation_attention(m)  # Cx1
            agg_weights = F.softmax(agg_scores, dim=0)  # Cx1, weights sum to 1
            # Weighted sum of branch representations
            aggregated_m = torch.sum(agg_weights * m, dim=0, keepdim=True)  # 1xM

        elif self.heads_aggregation == "mean":
            # Simple average over all branches
            aggregated_m = torch.mean(m, dim=0, keepdim=True)  # 1xM

        elif self.heads_aggregation == "custom" and self.custom_weights is not None:
            # Use custom weights to aggregate branches
            # Move custom weights to the same device as m
            custom_weights = self.custom_weights.to(m.device)  # C

            # Dynamically normalize weights based on present cell types
            # Check which cell types are present (non-zero rows in m)
            # A cell type is present if its representation is not all zeros
            present_mask = torch.any(m != 0, dim=1)  # C (boolean mask)

            # Zero out weights for absent cell types
            adjusted_weights = custom_weights * present_mask.float()  # C

            # Renormalize so that weights of present cell types sum to 1
            weight_sum = torch.sum(adjusted_weights)
            if weight_sum > 0:
                adjusted_weights = adjusted_weights / weight_sum
            # If no cell types are present (edge case), weights remain zeros

            # Weight each branch representation by adjusted custom weights
            weighted_m = adjusted_weights.unsqueeze(1) * m  # CxM
            # Sum over branches to get final representation
            aggregated_m = torch.sum(weighted_m, dim=0, keepdim=True)  # 1xM

        else:  # self.heads_aggregation == "concatenation"
            # Concatenate all branch representations
            aggregated_m = m.flatten().unsqueeze(0)  # 1x(C*M)

        logits = self.classifier(aggregated_m)  # n_classes

        return logits, {"attention": a, "features": h, "m": m}


class LitHead4Type(LitGeneral):
    """
    Lightning wrapper for Head4Type model.

    This class extends the base LitGeneral class to provide Lightning-specific functionality
    for the Ours model.

    Args:
        model (nn.Module): The Ours model instance.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        loss (nn.Module, optional): Loss function. Defaults to nn.CrossEntropyLoss().
        lr_scheduler (LRScheduler | None, optional): Learning rate scheduler. Defaults to None.
        use_aem (bool, optional): Whether to use AEM regularization. Defaults to False.
        aem_weight_initial (float, optional): Initial weight for AEM loss. Defaults to 0.001.
        aem_weight_final (float, optional): Final weight for AEM loss after annealing. Defaults to 0.0.
        aem_annealing_epochs (int, optional): Number of epochs to anneal AEM weight. Defaults to 50.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss: nn.Module = nn.CrossEntropyLoss(),
        lr_scheduler: LRScheduler | None = None,
        subsampling: float = 0.8,
        use_aem: bool = False,
        aem_weight_initial: float = 0.0001,
        aem_weight_final: float = 0.0,
        aem_annealing_epochs: int = 25,
    ) -> None:
        super().__init__(model, optimizer, loss, lr_scheduler)
        self.n_classes = model.n_classes
        self.subsampling = subsampling
        self.use_aem = use_aem

        if self.use_aem:
            self.aem = AEM(
                weight_initial=aem_weight_initial,
                weight_final=aem_weight_final,
                annealing_epochs=aem_annealing_epochs,
            )

        model_config: dict[str, Any] = {
            "model_class": model.__class__.__name__,
            "size_arg": model.size_arg,
            "n_classes": model.n_classes,
            "temperature": model.temperature,
            "embed_dim": model.embed_dim,
            "cell_types": model.cell_types,
            "heads_aggregation": model.heads_aggregation,
            "dropout": model.dropout,
            "custom_aggregation_weights": model.custom_weights.cpu().numpy().tolist()  # type: ignore
            if model.custom_weights is not None  # type: ignore
            else None,
        }

        self.save_hyperparameters(
            {
                **model_config,
                "optimizer_class": optimizer.__class__.__name__,
                "optimizer_lr": optimizer.param_groups[0]["lr"],
                "loss": loss,
                "lr_scheduler_class": lr_scheduler.__class__.__name__
                if lr_scheduler
                else None,
                "subsampling": subsampling,
                "use_aem": use_aem,
                "aem_weight_initial": aem_weight_initial,
                "aem_weight_final": aem_weight_final,
                "aem_annealing_epochs": aem_annealing_epochs,
            }
        )

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
        Load a model from a checkpoint.

        Args:
            checkpoint_path (str | Path | IO[bytes]): Path to the checkpoint file or a file-like object.
            map_location (optional): Device mapping for loading the model.
            hparams_file (optional): Path to a YAML file containing hyperparameters.
            strict (optional): Whether to strictly enforce that the keys in state_dict match the keys returned by the model's state_dict function.
            **kwargs: Additional keyword arguments passed to the model's constructor.

        Returns:
            An instance of LitAttentionDeepMIL.
        """

        checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)  # type: ignore
        hparams = checkpoint.get("hyper_parameters", {})

        model_class = Head4Type
        model = model_class(
            embed_dim=hparams.get("embed_dim", 1024),
            n_classes=hparams.get("n_classes", 2),
            size_arg=hparams.get("size_arg", [512, 128]),
            temperature=hparams.get("temperature", 1.0),
            cell_types=hparams.get("cell_types", 5),
            heads_aggregation=hparams.get("heads_aggregation", "weighted_mean"),
            dropout=hparams.get("dropout", 0.25),
            custom_aggregation_weights=hparams.get("custom_aggregation_weights", None),
        )

        optimizer_cls = getattr(torch.optim, hparams.get("optimizer_class", "Adam"))
        optimizer = optimizer_cls(
            model.parameters(), lr=hparams.get("optimizer_lr", 1e-3)
        )

        loss_fn = hparams.get("loss", "CrossEntropyLoss")

        lit_model = cls(
            model=model,
            optimizer=optimizer,
            loss=loss_fn,
            lr_scheduler=None,  # type: ignore
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

    def forward(self, x: torch.Tensor, cell_types: torch.Tensor) -> torch.Tensor:  # type: ignore
        logits, _ = self.model(x, cell_types)
        return logits

    def _shared_step(  # type: ignore
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        stage: str,
        log: bool = True,
    ):
        x, cell_types, y = batch

        # Ensure MIL batch size is 1
        assert x.size(0) == 1, "Batch size must be 1 for MIL"
        x = x.squeeze(0)  # [n_instances, feat_dim]
        cell_types = cell_types.squeeze(0)  # [n_instances, n_cell_types]

        # Apply subsampling during training
        if stage == "train" and self.subsampling < 1.0:
            # Calculate the number of samples to keep
            num_samples = int(self.subsampling * x.shape[0])
            # Generate random permutation of indices
            indices = torch.randperm(x.shape[0], device=x.device)
            # Select the first N samples from the permuted indices
            sampled_indices = indices[:num_samples]
            # Use the sampled indices to select instances
            x = x[sampled_indices]
            cell_types = cell_types[sampled_indices]

        logits, output_dict = self.model(x, cell_types)
        loss = self.loss(logits, y)

        # AEM (Attention Entropy Maximization)
        current_epoch = self.current_epoch if hasattr(self, "current_epoch") else 0
        aem: torch.Tensor | None = None
        if self.use_aem and stage == "train":
            attention_weights = output_dict[
                "attention"
            ]  # Get attention weights from model output
            aem = self.aem.get_aem(current_epoch, attention_weights)
            loss = loss + aem

        if torch.isnan(loss):
            print("Loss is NaN!")
            print(f"logits: {logits}")
            print(f"y: {y}")
            print(f"aem: {aem}")
            input("Press Enter to continue...")

        if log:
            self.log(
                f"{stage}/total_loss",
                loss,
                prog_bar=(stage != "train"),
                on_step=(stage == "train"),
                on_epoch=True,
            )

            if self.use_aem and stage == "train" and aem is not None:
                self.log(
                    f"{stage}/aem", aem, prog_bar=True, on_step=False, on_epoch=True
                )

        return loss, logits, y

    def get_attention_weights(
        self, x: torch.Tensor, cell_types: torch.Tensor
    ) -> torch.Tensor:
        """
        Get attention weights for the input instances.

        Args:
            x (torch.Tensor): Input tensor of shape [n_instances, feat_dim].
            cell_types (torch.Tensor): Cell type tensor of shape [n_instances, n_cell_types].

        Returns:
            torch.Tensor: Attention weights of shape [cell_types, n_instances].
        """
        self.model.eval()
        if len(x.shape) != 2:
            raise ValueError("Input tensor must be of shape [n_instances, feat_dim]")
        if len(cell_types.shape) != 2:
            raise ValueError(
                "Cell types tensor must be of shape [n_instances, n_cell_types]"
            )

        _, output_dict = self.model(x, cell_types)
        return output_dict["attention"]


class LitSurvHead4Type(LitHead4Type):
    def __init__(
        self,
        model: Head4Type,
        optimizer: torch.optim.Optimizer,
        loss: nn.Module = NegativeLogLikelihoodSurvLoss(),
        lr_scheduler: LRScheduler | None = None,
        subsampling: float = 0.8,
        use_aem: bool = True,
        aem_weight_initial: float = 0.0001,
        aem_weight_final: float = 0.0,
        aem_annealing_epochs: int = 25,
    ):
        super().__init__(
            model,
            optimizer,
            loss,
            lr_scheduler,
            subsampling,
            use_aem,
            aem_weight_initial,
            aem_weight_final,
            aem_annealing_epochs,
        )

        # For logistic hazard, n_classes should equal num_bins
        # Store this for converting back to continuous risk scores
        self.num_bins = model.n_classes

        # Setup survival-specific metrics
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

    def predict_step(  # type: ignore
        self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ):
        """Prediction step returns logits for discrete-time hazard intervals."""
        x, cell_types, _ = batch

        # Ensure MIL batch size is 1
        assert x.size(0) == 1, "Batch size must be 1 for MIL"
        x = x.squeeze(0)  # [n_instances, feat_dim]
        cell_types = cell_types.squeeze(0)  # [n_instances, n_cell_types]

        logits, _ = self.model(x, cell_types)
        return logits  # Return logits, not hazards

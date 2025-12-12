# Attention-based Deep Multiple Instance Learning Model Implementation
#
# Reference:
# Ilse, M., Tomczak, J. M., & Welling, M. (2018). Attention-based Deep Multiple Instance Learning.
# arXiv preprint arXiv:1802.04712

from typing_extensions import Self
import torch
import torchmetrics
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LRScheduler
from .utils import LitGeneral, AEM
from typing import IO, Any, Callable
from pathlib import Path
from cellmil.utils.train.losses import NegativeLogLikelihoodSurvLoss
from cellmil.utils.train.metrics import ConcordanceIndex, BrierScore


class AttentionDeepMIL(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        n_classes: int = 2,
        size_arg: list[int] = [500, 128],
        attention_branches: int = 1,
        temperature: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()  # type: ignore
        self.M = size_arg[-2]
        self.L = size_arg[-1]
        self.embed_dim = embed_dim
        self.ATTENTION_BRANCHES = attention_branches
        self.temperature = temperature
        self.dropout = dropout
        self.n_classes = n_classes

        # Build feature extractor layers based on size_arg
        # If size_arg has more than 2 values, add intermediate layers
        fe_layers: list[nn.Module] = []
        input_dim = self.embed_dim
        for hidden_dim in size_arg[:-1]:  # All dims except the last one (L)
            fe_layers.append(nn.Linear(input_dim, hidden_dim))
            fe_layers.append(nn.ReLU())
            input_dim = hidden_dim
        self.feature_extractor_part2 = nn.Sequential(*fe_layers)

        self.attention = nn.Sequential(
            nn.Linear(self.M, self.L),  # matrix V
            nn.Tanh(),
            nn.Linear(
                self.L, self.ATTENTION_BRANCHES
            ),  # matrix w (or vector w if self.ATTENTION_BRANCHES==1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(self.dropout),
            nn.Linear(self.M * self.ATTENTION_BRANCHES, self.n_classes),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if len(x.shape) != 2:
            raise ValueError("Input tensor must be 2D (KxD)")

        h = self.feature_extractor_part2(x)  # KxM

        a = self.attention(h)  # KxATTENTION_BRANCHES
        a = torch.transpose(a, 1, 0)  # ATTENTION_BRANCHESxK
        a = F.softmax(a / self.temperature, dim=1)  # softmax over K

        z = torch.mm(a, h)  # ATTENTION_BRANCHESxM

        logits = self.classifier(z.unsqueeze(0))
        y_prob = F.softmax(logits, dim=1)
        y_hat = torch.topk(y_prob, 1, dim=1)[1]

        output_dict = {
            "y_prob": y_prob,
            "y_hat": y_hat,
            "attention": a,
        }

        return logits, output_dict


class LitAttentionDeepMIL(LitGeneral):
    """
    Lightning wrapper for AttentionDeepMIL model .

    This class extends the base LitGeneral class to provide Lightning-specific functionality
    for the AttentionDeepMIL model..

    Args:
        model (nn.Module): The AttentionDeepMIL model instance.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        loss (nn.Module, optional): Loss function. Defaults to nn.CrossEntropyLoss().
        lr_scheduler (LRScheduler | None, optional): Learning rate scheduler. Defaults to None.
        subsampling (float, optional): Fraction of instances to use during training (between 0 and 1). Defaults to 1.0 (no subsampling).
        use_aem (bool, optional): Whether to use AEM regularization. Defaults to False.
        aem_weight_initial (float, optional): Initial weight for AEM loss. Defaults to 0.001.
        aem_weight_final (float, optional): Final weight for AEM loss after annealing. Defaults to 0.0.
        aem_annealing_epochs (int, optional): Number of epochs to anneal AEM weight. Defaults to 25.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss: nn.Module = nn.CrossEntropyLoss(),
        lr_scheduler: LRScheduler | None = None,
        subsampling: float = 1.0,
        use_aem: bool = False,
        aem_weight_initial: float = 0.0001,
        aem_weight_final: float = 0.0,
        aem_annealing_epochs: int = 25,
    ):
        super().__init__(model, optimizer, loss, lr_scheduler)
        self.n_classes = 2
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
            "size_arg": [model.M, model.L],
            "n_classes": model.n_classes,
            "attention_branches": model.ATTENTION_BRANCHES,
            "temperature": model.temperature,
            "embed_dim": model.embed_dim,
            "dropout": model.dropout,
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

        checkpoint = torch.load(
            checkpoint_path,
            map_location=map_location,  # type: ignore
            weights_only=False,
        )
        hparams = checkpoint.get("hyper_parameters", {})

        model_class = AttentionDeepMIL
        model = model_class(
            embed_dim=hparams.get("embed_dim", 1024),
            n_classes=hparams.get("n_classes", 2),
            size_arg=hparams.get("size_arg", [500, 128]),
            attention_branches=hparams.get("attention_branches", 1),
            temperature=hparams.get("temperature", 1.0),
            dropout=hparams.get("dropout", 0.25),
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.model(x)
        return logits

    def _shared_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], stage: str, log: bool = True
    ):
        x, y = batch

        # Ensure MIL batch size is 1
        assert x.size(0) == 1, "Batch size must be 1 for MIL"
        x = x.squeeze(0)  # [n_instances, feat_dim]

        # Apply subsampling during training
        if stage == "train" and self.subsampling != 1.0:
            # Calculate the number of samples to keep
            if 0 < self.subsampling < 1.0:
                # Treat as percentage
                num_samples = int(self.subsampling * x.shape[0])
            elif self.subsampling >= 1.0:
                # Treat as absolute count
                num_samples = min(int(self.subsampling), x.shape[0])
            else:
                raise ValueError(f"Invalid subsampling value: {self.subsampling}")

            # Generate random permutation of indices
            indices = torch.randperm(x.shape[0], device=x.device)
            # Select the first N samples from the permuted indices
            sampled_indices = indices[:num_samples]
            # Use the sampled indices to select instances
            x = x[sampled_indices]

        logits, output_dict = self.model(x)
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

        if log:
            self.log(
                f"{stage}/total_loss",
                loss,
                prog_bar=(stage != "train"),
                on_step=(stage == "train"),
                on_epoch=True,
            )
            if current_epoch == 0 and stage in ["train", "val"]:
                self.log(
                    f"{stage}/num_instances",
                    batch[0].squeeze(0).shape[0],
                    prog_bar=False,
                    on_step=True,
                    on_epoch=False,
                )

            if self.use_aem and stage == "train" and aem is not None:
                self.log(
                    f"{stage}/aem", aem, prog_bar=True, on_step=False, on_epoch=True
                )

        return loss, logits, y

    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get attention weights for the input instances.

        Args:
            x (torch.Tensor): Input tensor of shape [n_instances, feat_dim].

        Returns:
            torch.Tensor: Attention weights of shape [attention_branches, n_instances].
        """
        self.model.eval()
        if len(x.shape) != 2:
            raise ValueError("Input tensor must be of shape [n_instances, feat_dim]")

        _, output_dict = self.model(x)
        return output_dict["attention"]


class LitSurvAttentionDeepMIL(LitAttentionDeepMIL):
    def __init__(
        self,
        model: AttentionDeepMIL,
        optimizer: torch.optim.Optimizer,
        loss: nn.Module = NegativeLogLikelihoodSurvLoss(),
        lr_scheduler: LRScheduler | None = None,
        subsampling: float = 1.0,
        use_aem: bool = False,
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

    def predict_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        """Prediction step returns logits for discrete-time hazard intervals."""
        x, _ = batch

        # Ensure MIL batch size is 1
        assert x.size(0) == 1, "Batch size must be 1 for MIL"
        x = x.squeeze(0)  # [n_instances, feat_dim]

        logits, _ = self.model(x)
        return logits  # Return logits, not hazards

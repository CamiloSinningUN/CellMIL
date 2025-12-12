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


def _groupnorm_groups(channels: int) -> int:
    for candidate in (32, 16, 8, 4, 2):
        if channels % candidate == 0:
            return candidate
    return 1


class _ResidualConvBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()  # type: ignore[misc]
        padding = ((kernel_size - 1) // 2) * dilation
        groups = _groupnorm_groups(channels)
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=channels,
            bias=False,
        )
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.depthwise(x)
        out = self.norm1(out)
        out = self.act(out)
        out = self.dropout(out)
        out = self.pointwise(out)
        out = self.norm2(out)
        out = self.dropout(out)
        out = out + residual
        return self.act(out)


class CellConv(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        n_classes: int = 2,
        convolution_depth: int = 3,
        size_arg: list[int] = [512, 128],
        attention_branches: int = 1,
        temperature: float = 1.0,
        dropout: float = 0.0,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()  # type: ignore
        if len(size_arg) != 2:
            raise ValueError("size_arg must contain [latent_dim, attention_dim]")

        self.convolution_depth = convolution_depth
        self.latent_dim = size_arg[0]
        self.attention_dim = size_arg[1]
        self.embed_dim = embed_dim
        self.temperature = temperature
        self.n_classes = n_classes
        self.dropout = dropout
        self.kernel_size = kernel_size
        self.ATTENTION_BRANCHES = attention_branches

        self.input_proj = nn.Sequential(
            nn.LazyLinear(self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.Dropout(self.dropout),
        )

        dilations: list[int] = []
        dilation = 1
        for _ in range(max(1, self.convolution_depth)):
            dilations.append(dilation)
            if dilation < 16:
                dilation *= 2

        self.conv_backbone = nn.ModuleList(
            [
                _ResidualConvBlock(
                    channels=self.embed_dim,
                    kernel_size=self.kernel_size,
                    dilation=d,
                    dropout=self.dropout,
                )
                for d in dilations
            ]
        )

        self.post_conv_norm = nn.LayerNorm(self.embed_dim)
        self.feature_head = nn.Sequential(
            nn.Linear(self.embed_dim, self.latent_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
        )

        self.attention = nn.Sequential(
            nn.Linear(self.latent_dim, self.attention_dim),
            nn.Tanh(),
            nn.Linear(self.attention_dim, self.ATTENTION_BRANCHES),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(self.dropout),
            nn.Linear(self.latent_dim * self.ATTENTION_BRANCHES, self.n_classes),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if x.dim() != 2:
            raise ValueError("Input tensor must be 2D (num_cells x feature_dim)")

        h = self.input_proj(x)
        conv_input = h.transpose(0, 1).unsqueeze(0)
        for block in self.conv_backbone:
            conv_input = block(conv_input)
        h = conv_input.squeeze(0).transpose(0, 1).contiguous()
        h = self.post_conv_norm(h)
        features = self.feature_head(h)

        attention_scores = self.attention(features)
        attention_scores = attention_scores.transpose(0, 1)
        attention_weights = F.softmax(attention_scores / self.temperature, dim=1)

        bag_representation = torch.mm(attention_weights, features)
        logits = self.classifier(bag_representation.unsqueeze(0))
        y_prob = F.softmax(logits, dim=1)
        y_hat = torch.topk(y_prob, 1, dim=1)[1]

        output_dict = {
            "y_prob": y_prob,
            "y_hat": y_hat,
            "attention": attention_weights,
            "bag_representation": bag_representation,
        }

        return logits, output_dict


class LitCellConv(LitGeneral):
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
    ) -> None:
        super().__init__(model, optimizer, loss, lr_scheduler)
        self.n_classes = getattr(model, "n_classes", 2)
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
            "embed_dim": getattr(model, "embed_dim", None),
            "latent_dim": getattr(model, "latent_dim", None),
            "attention_dim": getattr(model, "attention_dim", None),
            "attention_branches": getattr(model, "ATTENTION_BRANCHES", 1),
            "convolution_depth": getattr(model, "convolution_depth", None),
            "kernel_size": getattr(model, "kernel_size", None),
            "temperature": getattr(model, "temperature", 1.0),
            "dropout": getattr(model, "dropout", 0.0),
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
        checkpoint = torch.load(
            checkpoint_path,
            map_location=map_location,  # type: ignore[arg-type]
            weights_only=False,
        )
        hparams = checkpoint.get("hyper_parameters", {})

        model = CellConv(
            embed_dim=hparams.get("embed_dim", 256),
            n_classes=hparams.get("n_classes", 2),
            convolution_depth=hparams.get("convolution_depth", 3),
            size_arg=[
                hparams.get("latent_dim", 512),
                hparams.get("attention_dim", 128),
            ],
            attention_branches=hparams.get("attention_branches", 1),
            temperature=hparams.get("temperature", 1.0),
            dropout=hparams.get("dropout", 0.1),
            kernel_size=hparams.get("kernel_size", 3),
        )

        optimizer_cls = getattr(torch.optim, hparams.get("optimizer_class", "Adam"))
        optimizer = optimizer_cls(
            model.parameters(), lr=hparams.get("optimizer_lr", 1e-3)
        )

        loss_param = hparams.get("loss", "CrossEntropyLoss")
        if isinstance(loss_param, str):
            loss_fn = getattr(nn, loss_param)()
        else:
            loss_fn = loss_param

        lit_model = cls(
            model=model,
            optimizer=optimizer,
            loss=loss_fn,
            lr_scheduler=None,  # type: ignore[arg-type]
            subsampling=hparams.get("subsampling", 1.0),
            use_aem=hparams.get("use_aem", False),
            aem_weight_initial=hparams.get("aem_weight_initial", 0.0001),
            aem_weight_final=hparams.get("aem_weight_final", 0.0),
            aem_annealing_epochs=hparams.get("aem_annealing_epochs", 25),
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
        assert x.size(0) == 1, "Batch size must be 1 for MIL"
        x = x.squeeze(0)

        logits, output_dict = self.model(x)
        loss = self.loss(logits, y)

        current_epoch = self.current_epoch if hasattr(self, "current_epoch") else 0
        aem_term: torch.Tensor | None = None
        if self.use_aem and stage == "train":
            attention_weights = output_dict["attention"]
            aem_term = self.aem.get_aem(current_epoch, attention_weights)
            loss = loss + aem_term

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
            if self.use_aem and stage == "train" and aem_term is not None:
                self.log(
                    f"{stage}/aem",
                    aem_term,
                    prog_bar=True,
                    on_step=False,
                    on_epoch=True,
                )

        return loss, logits, y

    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        if x.dim() != 2:
            raise ValueError("Input tensor must be of shape [n_instances, feat_dim]")
        _, output_dict = self.model(x)
        return output_dict["attention"]

    def transfer_batch_to_device(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        device: torch.device,
        dataloader_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = batch
        if self.training and self.subsampling != 1.0:
            if x.size(0) != 1:
                raise ValueError("Batch size must be 1 for MIL")
            bag = x.squeeze(0)
            if 0 < self.subsampling < 1.0:
                num_samples = max(int(self.subsampling * bag.shape[0]), 1)
            elif self.subsampling >= 1.0:
                num_samples = min(int(self.subsampling), bag.shape[0])
            else:
                raise ValueError(f"Invalid subsampling value: {self.subsampling}")
            if num_samples < bag.shape[0]:
                sampled_indices = torch.randperm(bag.shape[0])[:num_samples]
                bag = bag.index_select(0, sampled_indices)
            x = bag.unsqueeze(0)

        return super().transfer_batch_to_device((x, y), device, dataloader_idx)


class LitSurvCellConv(LitCellConv):
    def __init__(
        self,
        model: CellConv,
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

import torch
import torch.nn as nn
from .utils import LitGeneral, AEM
from torch.optim.lr_scheduler import LRScheduler
from typing import IO, Any, Callable
from typing_extensions import Self
from pathlib import Path

class MultiFocus(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        n_classes: int = 2,
        size_arg: list[int] = [32],
        temperature: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()  # type: ignore
        self.embed_dim = embed_dim
        self.n_classes = n_classes
        self.size_arg = [embed_dim] + size_arg
        self.temperature = temperature
        self.dropout = dropout
        
        if len(self.size_arg) > 2:
            self.feature_extractor = nn.Sequential()
            for i in range(len(self.size_arg) - 2):
                self.feature_extractor.append(nn.Linear(self.size_arg[i], self.size_arg[i + 1]))
                self.feature_extractor.append(nn.ReLU())
                self.feature_extractor.append(nn.Dropout(self.dropout))

        self.attention = nn.Sequential(
            nn.Linear(self.size_arg[-2], self.size_arg[-1]),
            nn.Tanh(),
            nn.Dropout(self.dropout),
            nn.Linear(self.size_arg[-1], self.embed_dim),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim, self.n_classes),
        )
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if len(x.shape) != 2:
            raise ValueError(f"Expected input tensor of shape (N, D), got {x.shape}")
        
        if len(self.size_arg) > 2:
            h = self.feature_extractor(x) # (N, size_arg[-2])
        else:
            h = x  # (N, embed_dim)

        a = self.attention(h)  # (N, embed_dim)
        a = torch.transpose(a, 1, 0)  # (embed_dim, N)
        a = torch.softmax(a / self.temperature, dim=1)  # (embed_dim, N)
        
        m = torch.mm(a, x)  # (embed_dim, embed_dim)
        
        # Take diagonal elements
        m = torch.diagonal(m, 0)  # (embed_dim,)

        logits = self.classifier(m.unsqueeze(0))  # (1, n_classes)

        return logits, {"attention": a}
    
class LitMultiFocus(LitGeneral):
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss: nn.Module = nn.CrossEntropyLoss(),
        lr_scheduler: LRScheduler | None = None,
        subsampling: float = 0.8,
        use_aem: bool = True,
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

        checkpoint = torch.load(checkpoint_path, map_location=map_location)  # type: ignore
        hparams = checkpoint.get("hyper_parameters", {})

        model_class = MultiFocus
        model = model_class(
            embed_dim=hparams.get("embed_dim", 1024),
            n_classes=hparams.get("n_classes", 2),
            size_arg=hparams.get("size_arg", [512]),
            temperature=hparams.get("temperature", 1.0),
            dropout=hparams.get("dropout", 0.25),
        )

        optimizer_cls = getattr(torch.optim, hparams.get("optimizer_class", "Adam"))
        optimizer = optimizer_cls(
            model.parameters(), lr=hparams.get("optimizer_lr", 1e-4)
        )

        loss_fn = getattr(torch.nn, hparams.get("loss", "CrossEntropyLoss"))()

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        logits, _ = self.model(x)
        return logits

    def _shared_step(  # type: ignore
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        stage: str,
        log: bool = True,
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
        self, x: torch.Tensor
    ) -> torch.Tensor:
        """
        Get attention weights for the input instances.

        Args:
            x (torch.Tensor): Input tensor of shape [n_instances, feat_dim].

        Returns:
            torch.Tensor: Attention weights of shape [embed_dim, n_instances].
        """
        self.model.eval()
        if len(x.shape) != 2:
            raise ValueError("Input tensor must be of shape [n_instances, feat_dim]")

        _, output_dict = self.model(x)
        return output_dict["attention"]
import torch
import torchmetrics
import torch.nn as nn
import lightning as Pl
from pathlib import Path
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from typing_extensions import Self
from typing import Any, cast, IO, Callable
from torch_geometric.data import Data  # type: ignore
from torch_geometric.loader import NeighborLoader  # type: ignore
from .gnn import GNN, GAT, EGNN, SAGE, CHIMERA, GATv2, SmallWorld, SGFormer
from .pool import GlobalPooling_Classifier, CLAM, Standard, Attention, Mean_MLP
from ..utils import AEM

__all__ = [
    "GNN",
    "GAT",
    "GATv2",
    "EGNN",
    "SAGE",
    "CHIMERA",
    "GlobalPooling_Classifier",
    "CLAM",
    "Standard",
    "Attention",
    "Mean_MLP",
    "LitGraphMIL",
    "SmallWorld",
    "SGFormer",
]


class LitGraphMIL(Pl.LightningModule):
    """
    Lightning module for Graph-based Multiple Instance Learning.

    This model is designed to work with torch_geometric DataLoader and requires:
    - batch_size=1 for MIL tasks
    - Data objects with batch.y containing graph labels
    - GNNMILDataset from cellmil.datamodels.datasets.gnn_mil_dataset

    Example usage:
        from torch_geometric.loader import DataLoader
        from cellmil.datamodels.datasets.gnn_mil_dataset import GNNMILDataset

        dataset = GNNMILDataset(...)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

        model = LitGraphMIL(gnn=..., pooling_classifier=..., ...)
        trainer.fit(model, dataloader)
    """

    def __init__(
        self,
        gnn: GNN,
        pooling_classifier: GlobalPooling_Classifier,
        optimizer_cls: type[Optimizer],
        optimizer_kwargs: dict[str, Any],
        loss_fn: nn.Module = nn.CrossEntropyLoss(),
        scheduler_cls: type[LRScheduler] | None = None,
        scheduler_kwargs: dict[str, Any] | None = None,
        use_aem: bool = False,
        aem_weight_initial: float = 0.0001,
        aem_weight_final: float = 0.0,
        aem_annealing_epochs: int = 25,
        subsampling: float = 1.0,
        **kwargs: Any,
    ):
        super().__init__()
        self.gnn = gnn
        self.pooling_classifier = pooling_classifier
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = optimizer_kwargs
        self.loss_fn = loss_fn
        self.scheduler_cls = scheduler_cls
        self.scheduler_kwargs = scheduler_kwargs if scheduler_kwargs else {}
        self.subsampling = subsampling

        # AEM setup
        self.use_aem = use_aem and isinstance(pooling_classifier, (CLAM, Attention))
        if self.use_aem:
            self.aem = AEM(
                weight_initial=aem_weight_initial,
                weight_final=aem_weight_final,
                annealing_epochs=aem_annealing_epochs,
            )

        if isinstance(gnn.hidden_dim, list):
            gnn_hidden_dim = gnn.hidden_dim[-1]
        else:
            gnn_hidden_dim = gnn.hidden_dim
        assert gnn_hidden_dim == self.pooling_classifier.input_dim, (
            "GNN hidden dimension must match pooling classifier input dimension"
        )

        # Clean hyperparameter collection using the elegant approach
        hyperparams: dict[str, Any] = {
            # GNN hyperparameters with prefix
            **{f"gnn_{key}": value for key, value in gnn.get_hyperparameters().items()},
            # Pooling classifier hyperparameters with prefix
            **{
                f"pooling_{key}": value
                for key, value in pooling_classifier.get_hyperparameters().items()
            },
            # Optimizer parameters with prefix
            "optimizer_type": optimizer_cls.__name__,
            **{f"optimizer_{key}": value for key, value in optimizer_kwargs.items()},
            # Loss function
            "loss_fn": loss_fn.__class__.__name__
            if hasattr(loss_fn, "__class__")
            else str(loss_fn),
            # Scheduler parameters if provided
            "scheduler_type": scheduler_cls.__name__ if scheduler_cls else None,
            **(
                {
                    f"scheduler_{key}": value
                    for key, value in self.scheduler_kwargs.items()
                }
                if scheduler_cls
                else {}
            ),
            # AEM parameters
            "use_aem": self.use_aem,
            "aem_weight_initial": aem_weight_initial,
            "aem_weight_final": aem_weight_final,
            "aem_annealing_epochs": aem_annealing_epochs,
            # Subsampling parameters
            "subsampling": subsampling,
            # Any additional kwargs
            **kwargs,
        }

        self.save_hyperparameters(hyperparams)
        self._setup_metrics()

        self.bag_size: int = 0

        if isinstance(self.pooling_classifier, CLAM):
            self.weight_loss_slide: float = cast(
                float, kwargs.get("weight_loss_slide", 0.7)
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
            **kwargs: Additional keyword arguments passed to the model's constructor
        Returns:
            An instance of LitGraphMIL.
        """

        checkpoint = torch.load(
            checkpoint_path,
            map_location=map_location, # type: ignore
            weights_only=False,  
        )
        hparams = checkpoint.get("hyper_parameters", {})

        # Extract parameters with user overrides
        def get_param(key: str, default: Any = None) -> Any:
            return kwargs.get(key, hparams.get(key, default))

        # Reconstruct GNN - use the type from checkpoint, no default override
        gnn_type = hparams.get("gnn_type")
        if not gnn_type:
            raise ValueError("gnn_type not found in checkpoint hyperparameters")
        gnn_class = globals().get(gnn_type)
        if not gnn_class:
            raise ValueError(f"Unknown GNN type: {gnn_type}")
        gnn_params = {
            "input_dim": get_param("gnn_input_dim", 128),
            "hidden_dim": get_param("gnn_hidden_dim", 256),
            "n_layers": get_param("gnn_n_layers", 2),
            "dropout": get_param("gnn_dropout", 0.0),
        }
        # Add all other gnn_ parameters
        for key, value in hparams.items():
            if key.startswith("gnn_") and key not in [
                "gnn_type",
                "gnn_input_dim",
                "gnn_hidden_dim",
                "gnn_n_layers",
                "gnn_dropout",
            ]:
                param_name = key.replace("gnn_", "")
                gnn_params[param_name] = get_param(key, value)

        # Add any user-provided GNN parameters that might not be in the checkpoint
        for key, value in kwargs.items():
            if key.startswith("gnn_"):
                param_name = key.replace("gnn_", "")
                gnn_params[param_name] = value

        gnn = gnn_class(**gnn_params)

        # Reconstruct Pooling Classifier - use the type from checkpoint, no default override
        pooling_type = hparams.get("pooling_type")
        if not pooling_type:
            raise ValueError("pooling_type not found in checkpoint hyperparameters")
        pooling_class = globals().get(pooling_type)
        if not pooling_class:
            raise ValueError(f"Unknown pooling type: {pooling_type}")
        pooling_params = {
            "input_dim": get_param("pooling_input_dim", 256),
            "dropout": get_param("pooling_dropout", 0.0),
            "n_classes": get_param("pooling_n_classes", 2),
            "size_arg": get_param("pooling_size_arg", [128]),
        }
        # Add all other pooling_ parameters
        for key, value in hparams.items():
            if key.startswith("pooling_") and key not in [
                "pooling_type",
                "pooling_input_dim",
                "pooling_dropout",
                "pooling_n_classes",
                "pooling_size_arg",
            ]:
                param_name = key.replace("pooling_", "")
                # Special handling for instance_loss_fn
                if param_name == "instance_loss_fn" and value == "SmoothTop1SVM":
                    from topk.svm import SmoothTop1SVM  # type: ignore

                    pooling_params[param_name] = SmoothTop1SVM(
                        n_classes=pooling_params["n_classes"]
                    )
                else:
                    pooling_params[param_name] = get_param(key, value)

        # Add any user-provided pooling parameters that might not be in the checkpoint
        for key, value in kwargs.items():
            if key.startswith("pooling_"):
                param_name = key.replace("pooling_", "")
                pooling_params[param_name] = value

        pooling_classifier = pooling_class(**pooling_params)

        # Reconstruct other components
        optimizer_class = getattr(torch.optim, get_param("optimizer_type", "Adam"))
        optimizer_kwargs = {
            key.replace("optimizer_", ""): value
            for key, value in hparams.items()
            if key.startswith("optimizer_") and key != "optimizer_type"
        }

        loss_fn_name = get_param("loss_fn", "CrossEntropyLoss")
        loss_fn = getattr(nn, loss_fn_name, nn.CrossEntropyLoss)()

        scheduler_class = None
        scheduler_kwargs = None
        if hparams.get("scheduler_type"):
            scheduler_class = getattr(
                torch.optim.lr_scheduler, hparams["scheduler_type"]
            )
            scheduler_kwargs = {
                key.replace("scheduler_", ""): value
                for key, value in hparams.items()
                if key.startswith("scheduler_") and key != "scheduler_type"
            }

        # Additional kwargs for LitGraphMIL
        lit_kwargs = {
            k: v
            for k, v in kwargs.items()
            if not k.startswith(("gnn_", "pooling_", "optimizer_", "scheduler_"))
        }
        if isinstance(pooling_classifier, CLAM):
            lit_kwargs.setdefault(
                "weight_loss_slide", hparams.get("weight_loss_slide", 0.7)
            )

        # Add AEM parameters from checkpoint with defaults
        lit_kwargs.setdefault("use_aem", get_param("use_aem", False))
        lit_kwargs.setdefault(
            "aem_weight_initial", get_param("aem_weight_initial", 0.0001)
        )
        lit_kwargs.setdefault("aem_weight_final", get_param("aem_weight_final", 0.0))
        lit_kwargs.setdefault(
            "aem_annealing_epochs", get_param("aem_annealing_epochs", 25)
        )
        lit_kwargs.setdefault("subsampling", get_param("subsampling", 1.0))

        lit_model = cls(
            gnn=gnn,
            pooling_classifier=pooling_classifier,
            optimizer_cls=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            loss_fn=loss_fn,
            scheduler_cls=scheduler_class,
            scheduler_kwargs=scheduler_kwargs,
            **lit_kwargs,
        )
        lit_model.load_state_dict(
            checkpoint["state_dict"], strict=strict if strict is not None else True
        )
        return lit_model

    def _setup_metrics(self):
        metrics = torchmetrics.MetricCollection(
            {
                "accuracy": torchmetrics.Accuracy(
                    task="multiclass",
                    num_classes=self.pooling_classifier.n_classes,
                    average="none",
                ),
                "f1": torchmetrics.F1Score(
                    task="multiclass",
                    num_classes=self.pooling_classifier.n_classes,
                    average="macro",
                ),
                "precision": torchmetrics.Precision(
                    task="multiclass",
                    num_classes=self.pooling_classifier.n_classes,
                    average="macro",
                ),
                "recall": torchmetrics.Recall(
                    task="multiclass",
                    num_classes=self.pooling_classifier.n_classes,
                    average="macro",
                ),
                "auroc": torchmetrics.AUROC(
                    task="multiclass",
                    num_classes=self.pooling_classifier.n_classes,
                    average="macro",
                ),
            }
        )
        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

    def _subsample_graph(self, data: Data, subsampling: float) -> Data:
        """
        Sample subgraph using NeighborLoader to preserve local graph structure.
        
        This method uses k-hop neighborhood sampling which preserves the local
        connectivity around seed nodes, providing better context for GNN message
        passing compared to random node sampling.
        
        Note: This method is designed to work on CPU before GPU transfer when called
        from on_before_batch_transfer hook, saving GPU memory and transfer bandwidth.

        Args:
            data (Data): Input graph data (typically on CPU).
            subsampling (float): Fraction of nodes to keep (0 < subsampling < 1.0)
                                or absolute number of nodes (subsampling >= 1.0).

        Returns:
            Data: Sampled subgraph with k-hop neighborhoods around seed nodes.
            
        Note:
            This method requires either 'pyg-lib' or 'torch-sparse' to be installed.
            Install with: pip install pyg-lib torch-sparse -f https://data.pyg.org/whl/torch-{TORCH_VERSION}+{CUDA_VERSION}.html
        """
        
        num_nodes = data.num_nodes
        if num_nodes is None:
            raise ValueError("Data object must have num_nodes attribute")
        
        # Determine number of seed nodes to sample based on subsampling parameter
        if 0 < subsampling < 1.0:
            # Treat as percentage
            num_sample_nodes = int(subsampling * num_nodes)
        elif subsampling >= 1.0:
            # Treat as absolute count
            num_sample_nodes = min(int(subsampling), num_nodes)
        else:
            raise ValueError(f"Invalid subsampling value: {subsampling}")
        
        # Determine number of seed nodes to sample
        # Always sample on CPU to avoid unnecessary GPU operations
        if num_sample_nodes >= num_nodes:
            # If requesting more nodes than available, use all nodes
            input_nodes = torch.arange(num_nodes, device='cpu')
        else:
            # Randomly select seed nodes
            input_nodes = torch.randperm(num_nodes, device='cpu')[:num_sample_nodes]
        
        # Determine neighbor sampling sizes based on GNN depth
        gnn_n_layers = self.gnn.n_layers if hasattr(self.gnn, 'n_layers') else 2
        # Start with more neighbors for first hop, decrease for subsequent hops
        neighbor_sample_sizes = [max(15 - (i * 5), 5) for i in range(gnn_n_layers)]
        
        # Ensure data is on CPU for sampling
        if data.x is not None and data.x.is_cuda:
            print("-"*50)
            print("Data is on GPU, moving to CPU for sampling.")
            data = data.cpu()
        
        # Create NeighborLoader for this single graph
        # Note: We set batch_size to the number of seed nodes to get one subgraph
        loader = NeighborLoader(
            data,
            num_neighbors=neighbor_sample_sizes,
            input_nodes=input_nodes,
            batch_size=len(input_nodes),
            shuffle=False,  # We already shuffled the input_nodes
            num_workers=0,  # Must be 0 for inline sampling
        )
        
        # Get the sampled subgraph (only one batch since batch_size = len(input_nodes))
        sampled_subgraph = next(iter(loader))
        
        # Preserve the original label
        sampled_subgraph.y = data.y
        
        return sampled_subgraph

    def forward(self, data: Data, **kwargs: Any):
        # Process with GNN
        _data = self.gnn(data)

        # Extract batch assignment for pooling (important for batched graphs)
        batch = getattr(data, "batch", None)

        # Apply pooling classifier with appropriate arguments
        if isinstance(self.pooling_classifier, CLAM):
            _label = kwargs.get("label", None)
            _instance_eval = kwargs.get("instance_eval", False)

            # CLAM needs label and instance_eval parameters
            logits, output_dict = self.pooling_classifier(
                _data.x, batch, label=_label, instance_eval=_instance_eval
            )
        else:
            # Other pooling classifiers don't use label or instance_eval
            logits, output_dict = self.pooling_classifier(_data.x, batch)
        return logits, output_dict

    def on_before_batch_transfer(self, batch: Data, dataloader_idx: int) -> Data:
        """
        Hook called before batch is transferred to GPU.
        Performs subsampling on CPU to reduce memory usage and transfer overhead.
        
        Args:
            batch (Data): Input graph data on CPU.
            dataloader_idx (int): Index of the dataloader.
            
        Returns:
            Data: Potentially subsampled graph data (still on CPU).
        """
        # Only subsample during training
        if self.training and self.subsampling != 1.0:
            # Subsample on CPU before GPU transfer
            batch = self._subsample_graph(batch, self.subsampling)
        
        return batch

    def _shared_step(
        self,
        batch: Data,  # Changed from tuple to Data (torch_geometric batch)
        stage: str,
        log: bool = True,
    ):
        # Extract data and labels from torch_geometric batch
        # batch.y contains the graph labels
        # batch.batch contains the batch assignment for nodes
        data = batch
        label = batch.y

        # Verify batch_size=1 for MIL
        if hasattr(batch, "batch") and batch.batch is not None:
            num_graphs = batch.batch.max().item() + 1
            if num_graphs > 1:
                raise ValueError(
                    f"GraphMIL requires batch_size=1 for MIL. Found {num_graphs} graphs in batch."
                )

        # For single graph case, ensure label is properly shaped
        if isinstance(label, torch.Tensor) and label.dim() == 0:
            label = label.unsqueeze(0)
        elif not isinstance(label, torch.Tensor):
            raise ValueError(f"Expected label to be a torch.Tensor, got {type(label)}")

        # Subsampling now happens in on_before_batch_transfer hook (before GPU transfer)
        
        self.bag_size = cast(int, data.num_nodes)

        logits, output_dict = self(data, label=label, instance_eval=True)

        slide_loss = self.loss_fn(logits, label)

        instance_loss = output_dict.get(
            "instance_loss", torch.tensor(0.0, device=logits.device)
        )

        if isinstance(self.pooling_classifier, CLAM):
            total_loss = (
                self.weight_loss_slide * slide_loss
                + (1 - self.weight_loss_slide) * instance_loss
            )
        else:
            total_loss = slide_loss

        # AEM (Attention Entropy Maximization)
        current_epoch = self.current_epoch if hasattr(self, "current_epoch") else 0
        aem: torch.Tensor | None = None
        if (
            self.use_aem
            and stage == "train"
            and isinstance(self.pooling_classifier, (CLAM, Attention))
        ):
            attention_weights = output_dict.get("attention", None)
            if attention_weights is not None:
                aem = self.aem.get_aem(current_epoch, attention_weights)
                total_loss = total_loss + aem

        y_hat = logits.argmax(dim=1)
        y_prob = torch.softmax(logits, dim=1)

        error = self.calculate_error(y_hat, label)

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
                # Log number of nodes in the graph
                self.log(
                    f"{stage}/num_nodes",
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

        return total_loss, y_prob, label

    def training_step(
        self,
        batch: Data,  # Changed from tuple to Data
        batch_idx: int,
    ):
        loss, y_prob, label = self._shared_step(batch, stage="train")
        self.train_metrics(y_prob, label)
        return loss

    def validation_step(
        self,
        batch: Data,  # Changed from tuple to Data
        batch_idx: int,
    ):
        loss, y_prob, label = self._shared_step(batch, stage="val")
        self.val_metrics(y_prob, label)
        return loss

    def test_step(
        self,
        batch: Data,  # Changed from tuple to Data
        batch_idx: int,
    ):
        loss, y_prob, label = self._shared_step(batch, stage="test")
        self.test_metrics(y_prob, label)
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
        optimizer = self.optimizer_cls(self.parameters(), **self.optimizer_kwargs)
        if self.scheduler_cls is not None:
            scheduler: dict[str, Any] = {
                "scheduler": self.scheduler_cls(optimizer, **self.scheduler_kwargs),
                "interval": "epoch",
                "monitor": "val/total_loss",
                "frequency": 1,
                "strict": True,
                "name": "learning_rate",
            }
            return [optimizer], [scheduler]
        return optimizer

    def predict_step(self, batch: Data, batch_idx: int):
        _, y_prob, _ = self._shared_step(batch, stage="test", log=False)
        return y_prob.argmax(dim=-1)

    @staticmethod
    def calculate_error(y_hat: torch.Tensor, y: torch.Tensor):
        """Classification error = 1 - accuracy."""
        return 1.0 - y_hat.float().eq(y.float()).float().mean().item()

    def get_attention_weights(self, data: Data) -> dict[str, torch.Tensor]:
        """
        Get attention weights from both GNN layers and pooling classifier.

        This method delegates to the individual component classes for clean separation
        of concerns and better maintainability.

        Args:
            data (Data): Input graph data.

        Returns:
            dict[str, torch.Tensor]: Dictionary containing attention weights:
                - GNN attention weights (if available): 'gnn_attention_layer_{i}'
                - Pooling attention weights (if available): 'pooling_attention'
        """
        self.eval()
        attention_weights: dict[str, torch.Tensor] = {}

        # Get GNN attention weights (delegates to GNN class)
        if isinstance(self.gnn, (GAT, GATv2)):
            gnn_attention = self.gnn.get_attention_weights(data)
            attention_weights.update(gnn_attention)

        # Get pooling attention weights (delegates to pooling classifier)
        if isinstance(self.pooling_classifier, (Attention, CLAM)):
            # Process data through GNN first to get the right features
            processed_data = self.gnn(data.clone())  # Clone to avoid modifying original
            batch = getattr(processed_data, "batch", None)

            pooling_attention = self.pooling_classifier.get_attention_weights(
                processed_data.x, batch
            )
            if pooling_attention is not None:
                attention_weights["pooling_attention"] = pooling_attention

        return attention_weights

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC
from typing import Any, cast, Literal, Optional
from torch_geometric.nn import SAGEConv, GATConv, GATv2Conv, SAGPooling, GCNConv  # type: ignore
# from torch_geometric.nn.attention import SGFormerAttention  # type: ignore
from torch_geometric.data import Data  # type: ignore
from egnn_pytorch import EGNN_Sparse  # type: ignore
from cellmil.utils import logger


class GNN(nn.Module, ABC):
    """Abstract base class for Graph Neural Networks.

    This class defines the interface for GNN models but cannot be instantiated directly.
    Subclasses must implement the layer creation logic in their __init__ method.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int | list[int],
        n_layers: int,
        dropout: float,
        **kwargs: Any,
    ):
        super().__init__()  # type: ignore
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.kwargs = kwargs
        self.convs = nn.ModuleList()

    def get_hyperparameters(self) -> dict[str, Any]:
        """Get all hyperparameters for this GNN."""
        return {
            "type": self.__class__.__name__,
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "n_layers": self.n_layers,
            "dropout": self.dropout,
            **self.kwargs,
            **self._get_specific_hyperparameters(),
        }

    def _get_specific_hyperparameters(self) -> dict[str, Any]:
        """Override in subclasses to add specific hyperparameters."""
        return {}

    def forward(self, data: Data) -> Data:
        x, edge_index = data.x, data.edge_index
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        data.x = x
        return data

    def get_attention_weights(self, data: Data) -> dict[str, torch.Tensor]:
        """
        Extract attention weights from GNN layers.

        Args:
            data: Input graph data

        Returns:
            Dictionary with attention weights from each layer (empty for non-attention GNNs)
        """
        return {}


class GAT(GNN):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_layers: int,
        dropout: float,
        heads: int = 1,
        **kwargs: Any,
    ):
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout,
            **kwargs,
        )

        self.heads = heads

        # For multi-layer GAT, adjust_hidden_dim to be divisible by heads if needed
        if n_layers > 1 and hidden_dim % self.heads != 0:
            # Round up to next multiple of heads to ensure proper concatenation
            adjusted_hidden_dim = (
                (hidden_dim + self.heads - 1) // self.heads
            ) * self.heads
            logger.warning(
                f"GAT: hidden_dim ({hidden_dim}) adjusted to {adjusted_hidden_dim} "
                f"to be divisible by heads ({self.heads}) for proper concatenation"
            )
            self.hidden_dim = adjusted_hidden_dim
        else:
            adjusted_hidden_dim = hidden_dim

        for i in range(n_layers):
            in_dim = input_dim if i == 0 else adjusted_hidden_dim

            if i < n_layers - 1:
                out_dim = adjusted_hidden_dim // self.heads
                concat = True
            else:
                out_dim = adjusted_hidden_dim
                concat = False

            self.convs.append(
                GATConv(
                    in_dim, out_dim, heads=self.heads, dropout=dropout, concat=concat
                )
            )

    def _get_specific_hyperparameters(self) -> dict[str, Any]:
        return {"heads": self.heads}

    def get_attention_weights(self, data: Data) -> dict[str, torch.Tensor]:
        """
        Extract attention weights from each GAT layer.

        Args:
            data: Input graph data

        Returns:
            Dictionary with attention weights: {'gnn_attention_layer_{i}': weights}
        """
        attention_weights: dict[str, torch.Tensor] = {}

        if data.x is None:
            return attention_weights

        x = data.x
        edge_index = data.edge_index

        for i, conv in enumerate(self.convs):
            try:
                # Extract attention weights from GAT layer
                result = conv(x, edge_index, return_attention_weights=True)
                if isinstance(result, tuple) and len(result) == 2:  # type: ignore
                    x_new, attention_info = result  # type: ignore
                    if isinstance(attention_info, tuple) and len(attention_info) == 2:  # type: ignore
                        _, att_weights = attention_info  # type: ignore
                        if isinstance(att_weights, torch.Tensor):
                            attention_weights[f"gnn_attention_layer_{i}"] = att_weights
                    elif isinstance(attention_info, torch.Tensor):
                        attention_weights[f"gnn_attention_layer_{i}"] = attention_info
                    x = x_new  # type: ignore
                else:
                    # Fallback: normal forward pass
                    x = conv(x, edge_index)
            except Exception:
                # Fallback: normal forward pass
                x = conv(x, edge_index)

            # Apply activation and dropout
            x = F.relu(x)  # type: ignore
            x = F.dropout(x, p=self.dropout, training=self.training)

        return attention_weights


class GATv2(GNN):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int | list[int],
        n_layers: int,
        dropout: float,
        **kwargs: Any,
    ):
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout,
            **kwargs,
        )

        _heads = cast(int, kwargs.get("heads", 1))

        # Handle both int and list[int] for hidden_dim
        if isinstance(hidden_dim, list):
            if len(hidden_dim) != n_layers:
                raise ValueError("GATv2: hidden_dim list length must match n_layers")
            hidden_dims = hidden_dim
        else:
            # If int, use the same dimension for all layers
            hidden_dims = [hidden_dim] * n_layers

        # Adjust hidden dimensions to be divisible by heads if needed
        adjusted_hidden_dims: list[int] = []
        for i, h_dim in enumerate(hidden_dims):
            if h_dim % _heads != 0:
                # Round up to next multiple of heads to ensure proper concatenation
                adjusted_h_dim = ((h_dim + _heads - 1) // _heads) * _heads
                logger.warning(
                    f"GATv2: hidden_dim[{i}] ({h_dim}) adjusted to {adjusted_h_dim} "
                    f"to be divisible by heads ({_heads}) for proper concatenation"
                )
                adjusted_hidden_dims.append(adjusted_h_dim)
            else:
                adjusted_hidden_dims.append(h_dim)

        # Store the adjusted dimensions
        self.hidden_dim = (
            adjusted_hidden_dims
            if isinstance(hidden_dim, list)
            else adjusted_hidden_dims[0]
        )
        self.heads = _heads  # Store for hyperparameters

        for i in range(n_layers):
            in_dim = input_dim if i == 0 else adjusted_hidden_dims[i - 1]
            current_hidden_dim = adjusted_hidden_dims[i]

            if i < n_layers - 1:
                out_dim = current_hidden_dim // _heads
                concat = True
            else:
                out_dim = current_hidden_dim
                concat = False

            self.convs.append(
                GATv2Conv(in_dim, out_dim, heads=_heads, dropout=dropout, concat=concat)
            )

    def _get_specific_hyperparameters(self) -> dict[str, Any]:
        return {"heads": self.heads}

    def get_attention_weights(self, data: Data) -> dict[str, torch.Tensor]:
        """
        Extract attention weights from each GATv2 layer.

        Args:
            data: Input graph data

        Returns:
            Dictionary with attention weights: {'gnn_attention_layer_{i}': weights}
        """
        attention_weights: dict[str, torch.Tensor] = {}

        if data.x is None:
            return attention_weights

        x = data.x
        edge_index = data.edge_index

        for i, conv in enumerate(self.convs):
            try:
                # Extract attention weights from GATv2 layer
                result = conv(x, edge_index, return_attention_weights=True)
                if isinstance(result, tuple) and len(result) == 2:  # type: ignore
                    x_new, attention_info = result  # type: ignore
                    if isinstance(attention_info, tuple) and len(attention_info) == 2:  # type: ignore
                        _, att_weights = attention_info  # type: ignore
                        if isinstance(att_weights, torch.Tensor):
                            attention_weights[f"gnn_attention_layer_{i}"] = att_weights
                    elif isinstance(attention_info, torch.Tensor):
                        attention_weights[f"gnn_attention_layer_{i}"] = attention_info
                    x = x_new  # type: ignore
                else:
                    # Fallback: normal forward pass
                    x = conv(x, edge_index)
            except Exception:
                # Fallback: normal forward pass
                x = conv(x, edge_index)

            # Apply activation and dropout
            x = F.relu(x)  # type: ignore
            x = F.dropout(x, p=self.dropout, training=self.training)

        return attention_weights


class SAGE(GNN):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_layers: int,
        dropout: float,
        **kwargs: Any,
    ):
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout,
            **kwargs,
        )

        for i in range(n_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            out_dim = hidden_dim
            self.convs.append(SAGEConv(in_channels=in_dim, out_channels=out_dim))


class EGNN(GNN):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_layers: int,
        dropout: float,
        **kwargs: Any,
    ):
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout,
            **kwargs,
        )
        self.pos_dim = int(cast(int, kwargs.get("pos_dim", 2)))
        self.proj = nn.Linear(input_dim, hidden_dim)

        for _ in range(n_layers):
            self.convs.append(
                EGNN_Sparse(
                    feats_dim=hidden_dim,
                    pos_dim=self.pos_dim,
                    dropout=float(dropout),
                )
            )

    def _get_specific_hyperparameters(self) -> dict[str, Any]:
        return {"pos_dim": self.pos_dim}

    def forward(self, data: Data) -> Data:
        x = self.proj(data.x)
        x_all = torch.cat([cast(torch.Tensor, data.pos), x], dim=-1)

        edge_index = data.edge_index
        for conv in self.convs:
            x_all = conv(x_all, edge_index)

        data.pos = x_all[:, : self.pos_dim]
        data.x = x_all[:, self.pos_dim :]
        return data


class CHIMERA(GNN):
    def __init__(
        self,
        input_dim: int,
        dropout: float,
        heads: int = 1,
        residual: bool = True,
        n_layers: int = 3,
        hidden_dim: list[int] = [128, 256, 512],
        **kwargs: Any,
    ):
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim[0],
            n_layers=n_layers,
            dropout=dropout,
            **kwargs,
        )
        self.heads = heads
        self.residual = residual
        self.hidden_dim = hidden_dim

        if len(hidden_dim) != self.n_layers:
            raise ValueError("CHIMERA: hidden_dim list length must match n_layers")

        self.blocks = nn.ModuleList()
        for i in range(n_layers):
            in_channels = input_dim if i == 0 else hidden_dim[i - 1] * self.heads
            out_channels = hidden_dim[i]
            self.blocks.append(
                CHIMERA_block(
                    in_channels,
                    out_channels,
                    heads=self.heads,
                    dropout=dropout,
                    residual=residual,
                )
            )

    def _get_specific_hyperparameters(self) -> dict[str, Any]:
        return {"heads": self.heads, "residual": self.residual}

    def forward(self, data: Data) -> Data:
        for block in self.blocks:
            data = block(data)
        return data


class CHIMERA_block(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int,
        dropout: float,
        residual: bool = True,
    ):
        super().__init__()  # type: ignore

        self.convs = nn.ModuleList()
        _in_channels = in_channels
        _out_channels = out_channels
        for _ in range(2):
            self.convs.append(
                GATv2Conv(
                    _in_channels,
                    _out_channels,
                    heads=heads,
                    concat=True,
                    dropout=dropout,
                    residual=residual,
                )
            )
            _in_channels = _out_channels

        self.norms = nn.ModuleList()
        for _ in range(2):
            self.norms.append(nn.BatchNorm1d(_out_channels * heads))

        self.pool = SAGPooling(_out_channels * heads, ratio=0.5)

    def forward(self, data: Data) -> Data:
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for i in range(2):
            x = self.convs[i](x, edge_index)
            x = F.elu(x)
            x = self.norms[i](x)
        x, edge_index, _, batch, _, _ = self.pool(x, edge_index, batch=batch)
        data.x = x
        data.edge_index = edge_index
        data.batch = batch
        return data


class SmallWorld(GNN):
    """
    SmallWorld GNN that creates additional connections between high-attention nodes.

    After each layer, uses SAGPooling to identify important nodes (top 1% by attention score)
    and creates additional edges between them, forming a small-world-like topology where
    important nodes are more densely connected.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_layers: int,
        dropout: float,
        top_k_ratio: float = 0.005,
        heads: int = 1,
        layer_type: Literal["GCN", "GAT", "SAGE"] = "SAGE",
        **kwargs: Any,
    ):
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout,
            **kwargs,
        )

        self.top_k_ratio = top_k_ratio
        self.heads = heads
        self.layer_type = layer_type

        # Adjust hidden_dim to be divisible by heads if needed (only for GAT)
        if layer_type == "GAT" and hidden_dim % heads != 0:
            adjusted_hidden_dim = ((hidden_dim + heads - 1) // heads) * heads
            logger.warning(
                f"SmallWorld: hidden_dim ({hidden_dim}) adjusted to {adjusted_hidden_dim} "
                f"to be divisible by heads ({heads}) for proper concatenation"
            )
            self.hidden_dim = adjusted_hidden_dim
        else:
            adjusted_hidden_dim = hidden_dim

        # Create convolution layers based on layer_type
        for i in range(n_layers):
            in_dim = input_dim if i == 0 else adjusted_hidden_dim

            if layer_type == "GAT":
                if i < n_layers - 1:
                    out_dim = adjusted_hidden_dim // heads
                    concat = True
                else:
                    out_dim = adjusted_hidden_dim
                    concat = False

                self.convs.append(
                    GATConv(
                        in_dim, out_dim, heads=heads, dropout=dropout, concat=concat
                    )
                )
            elif layer_type == "SAGE":
                out_dim = adjusted_hidden_dim
                self.convs.append(SAGEConv(in_channels=in_dim, out_channels=out_dim))
            elif layer_type == "GCN":
                out_dim = adjusted_hidden_dim
                self.convs.append(GCNConv(in_channels=in_dim, out_channels=out_dim))

        # Create SAGPooling layers for attention scoring (not for actual pooling)
        self.attention_pools = nn.ModuleList()
        for i in range(n_layers - 1):  # No pooling after last layer
            # The attention pool receives features AFTER the conv layer has been applied
            # For GAT with concat=True, output is hidden_dim (heads * out_dim)
            # For other layers, output is adjusted_hidden_dim
            pool_in_dim = adjusted_hidden_dim
            self.attention_pools.append(SAGPooling(pool_in_dim, ratio=1.0))

    def _get_specific_hyperparameters(self) -> dict[str, Any]:
        return {
            "top_k_ratio": self.top_k_ratio,
            "heads": self.heads,
            "layer_type": self.layer_type,
        }

    def _add_small_world_edges(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        attention_pool: SAGPooling,
    ) -> torch.Tensor:
        """
        Add edges between top-k nodes based on attention scores.

        Args:
            x: Node features
            edge_index: Current edge index
            batch: Batch assignment for nodes
            attention_pool: SAGPooling layer to compute attention scores

        Returns:
            Updated edge index with additional small-world connections
        """
        # Get attention scores from SAGPooling
        score = attention_pool.gnn(x, edge_index).view(-1)

        # Process each graph in the batch separately
        new_edges: list[torch.Tensor] = []
        for batch_id in cast(torch.Tensor, torch.unique(batch)):  # type: ignore
            mask = batch == batch_id
            batch_scores = score[mask]
            batch_indices = torch.where(mask)[0]

            # Get top-k nodes
            k = max(1, int(len(batch_scores) * self.top_k_ratio))
            _, top_k_local_indices = torch.topk(batch_scores, k)
            top_k_indices = batch_indices[top_k_local_indices]

            # Create fully connected edges among top-k nodes
            if len(top_k_indices) > 1:
                # Create all pairs of connections
                src = top_k_indices.repeat_interleave(len(top_k_indices))
                dst = top_k_indices.repeat(len(top_k_indices))
                # Remove self-loops
                mask_no_self_loop = src != dst
                src = src[mask_no_self_loop]
                dst = dst[mask_no_self_loop]

                # Add bidirectional edges
                batch_new_edges = torch.stack(
                    [torch.cat([src, dst]), torch.cat([dst, src])], dim=0
                )
                new_edges.append(batch_new_edges)

        if new_edges:
            # Concatenate all new edges
            all_new_edges = torch.cat(new_edges, dim=1)
            # Combine with existing edges and remove duplicates
            combined_edges = torch.cat([edge_index, all_new_edges], dim=1)
            # Remove duplicate edges
            combined_edges = cast(torch.Tensor, torch.unique(combined_edges, dim=1))  # type: ignore
            return combined_edges

        return edge_index

    def forward(self, data: Data) -> Data:
        x, edge_index = data.x, data.edge_index

        if x is None or edge_index is None:
            raise ValueError(
                "SmallWorld GNN requires node features (data.x) and edge index (data.edge_index) to be present."
            )

        if not hasattr(data, "batch") or data.batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        else:
            batch = data.batch

        for i, conv in enumerate(self.convs):
            # Apply GNN layer
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

            # Add small-world edges for next layer (except after last layer)
            if i < len(self.convs) - 1:
                edge_index = self._add_small_world_edges(
                    x,
                    edge_index,
                    batch,
                    self.attention_pools[i],  # type: ignore
                )

        data.x = x
        data.edge_index = edge_index
        return data


class SGFormer(GNN):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_layers: int,
        dropout: float = 0.25,
        heads: int = 1,
        alpha: float = 0.5,
        **kwargs: Any,
    ):
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout,
            **kwargs,
        )

        self.hidden_dim = hidden_dim
        self.heads = heads
        self.mlp = nn.Linear(input_dim, hidden_dim)
        self.global_attention = cast(
            nn.Module,
            SGFormerAttention(
                channels=hidden_dim, heads=heads, head_channels=hidden_dim // heads
            ),
        )
        self.alpha = alpha
        
        for _ in range(n_layers):
            self.convs.append(GCNConv(in_channels=hidden_dim, out_channels=hidden_dim))

    def forward(self, data: Data) -> Data:
        x, edge_index = data.x, data.edge_index

        # Project input to hidden dimension
        x = self.mlp(x)
        x = F.relu(x)

        # Global attention branch
        z_global = self.global_attention(x)

        # GNN branch
        z_gnn = x
        for conv in self.convs:
            z_gnn = conv(z_gnn, edge_index)
            z_gnn = F.relu(z_gnn)
            z_gnn = F.dropout(z_gnn, p=self.dropout, training=self.training)

        # Combine both branches (residual connection)
        z_out = self.alpha*z_global + (1-self.alpha)*z_gnn

        data.x = z_out
        return data
    
class SGFormerAttention(torch.nn.Module):
    r"""Implements the logic from the SGFormer paper's official repository
    (function `full_attention_conv`), adapted for non-batched input and
    numerical stability.

    Note: This logic is different from Equations (2) & (3) in the paper text.
    This implementation corresponds to:
    Z = (Q(K^T V) + N*V) / (Q(K^T * 1) + N)

    Args:
        channels (int): Size of each input sample (hidden_dim).
        heads (int, optional): Number of parallel attention heads.
        head_channels (int, optional): Size of each attention head.
        qkv_bias (bool, optional): If specified, add bias to query, key
            and value.
    """
    def __init__(
        self,
        channels: int,
        heads: int = 1,
        head_channels: int = 64,
        qkv_bias: bool = False,
    ) -> None:
        super().__init__() # type: ignore
        assert channels == heads * head_channels, \
            "channels must be equal to heads * head_channels"

        self.channels = channels
        self.heads = heads
        self.head_channels = head_channels
        self.epsilon = 1e-8  # For numerical stability

        self.q = torch.nn.Linear(channels, channels, bias=qkv_bias)
        self.k = torch.nn.Linear(channels, channels, bias=qkv_bias)
        self.v = torch.nn.Linear(channels, channels, bias=qkv_bias)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        # Note: 'mask' is unused in non-batched setup but kept for API.
        
        # x shape is [N, C_in]
        N, C_in = x.shape 

        # 1. Compute Q, K, V and reshape for multi-head
        # [N, C_in] -> [N, H, C_head]
        qs = self.q(x).reshape(N, self.heads, self.head_channels)
        ks = self.k(x).reshape(N, self.heads, self.head_channels)
        vs = self.v(x).reshape(N, self.heads, self.head_channels)

        # 2. Normalize Q and K
        # We normalize the entire (N, H, C_head) tensor by its Frobenius norm
        # This matches the original function's torch.norm(..., p=2)
        
        # *** STABILITY FIX 1: Add epsilon to denominator ***
        qs_norm = cast(torch.Tensor, torch.norm(qs, p='fro')) # type: ignore
        ks_norm = cast(torch.Tensor, torch.norm(ks, p='fro')) # type: ignore
        
        qs = qs / (qs_norm + self.epsilon)
        ks = ks / (ks_norm + self.epsilon)
        
        # Clamp normalized values to prevent extreme values
        qs = torch.clamp(qs, min=-10.0, max=10.0)
        ks = torch.clamp(ks, min=-10.0, max=10.0)
        
        # --- Logic from full_attention_conv ---

        # 3. Numerator
        # Original: kvs = torch.einsum("lhm,lhd->hmd", ks, vs)
        # 'l' is the node dim (N)
        kvs = torch.einsum("nhm,nhd->hmd", ks, vs) # [H, C_head, C_head]

        # Original: attention_num = torch.einsum("nhm,hmd->nhd", qs, kvs)
        attention_num = torch.einsum("nhm,hmd->nhd", qs, kvs) # [N, H, C_head]

        # Original: attention_num += N * vs
        # attention_num = attention_num + N * vs # [N, H, C_head]
        
        # Use a scaling factor to prevent overflow with large N
        # Scale down the entire formula by sqrt(N) to maintain numerical stability
        scale_factor = torch.sqrt(torch.tensor(float(N), device=vs.device, dtype=vs.dtype))
        attention_num = attention_num / scale_factor + (N / scale_factor) * vs # [N, H, C_head]

        # 4. Denominator
        # Original: ks_sum = torch.einsum("lhm,l->hm", ks, all_ones)
        # We can just sum over the node dimension (dim=0)
        ks_sum = torch.sum(ks, dim=0) # [H, C_head]

        # Original: attention_normalizer = torch.einsum("nhm,hm->nh", qs, ks_sum)
        attention_normalizer = torch.einsum("nhm,hm->nh", qs, ks_sum) # [N, H]

        # 5. Attentive aggregated results
        # Original: attention_normalizer = torch.unsqueeze(...)
        attention_normalizer = attention_normalizer.unsqueeze(-1) # [N, H, 1]

        # Original: attention_normalizer += torch.ones_like(...) * N
        # attention_normalizer = attention_normalizer + (N) # [N, H, 1]
        
        # Apply same scaling factor to denominator to maintain correct ratio
        attention_normalizer = attention_normalizer / scale_factor + (N / scale_factor) # Use broadcasting

        # *** STABILITY FIX 2: Add epsilon to denominator ***
        # Original: attn_output = attention_num / attention_normalizer
        attn_output = attention_num / (attention_normalizer + self.epsilon) # [N, H, C_head]

        # 6. Reshape back to [N, C_in]
        # [N, H, C_head] -> [N, H * C_head] -> [N, C_in]
        return attn_output.reshape(N, C_in)

    def reset_parameters(self):
        self.q.reset_parameters()
        self.k.reset_parameters()
        self.v.reset_parameters()

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'channels={self.channels}, '
                f'heads={self.heads}, '
                f'head_channels={self.head_channels})')
"""
Core attention extraction module for different MIL model types.

This module provides a unified interface for extracting attention weights from
various MIL models including CLAM, AttentionDeepMIL, GraphMIL, etc.

Attention Normalization:
The normalization is applied head-wise/class-wise along the instance dimension,
meaning each head or class gets normalized independently. This preserves the
relative importance within each head while ensuring all heads contribute equally
to visualization and analysis.

Supported normalization methods:
- min_max: Min-Max scaling to [0, 1] per head (default)
- z_score: Z-score standardization per head (mean=0, std=1)
- robust: Robust scaling per head using median and IQR
- softmax: Softmax normalization per head (creates probability distribution)
- sigmoid: Sigmoid activation per head to [0, 1] range
- none: No normalization applied
"""

import torch
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Protocol, cast
from torch_geometric.utils import add_self_loops  # type: ignore

from cellmil.interfaces.AttentionExplainerConfig import (
    AttentionExplainerConfig,
    Aggregation,
    Normalization,
)
from cellmil.utils import logger
import lightning as Pl


class AttentionModel(Protocol):
    """Protocol for models that support attention extraction."""

    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Extract attention weights from the model."""
        ...


class AttentionResult:
    """Container for attention extraction results."""

    def __init__(
        self,
        attention_weights: Dict[str, torch.Tensor],
        metadata: Dict[str, Any],
        model_type: str,
    ):
        self.attention_weights = attention_weights
        self.metadata = metadata
        self.model_type = model_type

    def get_attention(self, key: str) -> Optional[torch.Tensor]:
        """Get specific attention weights by key."""
        return self.attention_weights.get(key)

    def get_all_keys(self) -> list[str]:
        """Get all available attention keys."""
        return list(self.attention_weights.keys())

    def get_shape_info(self) -> Dict[str, tuple[int, ...]]:
        """Get shape information for all attention weights."""
        return {key: weights.shape for key, weights in self.attention_weights.items()}


class BaseAttentionExtractor(ABC):
    """Base class for attention extractors."""

    def __init__(self, config: AttentionExplainerConfig):
        self.config = config

    @abstractmethod
    def extract(self, model: Any, data: Any, **kwargs: Any) -> AttentionResult:
        """Extract attention weights from the model."""
        pass

    def _normalize_attention(self, attention: torch.Tensor) -> torch.Tensor:
        """
        Normalize attention weights using the configured normalization method.

        For multi-dimensional tensors (e.g., [n_heads, n_instances] or [n_classes, n_instances]),
        normalization is applied independently to each head/class along the instance dimension.
        """
        if not self.config.normalize_attention:
            return attention

        # Get the normalization method
        norm_method = self.config.normalization

        if norm_method == Normalization.none:
            return attention

        # Handle different tensor shapes
        if attention.dim() == 1:
            # Single dimension [n_instances] - normalize directly along the single dimension
            instance_dim = 0
            attention_work = attention
        elif attention.dim() == 2:
            # Two dimensions [n_heads/classes, n_instances] - normalize along last dim
            instance_dim = 1
            attention_work = attention
        else:
            # Higher dimensions - flatten all but last dimension, then normalize along last
            original_shape = attention.shape
            attention_work = attention.view(-1, original_shape[-1])  # [*, n_instances]
            instance_dim = 1

        if norm_method == Normalization.min_max:
            # Min-Max normalization to [0, 1] along instance dimension
            if attention.dim() == 1:
                # For 1D tensors, compute min/max directly
                min_val = attention_work.min()
                max_val = attention_work.max()
                range_val = max_val - min_val

                if range_val > 1e-8:
                    normalized = (attention_work - min_val) / range_val
                else:
                    normalized = torch.zeros_like(attention_work)
            else:
                # For multi-dimensional tensors, compute along instance dimension
                min_vals = attention_work.min(dim=instance_dim, keepdim=True)[0]
                max_vals = attention_work.max(dim=instance_dim, keepdim=True)[0]

                # Avoid division by zero - if min == max, set to zeros
                range_vals = max_vals - min_vals
                safe_range = torch.where(
                    range_vals > 1e-8, range_vals, torch.ones_like(range_vals)
                )
                normalized = torch.where(
                    range_vals > 1e-8,
                    (attention_work - min_vals) / safe_range,
                    torch.zeros_like(attention_work),
                )

        elif norm_method == Normalization.z_score:
            # Z-score standardization (mean=0, std=1) along instance dimension
            if attention.dim() == 1:
                # For 1D tensors, compute mean/std directly
                mean_val = attention_work.mean()
                std_val = attention_work.std()

                if std_val > 1e-8:
                    normalized = (attention_work - mean_val) / std_val
                else:
                    normalized = torch.zeros_like(attention_work)
            else:
                # For multi-dimensional tensors, compute along instance dimension
                mean_vals = attention_work.mean(dim=instance_dim, keepdim=True)
                std_vals = attention_work.std(dim=instance_dim, keepdim=True)

                # Avoid division by zero - if std == 0, set to zeros
                safe_std = torch.where(
                    std_vals > 1e-8, std_vals, torch.ones_like(std_vals)
                )
                normalized = torch.where(
                    std_vals > 1e-8,
                    (attention_work - mean_vals) / safe_std,
                    torch.zeros_like(attention_work),
                )

        elif norm_method == Normalization.robust:
            # Robust scaling using median and IQR along instance dimension
            if attention.dim() == 1:
                # For 1D tensors, compute median/quantiles directly
                median_val = attention_work.median()
                q75_val = attention_work.quantile(0.75)
                q25_val = attention_work.quantile(0.25)
                iqr_val = q75_val - q25_val

                if iqr_val > 1e-8:
                    normalized = (attention_work - median_val) / iqr_val
                else:
                    normalized = torch.zeros_like(attention_work)
            else:
                # For multi-dimensional tensors, compute along instance dimension
                median_vals = attention_work.median(dim=instance_dim, keepdim=True)[0]
                q75_vals = attention_work.quantile(0.75, dim=instance_dim, keepdim=True)
                q25_vals = attention_work.quantile(0.25, dim=instance_dim, keepdim=True)
                iqr_vals = q75_vals - q25_vals

                # Avoid division by zero - if IQR == 0, set to zeros
                safe_iqr = torch.where(
                    iqr_vals > 1e-8, iqr_vals, torch.ones_like(iqr_vals)
                )
                normalized = torch.where(
                    iqr_vals > 1e-8,
                    (attention_work - median_vals) / safe_iqr,
                    torch.zeros_like(attention_work),
                )

        elif norm_method == Normalization.softmax:
            # Softmax normalization (sum to 1) along instance dimension
            normalized = torch.softmax(attention_work, dim=instance_dim)

        elif norm_method == Normalization.sigmoid:
            # Sigmoid normalization to [0, 1] - applied element-wise
            normalized = torch.sigmoid(attention_work)

        else:
            raise ValueError(f"Unknown normalization method: {norm_method}")

        # Restore original shape if needed
        if attention.dim() > 2:
            normalized = normalized.view(attention.shape)

        return normalized


class CLAMAttentionExtractor(BaseAttentionExtractor):
    """Attention extractor for CLAM models."""

    def extract(self, model: Any, data: torch.Tensor, **kwargs: Any) -> AttentionResult:
        """
        Extract attention weights from CLAM model.

        Args:
            model: CLAM model instance
            data: Input tensor of shape [n_instances, feat_dim]

        Returns:
            AttentionResult with CLAM attention weights
        """
        logger.info("Extracting attention from CLAM model")

        try:
            # Get raw attention weights [n_classes, n_instances]
            attention_raw = model.get_attention_weights(data)

            attention_weights: dict[str, torch.Tensor] = {}
            metadata = {
                "n_classes": attention_raw.shape[0],
                "n_instances": attention_raw.shape[1],
                "has_multi_class": attention_raw.shape[0] > 1,
            }

            # Handle class-specific attention
            if self.config.class_index is not None:
                if self.config.class_index < attention_raw.shape[0]:
                    attention_weights["class_attention"] = self._normalize_attention(
                        attention_raw[
                            self.config.class_index : self.config.class_index + 1, :
                        ]
                    )
                    metadata["selected_class"] = self.config.class_index
                else:
                    logger.warning(
                        f"Class index {self.config.class_index} out of range for {attention_raw.shape[0]} classes"
                    )
                    attention_weights["class_attention"] = self._normalize_attention(
                        attention_raw[0:1, :]
                    )
            else:
                # Include all classes
                for i in range(attention_raw.shape[0]):
                    attention_weights[f"class_{i}_attention"] = (
                        self._normalize_attention(attention_raw[i : i + 1, :])
                    )

            return AttentionResult(attention_weights, metadata, "CLAM")

        except Exception as e:
            logger.error(f"Error extracting CLAM attention: {e}")
            raise


class AttentionDeepMILExtractor(BaseAttentionExtractor):
    """Attention extractor for AttentionDeepMIL models."""

    def extract(self, model: Any, data: torch.Tensor, **kwargs: Any) -> AttentionResult:
        """
        Extract attention weights from AttentionDeepMIL model.

        Args:
            model: AttentionDeepMIL model instance
            data: Input tensor of shape [n_instances, feat_dim]

        Returns:
            AttentionResult with AttentionDeepMIL attention weights
        """
        logger.info("Extracting attention from AttentionDeepMIL model")

        try:
            # Get raw attention weights [attention_branches, n_instances]
            attention_raw = model.get_attention_weights(data)

            attention_weights: dict[str, torch.Tensor] = {}
            metadata = {
                "n_heads": attention_raw.shape[0],
                "n_instances": attention_raw.shape[1],
                "has_multi_head": attention_raw.shape[0] > 1,
            }

            # Handle head-specific attention
            if self.config.attention_head is not None:
                if self.config.attention_head < attention_raw.shape[0]:
                    attention_weights["head_attention"] = self._normalize_attention(
                        attention_raw[
                            self.config.attention_head : self.config.attention_head + 1,
                            :,
                        ]
                    )
                    metadata["selected_head"] = self.config.attention_head
                else:
                    logger.warning(
                        f"Attention head {self.config.attention_head} out of range for {attention_raw.shape[0]} heads"
                    )
                    attention_weights["head_attention"] = self._normalize_attention(
                        attention_raw[0:1, :]
                    )
            else:
                if attention_raw.shape[0] == 1:
                    # Single head
                    attention_weights["head_attention"] = self._normalize_attention(
                        attention_raw
                    )
                else:
                    # Multiple heads - include mean and individual heads
                    attention_weights["mean_attention"] = self._normalize_attention(
                        attention_raw.mean(dim=0, keepdim=True)
                    )
                    for i in range(attention_raw.shape[0]):
                        attention_weights[f"head_{i}_attention"] = (
                            self._normalize_attention(attention_raw[i : i + 1, :])
                        )

            return AttentionResult(attention_weights, metadata, "AttentionDeepMIL")

        except Exception as e:
            logger.error(f"Error extracting AttentionDeepMIL attention: {e}")
            raise


class Head4TypeAttentionExtractor(BaseAttentionExtractor):
    """Attention extractor for Head4Type models."""

    def extract(
        self,
        model: Any,
        data: torch.Tensor,
        cell_types_tensor: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> AttentionResult:
        """
        Extract attention weights from Head4Type model.

        Args:
            model: Head4Type model instance
            data: Input tensor of shape [n_instances, feat_dim]
            cell_types_tensor: Cell types tensor of shape [n_instances, n_cell_types] (required for Head4Type)

        Returns:
            AttentionResult with Head4Type attention weights (one entry per head)
        """
        logger.info("Extracting attention from Head4Type model")

        if cell_types_tensor is None:
            raise ValueError("cell_types_tensor is required for Head4Type model")

        try:
            model.eval()
            with torch.no_grad():
                # Get raw attention weights from model
                # Head4Type returns attention of shape [n_heads, n_instances]
                attention_raw = model.get_attention_weights(data, cell_types_tensor)
                
                # Detailed debugging
                logger.info(f"Raw Head4Type attention shape: {attention_raw.shape}")
                attention_sum_by_head = attention_raw.sum(dim=1)
                logger.info(f"Sum of RAW attention weights by head: {attention_sum_by_head}")
                
                # Check for zeros/NaNs in raw attention
                for i in range(attention_raw.shape[0]):
                    head_data = attention_raw[i]
                    num_zeros = (head_data == 0).sum().item()
                    num_nonzeros = (head_data != 0).sum().item()
                    logger.info(f"  Head {i}: {num_nonzeros} non-zero, {num_zeros} zero values. Min: {head_data.min():.6f}, Max: {head_data.max():.6f}")

            attention_weights: dict[str, torch.Tensor] = {}
            metadata = {
                "n_heads": attention_raw.shape[0],
                "n_instances": attention_raw.shape[1],
                "has_multi_head": attention_raw.shape[0] > 1,
            }

            # Handle head-specific attention
            if self.config.attention_head is not None:
                if self.config.attention_head < attention_raw.shape[0]:
                    attention_weights["head_attention"] = self._normalize_attention(
                        attention_raw[
                            self.config.attention_head : self.config.attention_head + 1,
                            :,
                        ]
                    )
                    metadata["selected_head"] = self.config.attention_head
                else:
                    logger.warning(
                        f"Attention head {self.config.attention_head} out of range for {attention_raw.shape[0]} heads"
                    )
                    attention_weights["head_attention"] = self._normalize_attention(
                        attention_raw[0:1, :]
                    )
            else:
                if attention_raw.shape[0] == 1:
                    # Single head
                    attention_weights["head_attention"] = self._normalize_attention(
                        attention_raw
                    )
                else:
                    # Multiple heads - include mean and individual heads
                    mean_before_norm = attention_raw.mean(dim=0, keepdim=True)
                    logger.info(f"Mean attention BEFORE normalization - sum: {mean_before_norm.sum():.6f}")
                    attention_weights["mean_attention"] = self._normalize_attention(mean_before_norm)
                    logger.info(f"Mean attention AFTER normalization - sum: {attention_weights['mean_attention'].sum():.6f}")
                    
                    for i in range(attention_raw.shape[0]):
                        head_before_norm = attention_raw[i : i + 1, :]
                        logger.info(f"Head {i} BEFORE normalization - sum: {head_before_norm.sum():.6f}, min: {head_before_norm.min():.6f}, max: {head_before_norm.max():.6f}")
                        normalized_head = self._normalize_attention(head_before_norm)
                        logger.info(f"Head {i} AFTER normalization - sum: {normalized_head.sum():.6f}, min: {normalized_head.min():.6f}, max: {normalized_head.max():.6f}")
                        attention_weights[f"head_{i}_attention"] = normalized_head

            return AttentionResult(attention_weights, metadata, "Head4Type")

        except Exception as e:
            logger.error(f"Error extracting Head4Type attention: {e}")
            raise


class GraphMILAttentionExtractor(BaseAttentionExtractor):
    """Attention extractor for GraphMIL models."""

    def extract(self, model: Any, data: Any, **kwargs: Any) -> AttentionResult:
        """
        Extract attention weights from GraphMIL model.

        Args:
            model: GraphMIL model instance
            data: PyTorch Geometric Data object

        Returns:
            AttentionResult with GraphMIL attention weights
        """
        logger.info("Extracting attention from GraphMIL model")

        try:
            # Get all available attention weights
            attention_raw = model.get_attention_weights(data)

            attention_weights: dict[str, torch.Tensor] = {}
            metadata: dict[str, Any] = {
                "has_gnn_attention": any(
                    "gnn_attention" in key for key in attention_raw.keys()
                ),
                "has_pooling_attention": any(
                    "pooling_attention" in key for key in attention_raw.keys()
                ),
                "gnn_layers": len(
                    [key for key in attention_raw.keys() if "gnn_attention" in key]
                ),
                "edge_index": data.edge_index if hasattr(data, "edge_index") else None,
                "num_nodes": data.x.shape[0] if hasattr(data, "x") else None,
            }

            # Apply aggregation strategy
            attention_weights = self._apply_aggregation(attention_raw, metadata)

            return AttentionResult(
                attention_weights, metadata, model.__class__.__name__
            )

        except Exception as e:
            logger.error(f"Error extracting GraphMIL attention: {e}")
            raise

    def _apply_aggregation(
        self, attention_raw: Dict[str, torch.Tensor], metadata: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        """Apply the configured aggregation strategy to GraphMIL attention."""

        aggregation = self.config.attention_aggregation
        attention_weights: dict[str, torch.Tensor] = {}

        # Separate GNN and pooling attention
        gnn_attention = {k: v for k, v in attention_raw.items() if "gnn_attention" in k}
        pooling_attention = {
            k: v for k, v in attention_raw.items() if "pooling_attention" in k
        }

        if aggregation == Aggregation.pooling_only:
            if pooling_attention:
                for key, weights in pooling_attention.items():
                    attention_weights[key] = self._normalize_attention(weights)

                # TODO: Send all the heads not only the average
                # if current_attention.shape[0] > 1:
                # # Store individual head results for analysis
                # for head_idx in range(current_attention.shape[0]):
                #     head_attention = current_attention[head_idx : head_idx + 1]
                #     attention_weights[f"pooling_attention_head_{head_idx}"] = head_attention
            else:
                logger.warning("No pooling attention found")

        elif aggregation == Aggregation.gnn_only:
            if gnn_attention:
                for key, weights in gnn_attention.items():
                    attention_weights[key] = self._normalize_attention(weights)
            else:
                logger.warning("No GNN attention found")

        elif aggregation == Aggregation.gnn_layer:
            layer_index = self.config.gnn_layer_index or 0
            target_key = f"gnn_attention_layer_{layer_index}"

            if target_key in gnn_attention:
                attention_weights["selected_gnn_layer"] = self._normalize_attention(
                    gnn_attention[target_key]
                )
            else:
                logger.warning(
                    f"GNN layer {layer_index} not found. Available: {list(gnn_attention.keys())}"
                )
                if gnn_attention:
                    # Fallback to first available layer
                    first_key = list(gnn_attention.keys())[0]
                    attention_weights["selected_gnn_layer"] = self._normalize_attention(
                        gnn_attention[first_key]
                    )

        elif aggregation == Aggregation.random_walk:
            # Random walk: a = p.T @ P_(l) @ P_(l-1) @ ... @ P_(1)
            # where p is pooling attention (initial state) and P_i are GNN layers (transitions)

            if not pooling_attention:
                logger.warning(
                    "No pooling attention found for random walk initialization"
                )
                return attention_weights

            if not gnn_attention:
                logger.warning("No GNN attention found for random walk transitions")
                # Fallback to pooling only
                for key, weights in pooling_attention.items():
                    attention_weights[key] = self._normalize_attention(weights)
                return attention_weights

            # Get pooling attention as initial state probabilities
            pooling_key = list(pooling_attention.keys())[0]
            initial_state = pooling_attention[
                pooling_key
            ]  # Shape: [heads, n_nodes] or [n_nodes]

            # Handle multi-head pooling attention - keep all heads for proper probability distributions
            if initial_state.dim() == 1:
                initial_state = initial_state.unsqueeze(0)  # [1, n_nodes]
            elif initial_state.dim() == 2:
                # Keep as [heads, n_nodes] to maintain separate probability distributions
                pass
            else:
                raise ValueError(
                    f"Unexpected pooling attention shape: {initial_state.shape}"
                )

            # Validate that pooling attention sums to approximately 1 for each head
            pooling_sum = initial_state.sum(dim=-1)  # [heads]
            if not torch.allclose(pooling_sum, torch.ones_like(pooling_sum), atol=1e-4):
                logger.warning(
                    f"Pooling attention doesn't sum to 1 for all heads: {pooling_sum}"
                )
                # Normalize to sum to 1 for each head
                initial_state = initial_state / pooling_sum.unsqueeze(-1)

            # Sort GNN layers from last to first for proper random walk order
            gnn_keys = sorted(
                gnn_attention.keys(), key=lambda x: int(x.split("_")[-1]), reverse=True
            )

            # Perform random walk: multiply through layers from last to first
            current_attention = initial_state

            # We need edge_index to convert edge attention to dense matrices
            edge_index = metadata.get("edge_index")
            num_nodes = metadata.get("num_nodes")

            if edge_index is None or num_nodes is None:
                raise ValueError(
                    "edge_index and num_nodes are required in metadata for random walk aggregation"
                )

            original_num_edges = edge_index.shape[1]
            expected_num_edges_with_self_loops = original_num_edges + num_nodes

            layer_details: list[str] = []

            for layer_key in gnn_keys:
                gnn_att_weights = gnn_attention[
                    layer_key
                ]  # Shape: [num_edges_with_self_loops, num_heads]

                logger.info(
                    f"Processing {layer_key}: {gnn_att_weights.shape[0]} attention weights, "
                    f"{original_num_edges} original edges, {num_nodes} nodes"
                )

                if gnn_att_weights.dim() == 1:
                    g = gnn_att_weights.unsqueeze(1)
                elif gnn_att_weights.dim() == 2:
                    g = gnn_att_weights
                else:
                    raise ValueError(
                        f"Unexpected GNN attention shape for {layer_key}: {gnn_att_weights.shape}"
                    )

                # detect whether first axis is edges or heads
                if (
                    g.shape[0] == original_num_edges
                    or g.shape[0] == expected_num_edges_with_self_loops
                ):
                    # g is [num_edges, num_heads] -> good
                    att_edges_heads = g
                elif (
                    g.shape[1] == original_num_edges
                    or g.shape[1] == expected_num_edges_with_self_loops
                ):
                    # transpose to [num_edges, num_heads]
                    att_edges_heads = g.t()
                else:
                    raise ValueError(
                        f"Cannot deduce edge/head orientation for {layer_key}: got shape {g.shape}, "
                        f"expected {original_num_edges} or {expected_num_edges_with_self_loops} in one axis"
                    )

                if att_edges_heads.shape[1] > 1:
                    # average over heads (alternatives: sum, keep per-head and propagate per-head)
                    edge_attention = att_edges_heads.mean(dim=1)  # [num_edges]
                else:
                    edge_attention = att_edges_heads.squeeze(1)  # [num_edges]

                # Validate that attention weights are non-negative
                if (edge_attention < -1e-8).any():
                    raise ValueError(f"Negative attention weights found in {layer_key}")

                if att_edges_heads.shape[0] == expected_num_edges_with_self_loops:
                    active_edge_index, _ = add_self_loops(
                        edge_index, num_nodes=num_nodes
                    )
                elif att_edges_heads.shape[0] == original_num_edges:
                    active_edge_index = edge_index
                else:
                    raise ValueError(
                        f"GNN attention for {layer_key} has {att_edges_heads.shape[0]} weights, expected "
                        f"{original_num_edges} or {expected_num_edges_with_self_loops}."
                    )

                # Move index & values to same device
                device = edge_attention.device
                active_edge_index = active_edge_index.to(device)
                edge_attention = edge_attention.to(device)

                target_nodes = active_edge_index[
                    1
                ]  # convention: edge_index = [source, target]
                col_sums = torch.zeros(
                    num_nodes, device=device, dtype=edge_attention.dtype
                )
                col_sums.scatter_add_(
                    0, target_nodes, edge_attention
                )  # col_sums[i] = sum_{j->i} alpha_{j,i}

                # If the per-destination sums are not 1 (e.g., raw scores) normalize them.
                if not torch.allclose(col_sums, torch.ones_like(col_sums), atol=1e-4):
                    # avoid dividing by zero
                    col_sums_safe = col_sums.clone()
                    col_sums_safe[col_sums_safe <= 1e-12] = 1.0
                    col_sums_for_edges = col_sums_safe.gather(0, target_nodes)
                    normalized_edge_attention = edge_attention / col_sums_for_edges
                    logger.info(
                        f"Normalized edge attention for {layer_key} (per-target sums not 1)"
                    )
                else:
                    normalized_edge_attention = edge_attention

                indices = active_edge_index.long()  # shape [2, num_edges]
                attention_matrix = torch.sparse_coo_tensor(  # type: ignore
                    indices,
                    normalized_edge_attention,
                    size=(num_nodes, num_nodes),
                    dtype=normalized_edge_attention.dtype,
                    device=device,
                )
                transition_matrix = attention_matrix.t()
                # transition_matrix = attention_matrix

                current_attention = cast(
                    torch.Tensor,
                    torch.sparse.mm(transition_matrix, current_attention.T).T, # type: ignore
                )  # [heads, n_nodes] 

                # Check if each row in current_attention sums to 1 (within tolerance)
                row_sums = current_attention.sum(dim=1, keepdim=True)  # [heads, 1]
                if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6):
                    logger.warning(
                        f"Rows in current_attention do not sum to 1. Min: {row_sums.min().item()}, Max: {row_sums.max().item()}"
                    )
                    row_sums = torch.where(
                        row_sums > 1e-12, row_sums, torch.ones_like(row_sums)
                    )
                    current_attention = current_attention / row_sums

                # attention_weights[f"initial_state_after_{layer_key}"] = (
                #     self._normalize_attention(current_attention)
                # )

                layer_details.append(f"Applied {layer_key}")

            # Final random walk attention - take mean across heads at the end
            if current_attention.shape[0] > 1:
                # Multiple heads: take mean to get final attention
                final_attention = current_attention.mean(
                    dim=0, keepdim=True
                )  # [1, num_nodes]
            else:
                # Single head: keep as is
                final_attention = current_attention

            # Store the raw random walk result (preserves probability distribution)
            attention_weights["random_walk_attention"] = self._normalize_attention(
                final_attention
            )

            # Store intermediate results for debugging/analysis
            attention_weights["initial_state"] = self._normalize_attention(
                initial_state
            )
            if current_attention.shape[0] > 1:
                # Store individual head results for analysis
                for head_idx in range(current_attention.shape[0]):
                    head_attention = current_attention[head_idx : head_idx + 1]
                    attention_weights[f"random_walk_head_{head_idx}"] = (
                        self._normalize_attention(head_attention)
                    )

            # Add metadata about the random walk process
            metadata["random_walk_layers"] = layer_details
            metadata["num_walk_steps"] = len(gnn_keys)
            metadata["num_heads"] = current_attention.shape[0]

            logger.info(
                f"Random walk completed through {len(gnn_keys)} GNN layers with {current_attention.shape[0]} heads"
            )

        return attention_weights


class AttentionExtractorFactory:
    """Factory for creating appropriate attention extractors."""

    @staticmethod
    def create_extractor(
        model: Pl.LightningModule, config: AttentionExplainerConfig
    ) -> BaseAttentionExtractor:
        """Create the appropriate attention extractor for the model type."""

        extractors: dict[str, type[BaseAttentionExtractor]] = {
            "litclam": CLAMAttentionExtractor,  # Support Lightning model names
            "litattentiondeepmil": AttentionDeepMILExtractor,  # Support Lightning model names
            "litgraphmil": GraphMILAttentionExtractor,  # Support Lightning model names
            "lithead4type": Head4TypeAttentionExtractor,
        }

        if model.__class__.__name__.lower() in extractors:
            return extractors[model.__class__.__name__.lower()](config)
        else:
            raise ValueError(
                f"Unsupported model type: {model.__class__.__name__}. Supported types: {list(extractors.keys())}"
            )

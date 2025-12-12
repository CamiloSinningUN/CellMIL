# -*- coding: utf-8 -*-
# HistoBistro - Transformer for MIL classification.
#
# References:
# Transformer-based biomarker prediction from colorectal cancer histology: A large-scale multicentric study
# Wagner, Sophia J et al., Cancer Cell, Elsevier
# DOI: https://doi.org/10.1016/j.ccell.2023.02.002

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from einops import rearrange
from typing import Literal, Any, cast


class Attention(nn.Module):
    """Multi-head self-attention mechanism for transformers.

    This class implements the standard multi-head self-attention as described in
    "Attention Is All You Need" (Vaswani et al., 2017). It includes methods for
    extracting and saving attention maps for interpretability.

    Args:
        dim: Input dimension of the features.
        heads: Number of attention heads.
        dim_head: Dimension of each attention head. If not provided, will be calculated as dim / heads.
        dropout: Dropout rate applied to the output projection.
    """

    def __init__(
        self,
        dim: int = 512,
        heads: int = 8,
        dim_head: int = 512 // 8,
        dropout: float = 0.1,
    ):
        super().__init__()  # type: ignore
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, register_hook: bool = False):
        """Forward pass for self-attention.

        Args:
            x: Input tensor of shape [batch_size, num_tokens, dim].
            register_hook: If True, registers a gradient hook on the attention map for interpretability.

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, num_tokens, dim].
        """
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        # save self-attention maps
        self.save_attention_map(attn)
        if register_hook:
            attn.register_hook(self.save_attn_gradients)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)

    def save_attn_gradients(self, attn_gradients: torch.Tensor):
        """Save attention gradients during backpropagation for interpretability.

        Args:
            attn_gradients: Gradient tensor for attention maps.
        """
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        """Retrieve saved attention gradients.

        Returns:
            torch.Tensor: The saved attention gradients.
        """
        return self.attn_gradients

    def save_attention_map(self, attention_map: torch.Tensor):
        """Save attention map from the forward pass.

        Args:
            attention_map: Attention map tensor from the forward pass.
        """
        self.attention_map = attention_map

    def get_attention_map(self):
        """Retrieve the saved attention map.

        Returns:
            torch.Tensor: The saved attention map.
        """
        return self.attention_map

    def get_self_attention(self, x: torch.Tensor):
        """Calculate self-attention maps without performing the full forward pass.

        This is useful for visualization and interpretability.

        Args:
            x: Input tensor of shape [batch_size, num_tokens, dim].

        Returns:
            torch.Tensor: Self-attention map of shape [batch_size, heads, num_tokens, num_tokens].
        """
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, _ = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        return attn


class FeedForward(nn.Module):
    """Feed-forward network used in transformer blocks.

    This implements a standard MLP with GELU activation and dropout,
    as commonly used in transformer architectures.

    Args:
        dim: Input dimension.
        hidden_dim: Hidden dimension, typically 2-4x the input dimension.
        dropout: Dropout rate applied after each layer.
    """

    def __init__(self, dim: int = 512, hidden_dim: int = 1024, dropout: float = 0.1):
        super().__init__()  # type: ignore
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor):
        """Forward pass through the feed-forward network.

        Args:
            x: Input tensor of shape [batch_size, seq_length, dim].

        Returns:
            torch.Tensor: Output tensor with the same shape as input.
        """
        return self.net(x)


class PreNorm(nn.Module):
    """Layer normalization module that applies normalization before a function.

    This implements the "Pre-LN" (Pre-LayerNorm) variant of transformers,
    which applies layer normalization before the attention or feed-forward operations.
    This variant is known to be more stable during training compared to Post-LN.

    Args:
        dim: Input dimension for the main tensor.
        fn: Function to apply after normalization.
        context_dim: Optional dimension for a context tensor that will also be normalized.
    """

    def __init__(self, dim: int, fn: nn.Module, context_dim: int | None = None):
        super().__init__()  # type: ignore
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = (
            nn.LayerNorm(context_dim) if context_dim is not None else None
        )

    def forward(self, x: torch.Tensor, **kwargs: dict[str, Any]):
        """Apply normalization and then the function.

        Args:
            x: Input tensor.
            **kwargs: Additional keyword arguments passed to the function.
                May include a 'context' tensor if context_dim was provided.

        Returns:
            torch.Tensor: Output after normalization and function application.
        """
        x = self.norm(x)

        if self.norm_context is not None:
            context = kwargs["context"]
            normed_context = self.norm_context(context)
            kwargs.update(context=normed_context)

        return self.fn(x, **kwargs)


class TransformerBlocks(nn.Module):
    """Stack of transformer encoder blocks.

    Each block consists of a multi-head self-attention layer followed by a feed-forward network,
    both with residual connections and layer normalization.

    Args:
        dim: Input dimension.
        depth: Number of transformer blocks.
        heads: Number of attention heads.
        dim_head: Dimension of each attention head.
        mlp_dim: Hidden dimension of the feed-forward network.
        dropout: Dropout rate applied in both attention and feed-forward.
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()  # type: ignore
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim,
                            Attention(
                                dim, heads=heads, dim_head=dim_head, dropout=dropout
                            ),
                        ),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                    ]
                )
            )

    def forward(self, x: torch.Tensor, register_hook: bool = False):
        """Forward pass through the transformer blocks.

        Args:
            x: Input tensor of shape [batch_size, seq_length, dim].
            register_hook: Whether to register gradient hooks for interpretability.

        Returns:
            torch.Tensor: Output tensor after passing through all transformer blocks.
        """
        for attn, ff in self.layers:  # type: ignore
            x = cast(torch.Tensor, attn(x, register_hook=register_hook) + x)
            x = cast(torch.Tensor, ff(x) + x)
        return x


class HistoBistro(nn.Module):
    """HistoBistro - Histopathology Bi-level Transformer for MIL classification.

    This is the main model class implementing the HistoBistro architecture,
    a transformer-based model for multiple instance learning (MIL) in histopathology images.
    The model processes a bag of instances (patches/cells) and makes a bag-level prediction.
    It includes interpretability methods to identify important instances.

    Args:
        num_classes: Number of output classes for classification.
        input_dim: Input dimension of instance features.
        dim: Internal dimension for transformer processing.
        depth: Number of transformer blocks.
        heads: Number of attention heads.
        mlp_dim: Hidden dimension of the feed-forward network.
        pool: Pooling strategy ('cls' for class token or 'mean' for mean pooling).
        dim_head: Dimension of each attention head.
        dropout: Dropout rate in transformer blocks.
        emb_dropout: Dropout rate applied to the input embeddings.
        pos_enc: Optional positional encoding module.
    """

    def __init__(
        self,
        num_classes: int,
        input_dim: int = 2048,
        dim: int = 512,
        depth: int = 2,
        heads: int = 8,
        mlp_dim: int = 512,
        pool: Literal["cls", "mean"] = "cls",
        dim_head: int = 64,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
        pos_enc: nn.Module | None = None,
    ):
        super(HistoBistro, self).__init__()  # type: ignore
        assert pool in {"cls", "mean"}, (
            "pool type must be either cls (class token) or mean (mean pooling)"
        )

        self.projection = nn.Sequential(
            nn.Linear(input_dim, heads * dim_head, bias=True), nn.ReLU()
        )
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(mlp_dim), nn.Linear(mlp_dim, num_classes)
        )
        self.transformer = TransformerBlocks(
            dim, depth, heads, dim_head, mlp_dim, dropout
        )

        self.pool = pool
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(emb_dropout)

        self.pos_enc = pos_enc

        self.gradients: torch.Tensor | None = None

    def save_gradient(self, grad: torch.Tensor) -> None:
        """Save gradients during backpropagation for interpretability.

        Args:
            grad: The gradient tensor to save

        The gradient is processed to obtain a single importance value per instance
        by taking the mean across the feature dimension.
        """
        # Take the mean across the feature dimension (last dimension)
        # This gives us a [batch_size, num_instances] tensor
        self.gradients = grad.mean(dim=-1)

    def get_instance_gradients(self) -> torch.Tensor | None:
        """Return the saved gradients for instance importance.

        Returns:
            The gradients for each instance, if available
        """
        return self.gradients

    def get_normalized_gradients(self) -> torch.Tensor | None:
        """Return the absolute and normalized gradients for instance importance.

        Returns:
            Normalized gradients (values between 0 and 1) for each instance, if available
        """
        if self.gradients is None:
            return None

        # Take absolute values since both positive and negative gradients
        # indicate importance (direction depends on class)
        abs_grads = torch.abs(self.gradients)

        # Normalize per bag (slide) to get relative importance within each bag
        normalized_grads = torch.zeros_like(abs_grads)
        for i in range(abs_grads.shape[0]):  # For each bag/sample in batch
            # Min-max normalization per bag
            bag_grads = abs_grads[i]
            if bag_grads.max() > bag_grads.min():
                normalized_grads[i] = (bag_grads - bag_grads.min()) / (
                    bag_grads.max() - bag_grads.min()
                )
            else:
                # Handle edge case where all gradients are the same
                normalized_grads[i] = torch.ones_like(bag_grads)

        return normalized_grads

    def forward(
        self,
        x: torch.Tensor,
        coords: torch.Tensor | None = None,
        register_hook: bool = False,
    ):
        """Forward pass through the HistoBistro model.

        Args:
            x: Input tensor of shape [batch_size, num_instances, input_dim] containing instance features.
            coords: Optional tensor of shape [batch_size, num_instances, 2] containing spatial coordinates.
            register_hook: Whether to register gradient hooks for interpretability.

        Returns:
            tuple: A tuple containing:
                - logits: Raw classification scores of shape [batch_size, num_classes]
                - Y_prob: Softmax probabilities of shape [batch_size, num_classes]
                - Y_hat: Predicted class indices of shape [batch_size]
                - gradients: Instance gradients for interpretability (or None)
                - results_dict: Dictionary with additional results for analysis
        """
        b, _, _ = x.shape  # b: batch size

        # For interpretability, we need the features to have gradients
        if register_hook and not x.requires_grad:
            x.requires_grad_(True)  # In-place setting of requires_grad
            x.register_hook(lambda grad: self.save_gradient(grad))  # type: ignore

        # Continue with normal forward pass
        features = self.projection(x)

        if self.pos_enc and coords is not None:
            features = features + self.pos_enc(coords)

        if self.pool == "cls":
            cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=b)
            features = torch.cat((cls_tokens, features), dim=1)

        features = self.dropout(features)
        features = self.transformer(features, register_hook=register_hook)
        features = features.mean(dim=1) if self.pool == "mean" else features[:, 0]

        logits = self.mlp_head(self.norm(features))
        Y_hat = torch.argmax(logits, dim=1)
        Y_prob = F.softmax(logits, dim=1)

        # Store any additional results for further analysis
        results_dict: dict[str, torch.Tensor] = {}
        if self.gradients is not None:
            results_dict["raw_gradients"] = self.gradients

            # Get normalized gradients for easier interpretation
            norm_grads = self.get_normalized_gradients()
            if norm_grads is not None:
                results_dict["norm_gradients"] = norm_grads

        # Return the instance gradients for interpretability
        return logits, Y_prob, Y_hat, self.gradients, results_dict

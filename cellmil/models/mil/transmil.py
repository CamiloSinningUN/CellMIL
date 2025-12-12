# -*- coding: utf-8 -*-
# TransMIL Model Implementation
#
# References:
# Transmil: Transformer based correlated multiple instance learning for whole slide image classification
# Shao, Zhuchen et al., Advances in Neural Information Processing Systems, 2021
# DOI: https://proceedings.neurips.cc/paper/2021/hash/10c272d06794d3e5785d5e7c5356e9ff-Abstract.html


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nystrom_attention import NystromAttention  # type: ignore


class TransLayer(nn.Module):
    """Transformer Layer with Nystrom Attention.

    This layer implements a transformer block using Nystrom Attention, which is an efficient
    approximation of the standard self-attention mechanism. It's particularly useful for
    processing long sequences as it reduces the computational complexity from O(n²) to O(n).

    Args:
        norm_layer (type[nn.LayerNorm], optional): Normalization layer class. Defaults to nn.LayerNorm.
        dim (int, optional): Feature dimension. Defaults to 512.
    """

    def __init__(self, norm_layer: type[nn.LayerNorm] = nn.LayerNorm, dim: int = 512):
        super().__init__()  # type: ignore
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim=dim,
            dim_head=dim // 8,
            heads=8,
            num_landmarks=dim // 2,  # number of landmarks
            pinv_iterations=6,  # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual=True,  # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1,
        )

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for the transformer layer.

        Args:
            x (torch.Tensor): Input tensor of shape [B, N, C] where B is batch size,
                             N is sequence length, and C is feature dimension.
            return_attention (bool, optional): Whether to return attention maps. Defaults to False.

        Returns:
            torch.Tensor | tuple[torch.Tensor, torch.Tensor]: If return_attention is False,
                returns the output tensor of the same shape as input. If True, returns a tuple
                containing the output tensor and the attention map.
        """
        norm_x = self.norm(x)
        output, attention = self.attn(norm_x, return_attn=True)
        x = x + output

        if return_attention:
            return x, attention
        return x


class PPEG(nn.Module):
    """Pyramid Position Encoding Generator.

    PPEG is a positional encoding module that uses convolutional layers with different
    kernel sizes to capture positional information at multiple scales. It transforms
    tokens into a 2D spatial grid, applies convolutional operations, and reshapes
    back to the sequence format.

    This module helps the transformer model to be aware of the spatial relationships
    between tokens, which is crucial for vision tasks.

    Args:
        dim (int, optional): Feature dimension. Defaults to 512.
    """

    def __init__(self, dim: int = 512):
        super(PPEG, self).__init__()  # type: ignore
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7 // 2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5 // 2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3 // 2, groups=dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """Forward pass for the position encoding generator.

        Args:
            x (torch.Tensor): Input tensor of shape [B, N, C] where B is batch size,
                            N is sequence length (including class token), and C is feature dimension.
            H (int): Height of the feature map when arranged in a 2D grid.
            W (int): Width of the feature map when arranged in a 2D grid.

        Returns:
            torch.Tensor: Output tensor with positional encoding information added,
                        same shape as input [B, N, C].
        """
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat) + cnn_feat + self.proj1(cnn_feat) + self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


class TransMIL(nn.Module):
    """Transformer-based Multiple Instance Learning model.

    TransMIL is a model for MIL tasks using transformers, as presented in the paper
    "TransMIL: Transformer based Correlated Multiple Instance Learning for Whole Slide Image Classification".
    It processes a bag of instances (e.g., patches from a whole slide image) using
    transformer architecture with Nystrom attention to efficiently handle large sets of instances.

    The model includes:
    - A learnable class token similar to ViT
    - Positional encoding through a convolutional approach (PPEG)
    - Multiple transformer layers with Nystrom attention

    Args:
        n_classes (int): Number of output classes for classification.
    """

    def __init__(self, n_classes: int, embed_dim: int = 1024):
        super(TransMIL, self).__init__()  # type: ignore
        d = 512
        self.pos_layer = PPEG(dim=d)
        self._fc1 = nn.Sequential(nn.Linear(embed_dim, d), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, d))
        self.n_classes = n_classes
        self.layer1 = TransLayer(dim=d)
        self.layer2 = TransLayer(dim=d)
        self.norm = nn.LayerNorm(d)
        self._fc2 = nn.Linear(d, self.n_classes)
        # TODO: Review this, embeding is much smaller

    def forward(
        self, data: torch.Tensor
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]
    ]:
        """Forward pass for the TransMIL model.

        The model processes a bag of instances (features) using a transformer-based architecture.
        It first projects the features to a lower dimension, adds a class token, applies
        transformer layers with positional encoding, and finally produces classification outputs.

        Args:
            data (torch.Tensor): Input tensor of shape [B, n, D] where B is batch size,
                               n is the number of instances in each bag, and D is the
                               input feature dimension.

        Returns:
            tuple containing:
                - logits (torch.Tensor): Raw classification scores [B, n_classes]
                - Y_prob (torch.Tensor): Probability distribution over classes [B, n_classes]
                - Y_hat (torch.Tensor): Predicted class indices [B]
                - cls_attn (torch.Tensor): Class token's attention to each instance [B, n+1]
                - results_dict (dict[str, torch.Tensor]): Additional results/metrics
        """
        h = data.float()  # [B, n, D]

        h = self._fc1(h)  # [B, n, d]

        # ---->pad
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:, :add_length, :]], dim=1)  # [B, N, d]

        # ---->cls_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).cuda()
        h = torch.cat((cls_tokens, h), dim=1)

        # ---->Translayer x1
        h = self.layer1(h)  # [B, N, d]

        # ---->PPEG
        h = self.pos_layer(h, _H, _W)  # [B, N, d]

        # ---->Translayer x2
        h, attention = self.layer2(h, return_attention=True)  # [B, N, d]

        # Extract CLS token attention (first token's attention to all others)
        # Shape of attention is [B, h, N, N] where h is number of heads
        # We want the first row for each head (CLS token's attention to all tokens)
        cls_attention = attention[:, :, 0, :]  # [B, h, N]

        # Average across heads to get a single attention vector per batch item
        # Or keep all heads separate if you prefer
        cls_attention_avg = cls_attention.mean(dim=1)  # [B, N]

        # Remove padding if needed - get only attention to real tokens
        orig_seq_len = H + 1  # +1 for cls token
        cls_attn = cls_attention_avg[:, :orig_seq_len]

        # ---->cls_token
        h = self.norm(h)[:, 0]

        # ---->predict
        logits = self._fc2(h)  # [B, n_classes]
        Y_hat = torch.argmax(logits, dim=1)
        Y_prob = F.softmax(logits, dim=1)
        results_dict: dict[str, torch.Tensor] = {}

        return logits, Y_prob, Y_hat, cls_attn, results_dict

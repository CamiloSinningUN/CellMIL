"""
Core attention extraction components.

This module contains the core attention extraction logic for different
MIL model types.
"""

from .attention_extractor import (
    AttentionResult,
    BaseAttentionExtractor,
    CLAMAttentionExtractor,
    AttentionDeepMILExtractor,
    GraphMILAttentionExtractor,
    AttentionExtractorFactory,
)

__all__ = [
    "AttentionResult",
    "BaseAttentionExtractor",
    "CLAMAttentionExtractor",
    "AttentionDeepMILExtractor",
    "GraphMILAttentionExtractor",
    "AttentionExtractorFactory",
]

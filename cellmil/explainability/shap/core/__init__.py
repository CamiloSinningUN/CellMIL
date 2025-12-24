"""
Core SHAP computation module.

This module contains the logic for computing SHAP values
and performing stratified sampling based on attention scores.
"""

from .sampler import AttentionStratifiedSampler
from .computer import SHAPComputer

__all__ = ["AttentionStratifiedSampler", "SHAPComputer"]

"""
SHAP explainability module for MIL models.

This module provides SHAP-based feature importance analysis for attention-based
MIL models by creating cell-level datasets and computing SHAP values on stratified
samples based on attention scores.
"""

from .shap_explainer import SHAPExplainer

__all__ = ["SHAPExplainer"]

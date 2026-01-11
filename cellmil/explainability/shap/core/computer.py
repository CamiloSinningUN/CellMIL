"""
SHAP value computation module.

This module handles the computation of SHAP values using DeepExplainer or GradientExplainer
to explain the attention mechanism (MLP network).
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, cast
import shap  # type: ignore
from cellmil.interfaces.SHAPExplainerConfig import SHAPExplainerType
from cellmil.utils import logger
from cellmil.models.mil.attentiondeepmil import LitAttentionDeepMIL, AttentionDeepMIL
from cellmil.models.mil.head4type import LitHead4Type, Head4Type
from cellmil.models.mil.clam import LitCLAM, CLAM_SB, CLAM_MB
from .sampler import AttentionStratifiedSampler


class SHAPComputer:
    """
    Computes SHAP values for explaining attention mechanisms.

    Uses SHAP DeepExplainer or GradientExplainer to determine which features 
    contribute most to the attention weights assigned by the attention MLP.
    """

    def __init__(
        self,
        explainer_type: SHAPExplainerType = SHAPExplainerType.gradient,
        background_percentage: float = 0.2,
        nsamples: int = 100,
        explain_top_cells: Optional[int] = None,
        explain_per_head: bool = True,
        explain_mean_head: bool = True,
    ):
        """
        Initialize the SHAP computer.

        Args:
            explainer_type: Type of SHAP explainer (gradient, deep, or kernel)
            background_percentage: Percentage of sampled cells to use as background (0.0-1.0)
            nsamples: Number of coalitions for SHAP computation (only for kernel explainer)
            explain_top_cells: Number of top cells to explain (None = all)
            explain_per_head: Whether to compute SHAP for each attention head
            explain_mean_head: Whether to compute SHAP for mean attention across heads
        """
        self.explainer_type = explainer_type
        self.background_percentage = background_percentage
        self.nsamples = nsamples
        self.explain_top_cells = explain_top_cells
        self.explain_per_head = explain_per_head
        self.explain_mean_head = explain_mean_head

    def compute_shap_values(
        self,
        model: LitAttentionDeepMIL | LitHead4Type | LitCLAM,
        all_features: np.ndarray[Any, Any],
        device: torch.device,
        sampler: AttentionStratifiedSampler, 
        max_total_samples: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Compute SHAP values for the sampled cells.
        
        Steps:
        1. Compute attention scores for all cells
        2. Perform stratified sampling based on attention
        3. Compute SHAP values for sampled cells

        Args:
            model: The full MIL model (LitAttentionDeepMIL, LitHead4Type, or LitCLAM)
            all_features: RAW features of ALL cells [n_total_cells, n_features]
            device: Device to run computations on
            sampler: Stratified sampler for attention-based sampling
            max_total_samples: Maximum total samples to use

        Returns:
            Dictionary containing SHAP values, attention scores, sampling info, etc.
        """
        # Extract modules from model
        feature_extractor = self._get_feature_extractor(model)
        attention_module = self._get_attention_module(model)
        
        # Move to device and eval mode
        feature_extractor = feature_extractor.to(device).eval()
        attention_module = attention_module.to(device).eval()
        
        logger.info("Extracted feature_extractor and attention modules")
        
        # Step 1: Compute attention scores for ALL cells
        logger.info(f"Computing attention scores for {len(all_features):,} cells...")
        attention_scores = self._compute_attention_for_all_cells(
            feature_extractor=feature_extractor,
            attention_module=attention_module,
            all_features=all_features,
            device=device,
        )
        
        logger.info(
            f"Attention scores: min={attention_scores.min():.6f}, "
            f"max={attention_scores.max():.6f}, "
            f"mean={attention_scores.mean():.6f}"
        )
        
        # Step 2: Perform stratified sampling
        logger.info("Performing stratified sampling based on attention quantiles...")
        sampled_indices, sampling_info = sampler.sample(
            attention_scores,
            max_total_samples=max_total_samples,
        )
        print(sampled_indices)
        logger.info(f"Sampled {len(sampled_indices)} cells from {sampling_info['num_bins']} bins")
        
        # Step 3: Get sampled features and compute SHAP
        sampled_features = all_features[sampled_indices]
        logger.info(f"Computing SHAP on {len(sampled_features)} sampled cells")
        
        # Split into background and cells to explain (disjoint sets)
        n_samples = len(sampled_features)
        background_size = max(1, int(n_samples * self.background_percentage))
        
        # Use first portion for background, rest for explanation
        background_features = sampled_features[:background_size]
        explain_features = sampled_features[background_size:]
        
        logger.info(
            f"Split: {background_size} background cells, "
            f"{len(explain_features)} cells to explain"
        )
        
        # Compute attention for non-background cells (to select top cells if needed)
        logger.info("Computing attention for cells to be explained...")
        
        with torch.no_grad():
            x_tensor = torch.FloatTensor(explain_features).to(device)
            h = feature_extractor(x_tensor)  # Transform through feature extractor first
            attention_output = attention_module(h)  # [n_non_background, n_heads]
            
            # Handle tuple return (CLAM models return (attention_scores, h))
            if isinstance(attention_output, tuple):
                attention_logits = attention_output[0]
            else:
                attention_logits = attention_output
            
            # Get mean attention across heads
            if attention_logits.dim() > 1:
                attention_scores_explain = attention_logits.mean(dim=1).cpu().numpy()
            else:
                attention_scores_explain = attention_logits.cpu().numpy()

        # Determine which cells to explain
        if self.explain_top_cells is not None and self.explain_top_cells < len(explain_features):
            # Select top cells by attention score
            top_k_indices = np.argsort(attention_scores_explain)[-self.explain_top_cells:]
            cells_to_explain = explain_features[top_k_indices]
            # Compute actual indices in the original sampled_features
            actual_indices = np.arange(background_size, n_samples)[top_k_indices]
            logger.info(f"Explaining top {self.explain_top_cells} cells by attention")
        else:
            # Explain all non-background cells
            cells_to_explain = explain_features
            actual_indices = np.arange(background_size, n_samples)
            logger.info(f"Explaining all {len(explain_features)} non-background cells")

        # Convert to tensors
        background_tensor = torch.FloatTensor(background_features).to(device)
        explain_tensor = torch.FloatTensor(cells_to_explain).to(device)

        # Get number of attention heads
        with torch.no_grad():
            h_test = feature_extractor(explain_tensor[:1])
            test_output = attention_module(h_test)
            
            # Handle tuple return (CLAM models return (attention_scores, h))
            if isinstance(test_output, tuple):
                test_output = test_output[0]
            
            num_heads = test_output.shape[1] if test_output.dim() > 1 else 1

        logger.info(f"Attention module has {num_heads} head(s)")

        results: Dict[str, Any] = {
            "shap_values_per_head": {},
            "feature_importance_per_head": {},
            "top_features_per_head": {},
            "explained_cells_features": cells_to_explain,
            "explained_cells_indices": actual_indices,
            "background_data": background_features,
            "background_indices": np.arange(background_size),
            "num_heads": num_heads,
            "attention_scores": attention_scores,
            "sampled_indices": sampled_indices,  
            "sampling_info": sampling_info,  
        }

        # Compute SHAP for each head if requested
        if self.explain_per_head and num_heads > 1:
            for head_idx in range(num_heads):
                logger.info(f"\nComputing SHAP for head {head_idx+1}/{num_heads}...")
                head_results = self._compute_shap_for_head(
                    feature_extractor=feature_extractor,
                    attention_module=attention_module,
                    background_tensor=background_tensor,
                    explain_tensor=explain_tensor,
                    head_idx=head_idx,
                )
                results["shap_values_per_head"][f"head_{head_idx}"] = head_results["shap_values"]
                results["feature_importance_per_head"][f"head_{head_idx}"] = head_results["feature_importance"]
                results["top_features_per_head"][f"head_{head_idx}"] = head_results["top_features_idx"]

        # Compute SHAP for mean across heads if requested
        if self.explain_mean_head or num_heads == 1:
            logger.info("\nComputing SHAP for mean attention across heads...")
            mean_results = self._compute_shap_for_head(
                feature_extractor=feature_extractor,
                attention_module=attention_module,
                background_tensor=background_tensor,
                explain_tensor=explain_tensor,
                head_idx=None,  # None means average across all heads
            )
            results["shap_values_per_head"]["mean"] = mean_results["shap_values"]
            results["feature_importance_per_head"]["mean"] = mean_results["feature_importance"]
            results["top_features_per_head"]["mean"] = mean_results["top_features_idx"]

            # For single head, use mean as the main result
            if num_heads == 1:
                results["shap_values"] = mean_results["shap_values"]
                results["feature_importance"] = mean_results["feature_importance"]
                results["top_features_idx"] = mean_results["top_features_idx"]
                results["base_values"] = mean_results["base_value"]

        return results

    def _compute_shap_for_head(
        self,
        feature_extractor: torch.nn.Module,
        attention_module: torch.nn.Module,
        background_tensor: torch.Tensor,
        explain_tensor: torch.Tensor,
        head_idx: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Compute SHAP values for a specific attention head or mean across heads.

        Args:
            feature_extractor: The feature extraction module
            attention_module: The attention MLP module
            background_tensor: Background data tensor (RAW features)
            explain_tensor: Data to explain tensor (RAW features)
            head_idx: Which head to explain (None = mean across all heads)

        Returns:
            Dictionary with SHAP values and feature importance for this head
        """
        # Create wrapper module that applies feature_extractor → attention → head selection
        class AttentionHeadWrapper(torch.nn.Module):
            def __init__(
                self, 
                feature_extractor: nn.Module, 
                attention_module: nn.Module, 
                head_idx: int | None = None
            ):
                super().__init__() # type: ignore
                self.feature_extractor = feature_extractor
                self.attention_module = attention_module
                self.head_idx = head_idx

            def forward(
                self, 
                x: torch.Tensor
            ) -> torch.Tensor:
                # Transform raw features through feature extractor
                h = self.feature_extractor(x)
                # Get attention scores
                attention_output = self.attention_module(h)
                
                # Handle tuple return (CLAM models return (attention_scores, h))
                if isinstance(attention_output, tuple):
                    out = attention_output[0]
                else:
                    out = attention_output
                
                if self.head_idx is None:
                    # Return mean across heads
                    if out.dim() > 1:
                        return out.mean(dim=1, keepdim=True)
                    return out
                else:
                    # Return specific head
                    if out.dim() > 1:
                        return out[:, self.head_idx:self.head_idx+1]
                    return out

        wrapper = AttentionHeadWrapper(feature_extractor, attention_module, head_idx).eval()

        # Create SHAP explainer based on type
        logger.info(f"Initializing SHAP {self.explainer_type.value.capitalize()}Explainer...")

        if self.explainer_type == SHAPExplainerType.gradient:
            explainer = shap.GradientExplainer(wrapper, background_tensor)
        elif self.explainer_type == SHAPExplainerType.deep:
            explainer = shap.DeepExplainer(wrapper, background_tensor)
        elif self.explainer_type == SHAPExplainerType.kernel:
            # Kernel explainer needs a predict function
            def predict_fn(x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
                with torch.no_grad():
                    return wrapper(torch.FloatTensor(x).to(background_tensor.device)).cpu().numpy()
            explainer = shap.KernelExplainer(predict_fn, background_tensor.cpu().numpy()) # type: ignore
            explain_tensor = cast(np.ndarray[Any, Any], explain_tensor.cpu().numpy()) # type: ignore
        else:
            raise ValueError(f"Unknown explainer type: {self.explainer_type}")

        # Compute SHAP values
        head_name = f"head {head_idx}" if head_idx is not None else "mean"
        logger.info(f"Computing SHAP values for {head_name} ({len(explain_tensor)} cells)...")

        if self.explainer_type == SHAPExplainerType.kernel:
            shap_values = cast(np.ndarray[Any, Any] | torch.Tensor, explainer.shap_values(explain_tensor, nsamples=self.nsamples)) # type: ignore
        else:
            shap_values = cast(np.ndarray[Any, Any] | torch.Tensor, explainer.shap_values(explain_tensor)) # type: ignore
            # Convert to numpy if tensor
            if isinstance(shap_values, torch.Tensor):
                shap_values = cast(np.ndarray[Any, Any], shap_values.cpu().numpy()) # type: ignore

        # Handle output shape
        if isinstance(shap_values, np.ndarray):
            # If output has extra dimension, squeeze it
            if shap_values.ndim == 3 and shap_values.shape[1] == 1:
                shap_values = shap_values[:, 0, :]
            # Ensure we have 2D array [n_samples, n_features]
            shap_values = np.squeeze(shap_values)
            if shap_values.ndim == 1:
                shap_values = shap_values.reshape(-1, 1)

        logger.info(f"SHAP values shape: {shap_values.shape}")

        # Compute feature importance
        feature_importance = np.abs(shap_values).mean(axis=0)
        # Ensure feature_importance is 1D
        feature_importance = np.squeeze(feature_importance)
        
        logger.info(f"Feature importance shape: {feature_importance.shape}")
        logger.info(f"Feature importance stats: min={feature_importance.min():.6f}, max={feature_importance.max():.6f}, mean={feature_importance.mean():.6f}")
        
        top_features_idx = np.argsort(feature_importance)[::-1]

        logger.info(f"Top 10 features for {head_name}: {top_features_idx[:10]}")
        logger.info(f"Top 10 feature importances: {feature_importance[top_features_idx[:10]]}")

        return {
            "shap_values": shap_values,
            "feature_importance": feature_importance,
            "top_features_idx": top_features_idx,
            "base_value": explainer.expected_value if hasattr(explainer, "expected_value") else 0.0, # type: ignore
        }

    def _compute_attention_for_all_cells(
        self,
        feature_extractor: nn.Module,
        attention_module: nn.Module,
        all_features: np.ndarray[Any, Any],
        device: torch.device,
        batch_size: int = 1024,
    ) -> np.ndarray[Any, Any]:
        """
        Compute attention scores for all cells using the same pipeline as SHAP.
        
        Pipeline: raw_features → feature_extractor → attention_module → scores
        
        This ensures consistency between sampling (which uses attention) and 
        explanation (which explains the same attention pipeline).

        Args:
            feature_extractor: Module to transform raw features
            attention_module: Module to compute attention
            all_features: Raw features for all cells [n_cells, n_features]
            device: Device to run on
            batch_size: Batch size for processing

        Returns:
            Attention scores [n_cells] (mean across heads if multi-head)
        """
        n_cells = len(all_features)
        all_attention: list[np.ndarray[Any, Any]] = []
        
        with torch.no_grad():
            for start_idx in range(0, n_cells, batch_size):
                end_idx = min(start_idx + batch_size, n_cells)
                batch_features = torch.FloatTensor(all_features[start_idx:end_idx]).to(device)
                
                # Apply same pipeline as SHAP wrapper
                h = feature_extractor(batch_features)  # Transform features
                attention_output = attention_module(h)  # Compute attention
                
                # Handle tuple return (CLAM models return (attention_scores, h))
                if isinstance(attention_output, tuple):
                    attention_logits = attention_output[0]
                else:
                    attention_logits = attention_output
                
                # Get mean across heads if multi-head
                if attention_logits.dim() > 1:
                    attention_scores = attention_logits.mean(dim=1).cpu().numpy()
                else:
                    attention_scores = attention_logits.cpu().numpy()
                
                all_attention.append(attention_scores)
        
        # Concatenate all batches
        return np.concatenate(all_attention, axis=0)

    def _get_attention_module(self, lit_model: LitAttentionDeepMIL | LitHead4Type | LitCLAM) -> torch.nn.Module:
        """
        Extract the attention module from a MIL model.

        Args:
            model: The full MIL model (LitAttentionDeepMIL, LitHead4Type, or LitCLAM)

        Returns:
            The attention MLP module
        """
        # For Lightning models, get the underlying model
        model = cast(AttentionDeepMIL | Head4Type | CLAM_SB | CLAM_MB, lit_model.model)
        # Get the attention module
        if isinstance(model, (CLAM_SB, CLAM_MB)):
            # For CLAM, attention_net is Sequential: [Linear, ReLU, Dropout, Attn_Net/Attn_Net_Gated]
            # We need to extract the last module which is the actual attention network
            return model.attention_net[-1]
        elif hasattr(model, "attention"):
            return model.attention 
        else:
            raise ValueError(f"Model {type(model)} does not have an 'attention' module")

    def _get_feature_extractor(self, lit_model: LitAttentionDeepMIL | LitHead4Type | LitCLAM) -> torch.nn.Module:
        """
        Extract the feature extractor from a MIL model.

        This is needed to transform raw features before feeding to attention module.

        Args:
            model: The full MIL model

        Returns:
            The feature extractor module
        """
        # For Lightning models, get the underlying model
        model = cast(AttentionDeepMIL | Head4Type | CLAM_SB | CLAM_MB, lit_model.model)

        # Get the feature extractor
        if isinstance(model, (CLAM_SB, CLAM_MB)):
            # For CLAM, we need the first 3 layers of attention_net: [Linear, ReLU, Dropout]
            # These transform the raw features before the attention layer
            return nn.Sequential(*list(model.attention_net.children())[:-1])
        elif hasattr(model, "feature_extractor_part2"):
            return model.feature_extractor_part2
        else:
            raise ValueError(f"Model {type(model)} does not have a feature extractor module")

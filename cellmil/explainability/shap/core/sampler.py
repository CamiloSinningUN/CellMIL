"""
Attention-based stratified sampler for SHAP analysis.

This module implements stratified sampling based on attention score quantiles
to ensure representation across the full range of attention values.
"""

import numpy as np
from typing import Tuple, Any
from cellmil.utils import logger


class AttentionStratifiedSampler:
    """
    Performs stratified sampling based on attention score quantiles.

    This ensures we have representation from low, medium, and high attention cells,
    which is important for understanding the full attention mechanism behavior.
    """

    def __init__(self, num_bins: int, samples_per_bin: int, random_seed: int = 42):
        """
        Initialize the sampler.

        Args:
            num_bins: Number of quantile bins for stratification
            samples_per_bin: Number of samples to draw from each bin
            random_seed: Random seed for reproducibility
        """
        self.num_bins = num_bins
        self.samples_per_bin = samples_per_bin
        self.random_seed = random_seed
        np.random.seed(random_seed)

    def sample(
        self,
        attention_scores: np.ndarray[Any, Any],
        max_total_samples: int | None = None,
    ) -> Tuple[np.ndarray[Any, Any], dict[str, Any]]:
        """
        Perform stratified sampling based on attention quantiles.

        Args:
            attention_scores: Array of attention scores [total_cells]
            max_total_samples: Optional maximum total samples to return

        Returns:
            Tuple of (sampled_indices, sampling_info) where:
                - sampled_indices: Array of selected cell indices
                - sampling_info: Dictionary with bin statistics
        """
        # Calculate quantile bins
        quantiles = np.linspace(0, 1, self.num_bins + 1)
        bin_edges = np.quantile(attention_scores, quantiles)

        logger.info(f"Quantile bin edges: {bin_edges}")

        sampled_indices: list[int] = []
        bin_info: dict[str, Any] = {}

        # Sample from each bin
        for i in range(self.num_bins):
            lower = bin_edges[i]
            upper = bin_edges[i + 1]

            # Find cells in this bin
            if i == self.num_bins - 1:
                # Last bin: include upper boundary
                bin_mask = (attention_scores >= lower) & (attention_scores <= upper)
            else:
                bin_mask = (attention_scores >= lower) & (attention_scores < upper)

            bin_indices = np.where(bin_mask)[0]
            bin_size = len(bin_indices)

            logger.info(
                f"Bin {i + 1}/{self.num_bins}: [{lower:.6f}, {upper:.6f}] - {bin_size:,} cells"
            )

            if bin_size == 0:
                logger.warning(f"Bin {i + 1} is empty, skipping")
                bin_info[f"bin_{i + 1}"] = {
                    "range": (float(lower), float(upper)),
                    "total_cells": 0,
                    "sampled_cells": 0,
                }
                continue

            # Sample from this bin
            n_samples = min(self.samples_per_bin, bin_size)
            sampled = np.random.choice(bin_indices, size=n_samples, replace=False)
            sampled_indices.extend(sampled.tolist())

            bin_info[f"bin_{i + 1}"] = {
                "range": (float(lower), float(upper)),
                "total_cells": int(bin_size),
                "sampled_cells": int(n_samples),
                "sampling_rate": float(n_samples / bin_size),
            }

        sampled_indices_array = np.array(sampled_indices)

        # Apply max_total_samples if specified
        if (
            max_total_samples is not None
            and len(sampled_indices_array) > max_total_samples
        ):
            logger.info(
                f"Reducing samples from {len(sampled_indices_array)} to {max_total_samples}"
            )
            sampled_indices_array = np.random.choice(
                sampled_indices_array,
                size=max_total_samples,
                replace=False,
            )

        sampling_info: dict[str, Any] = {
            "num_bins": self.num_bins,
            "samples_per_bin": self.samples_per_bin,
            "total_sampled": len(sampled_indices_array),
            "bin_info": bin_info,
        }

        logger.info(
            f"Stratified sampling completed: {len(sampled_indices_array):,} cells sampled"
        )

        return sampled_indices_array, sampling_info

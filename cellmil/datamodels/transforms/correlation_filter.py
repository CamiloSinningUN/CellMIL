"""
Correlation filter transform for removing highly correlated features.
"""

from typing import Any, Dict, Optional, List, cast
import torch
import matplotlib.pyplot as plt

from cellmil.utils import logger
from .base_transform import FittableTransform


class CorrelationFilterTransform(FittableTransform):
    """
    Transform that removes highly correlated features based on a correlation threshold.

    Features with correlation above the threshold will have one feature removed.
    Also removes constant features (features with very low standard deviation).
    """

    def __init__(
        self,
        correlation_threshold: float = 0.9,
        plot_correlation_matrix: bool = False,
        constant_threshold: float = 1e-8,
    ):
        """
        Initialize the correlation filter transform.

        Args:
            correlation_threshold: Correlation threshold above which features will be removed
            plot_correlation_matrix: Whether to plot the correlation matrix during fitting
            constant_threshold: Threshold below which features are considered constant
        """
        super().__init__("correlation_filter")
        self.correlation_threshold = correlation_threshold
        self.plot_correlation_matrix = plot_correlation_matrix
        self.constant_threshold = constant_threshold

        # Fitted parameters
        self.keep_mask_: Optional[torch.Tensor] = None
        self.non_constant_mask_: Optional[torch.Tensor] = None
        self.num_original_features_: Optional[int] = None
        self.removed_feature_indices_: Optional[List[int]] = None

    def fit(
        self, features: torch.Tensor, feature_names: Optional[List[str]] = None
    ) -> "CorrelationFilterTransform":
        """
        Fit the correlation filter on training data.

        Args:
            features: Training features tensor of shape (n_instances, n_features)
            feature_names: Optional list of feature names for logging

        Returns:
            Self for method chaining
        """
        # Store shape info
        total_features = features.shape[1]
        total_instances = features.shape[0]
        self.num_original_features_ = total_features

        logger.info(
            f"Computing correlation matrix for {total_features} features using {total_instances} instances..."
        )

        # Find non-constant features
        feature_std = features.std(dim=0)
        self.non_constant_mask_ = feature_std > self.constant_threshold
        del feature_std

        if self.non_constant_mask_.sum() == 0:
            raise ValueError(
                "All features are constant. Cannot apply correlation filter."
            )

        logger.info(
            f"Found {self.non_constant_mask_.sum()} non-constant features out of {total_features} total features"
        )

        # Only compute correlation for non-constant features
        valid_features = features[:, self.non_constant_mask_]

        # Compute correlation matrix
        corr_matrix = self._compute_correlation_matrix(valid_features)

        # Plot correlation matrix if requested
        if self.plot_correlation_matrix:
            self._plot_correlation_matrix(corr_matrix)

        # Find highly correlated pairs and determine which features to remove
        features_to_remove = self._find_features_to_remove(corr_matrix)

        # Create final mask mapping back to original feature space
        self.keep_mask_ = torch.ones(total_features, dtype=torch.bool)

        # Map back to original indices
        valid_indices = torch.where(self.non_constant_mask_)[0]
        for idx_to_remove in features_to_remove:
            original_idx = valid_indices[idx_to_remove]
            self.keep_mask_[original_idx] = False

        # Also remove constant features
        self.keep_mask_ = self.keep_mask_ & self.non_constant_mask_

        # Store removed feature indices for logging
        self.removed_feature_indices_ = cast(
            list[int],
            torch.where(~self.keep_mask_)[0].tolist(),  # type: ignore
        )

        features_removed = (~self.keep_mask_).sum().item()
        features_kept = self.keep_mask_.sum().item()

        logger.info(
            f"Correlation filter: removed {features_removed} features, kept {features_kept} features"
        )

        # Log removed feature names if available
        if feature_names is not None and self.removed_feature_indices_:
            removed_names = [
                feature_names[i]
                for i in self.removed_feature_indices_
                if i < len(feature_names)
            ]
            if removed_names:
                logger.info(f"Removed feature names: {removed_names}")

        self.is_fitted = True
        return self

    def _transform_impl(self, features: torch.Tensor) -> torch.Tensor:
        """
        Apply the correlation filter to features.

        Args:
            features: Input features tensor

        Returns:
            Filtered features tensor
        """
        if features.size(1) != self.num_original_features_:
            raise ValueError(
                f"Feature dimension mismatch: expected {self.num_original_features_} features, "
                f"got {features.size(1)}"
            )

        return features[:, self.keep_mask_]

    def _compute_correlation_matrix(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute correlation matrix efficiently.

        Args:
            features: Feature tensor

        Returns:
            Correlation matrix
        """
        # Center the features
        feature_means = features.mean(dim=0)
        centered_features = features - feature_means

        # Compute covariance matrix
        n_samples = centered_features.shape[0]
        cov_matrix = torch.mm(centered_features.T, centered_features) / (n_samples - 1)

        # Compute correlation matrix
        std_devs = torch.sqrt(torch.diag(cov_matrix))
        corr_matrix = cov_matrix / torch.outer(std_devs, std_devs)

        return corr_matrix

    def _find_features_to_remove(self, corr_matrix: torch.Tensor) -> set[int]:
        """
        Find features to remove based on correlation threshold using iterative approach.

        Args:
            corr_matrix: Correlation matrix

        Returns:
            Set of feature indices to remove
        """
        features_to_remove: set[int] = set()
        working_corr_matrix = corr_matrix.clone()
        n_features = working_corr_matrix.shape[0]
        remaining_features = set(range(n_features))

        iteration = 0

        while True:
            iteration += 1

            # Create mask for remaining features
            remaining_indices = list(remaining_features)
            if len(remaining_indices) < 2:
                break

            # Extract submatrix for remaining features
            sub_corr_matrix = working_corr_matrix[remaining_indices][
                :, remaining_indices
            ]

            # Find highly correlated pairs in upper triangle
            upper_triangle = torch.triu(torch.abs(sub_corr_matrix), diagonal=1)
            high_corr_pairs = torch.where(upper_triangle > self.correlation_threshold)

            if len(high_corr_pairs[0]) == 0:
                # No more highly correlated pairs
                break

            # Find the most correlated pair
            max_corr_idx = torch.argmax(upper_triangle[high_corr_pairs])
            most_corr_i = int(high_corr_pairs[0][max_corr_idx])
            most_corr_j = int(high_corr_pairs[1][max_corr_idx])
            max_corr_value = float(upper_triangle[most_corr_i, most_corr_j])

            # Map back to original indices
            original_i = remaining_indices[most_corr_i]
            original_j = remaining_indices[most_corr_j]

            # Decide which feature to remove based on their mean correlation with other features
            # Remove the feature that has higher average correlation with remaining features
            remaining_for_avg = [
                idx for idx in remaining_indices if idx not in {original_i, original_j}
            ]

            if remaining_for_avg:
                avg_corr_i = torch.mean(
                    torch.abs(working_corr_matrix[original_i, remaining_for_avg])
                )
                avg_corr_j = torch.mean(
                    torch.abs(working_corr_matrix[original_j, remaining_for_avg])
                )
                feature_to_remove = (
                    original_i if avg_corr_i > avg_corr_j else original_j
                )
            else:
                # If only two features left, remove the second one (arbitrary choice)
                feature_to_remove = original_j

            # Remove the selected feature
            features_to_remove.add(feature_to_remove)
            remaining_features.remove(feature_to_remove)

            logger.debug(
                f"Iteration {iteration}: Removed feature {feature_to_remove} "
                f"(corr={max_corr_value:.4f} with feature {original_i if feature_to_remove == original_j else original_j})"
            )

        total_removed = len(features_to_remove)
        if total_removed > 0:
            logger.info(
                f"Iterative correlation filter completed in {iteration} iterations: "
                f"removed {total_removed} features (threshold: {self.correlation_threshold})"
            )
        else:
            logger.info(
                f"No features removed - no correlations exceeded threshold {self.correlation_threshold}"
            )

        return features_to_remove

    def _plot_correlation_matrix(self, corr_matrix: torch.Tensor) -> None:
        """
        Plot the correlation matrix.

        Args:
            corr_matrix: Correlation matrix to plot
        """
        try:
            # Convert to numpy for plotting
            corr_np = corr_matrix.detach().cpu().numpy()  # type: ignore

            # Create the plot
            _, ax = plt.subplots(figsize=(12, 10))  # type: ignore

            # Create heatmap
            im = ax.imshow(corr_np, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")  # type: ignore

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)  # type: ignore
            cbar.set_label("Correlation Coefficient", rotation=270, labelpad=20)  # type: ignore

            # Set title and labels
            ax.set_title(  # type: ignore
                f"Feature Correlation Matrix\n({corr_np.shape[0]} non-constant features)",
                fontsize=14,
                pad=20,
            )
            ax.set_xlabel("Feature Index", fontsize=12)  # type: ignore
            ax.set_ylabel("Feature Index", fontsize=12)  # type: ignore

            # Add grid for better readability
            ax.grid(True, alpha=0.3)  # type: ignore

            # Adjust layout and show
            plt.tight_layout()
            plt.show()  # type: ignore

            logger.info("Correlation matrix plot displayed")

        except Exception as e:
            logger.warning(f"Failed to create correlation matrix plot: {e}")

    def get_config(self) -> Dict[str, Any]:
        """Get the configuration dictionary for this transform."""
        config: dict[str, Any] = {
            "name": self.name,
            "correlation_threshold": self.correlation_threshold,
            "plot_correlation_matrix": self.plot_correlation_matrix,
            "constant_threshold": self.constant_threshold,
        }

        if self.is_fitted:
            config.update(
                {
                    "keep_mask": self.keep_mask_.tolist()  # type: ignore
                    if self.keep_mask_ is not None
                    else None,
                    "non_constant_mask": self.non_constant_mask_.tolist()  # type: ignore
                    if self.non_constant_mask_ is not None
                    else None,
                    "num_original_features": self.num_original_features_,
                    "removed_feature_indices": self.removed_feature_indices_,
                    "is_fitted": True,
                }
            )
        else:
            config["is_fitted"] = False

        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "CorrelationFilterTransform":
        """Create transform instance from configuration dictionary."""
        # Create instance with hyperparameters
        transform = cls(
            correlation_threshold=config["correlation_threshold"],
            plot_correlation_matrix=config.get("plot_correlation_matrix", False),
            constant_threshold=config.get("constant_threshold", 1e-8),
        )

        # Restore fitted state if available
        if config.get("is_fitted", False):
            transform.keep_mask_ = (
                torch.tensor(config["keep_mask"], dtype=torch.bool)
                if config["keep_mask"] is not None
                else None
            )
            transform.non_constant_mask_ = (
                torch.tensor(config["non_constant_mask"], dtype=torch.bool)
                if config["non_constant_mask"] is not None
                else None
            )
            transform.num_original_features_ = config["num_original_features"]
            transform.removed_feature_indices_ = config["removed_feature_indices"]
            transform.is_fitted = True

        return transform

    def get_feature_importance_mask(self) -> Optional[torch.Tensor]:
        """
        Get a boolean mask indicating which original features are kept.

        Returns:
            Boolean tensor of shape (n_original_features,) where True indicates
            the feature is kept after correlation filtering.
        """
        if not self.is_fitted:
            return None
        return self.keep_mask_

    def get_removed_feature_indices(self) -> Optional[List[int]]:
        """
        Get the indices of features that were removed.

        Returns:
            List of feature indices that were removed, or None if not fitted.
        """
        if not self.is_fitted:
            return None
        return self.removed_feature_indices_

"""
Robust scaler transform for feature normalization.
"""

from typing import Any, Dict, Optional, List, Tuple, cast
import torch

from cellmil.utils import logger
from .base_transform import FittableTransform


class RobustScalerTransform(FittableTransform):
    """
    Transform that applies robust scaling to features using median and IQR.

    Robust scaling is less sensitive to outliers than standard scaling.
    Formula: (x - median) / IQR, where IQR = Q3 - Q1

    Features are first log-transformed to handle skewed distributions and outliers,
    then robust scaling is applied.
    """

    def __init__(
        self,
        apply_log_transform: bool = True,
        quantile_range: Tuple[float, float] = (0.25, 0.75),
        clip_quantiles: Tuple[float, float] = (0.005, 0.995),
        constant_threshold: float = 1e-8,
    ):
        """
        Initialize the robust scaler transform.

        Args:
            apply_log_transform: Whether to apply log transformation before scaling
            quantile_range: Tuple of (lower_quantile, upper_quantile) for IQR computation
            clip_quantiles: Tuple of (lower_clip, upper_clip) for outlier clipping
            constant_threshold: Threshold below which IQR is considered too small
        """
        super().__init__("robust_scaler")
        self.apply_log_transform = apply_log_transform
        self.quantile_range = quantile_range
        self.clip_quantiles = clip_quantiles
        self.constant_threshold = constant_threshold

        # Fitted parameters
        self.median_values_: Optional[torch.Tensor] = None
        self.iqr_values_: Optional[torch.Tensor] = None
        self.clip_min_values_: Optional[torch.Tensor] = None
        self.clip_max_values_: Optional[torch.Tensor] = None
        self.num_features_: Optional[int] = None
        self.constant_features_mask_: Optional[torch.Tensor] = None

    def fit(
        self, features: torch.Tensor, feature_names: Optional[List[str]] = None
    ) -> "RobustScalerTransform":
        """
        Fit the robust scaler on training data.

        Args:
            features: Training features tensor of shape (n_instances, n_features)
            feature_names: Optional list of feature names for logging

        Returns:
            Self for method chaining
        """
        self.num_features_ = features.shape[1]

        logger.info(
            f"Computing robust scaling parameters for {features.shape[1]} features "
            f"using {features.shape[0]} instances..."
        )

        # Apply log transformation if requested
        if self.apply_log_transform:
            # First clip extreme outliers
            self.clip_min_values_ = torch.quantile(
                features, self.clip_quantiles[0], dim=0
            )
            self.clip_max_values_ = torch.quantile(
                features, self.clip_quantiles[1], dim=0
            )

            features_processed = torch.clamp(
                features, min=self.clip_min_values_, max=self.clip_max_values_
            )

            # Apply log transformation: sign(x) * log(1 + |x| + epsilon)
            epsilon = 1e-8
            features_processed = torch.sign(features_processed) * torch.log1p(
                torch.abs(features_processed) + epsilon
            )
        else:
            features_processed = features.clone()
            self.clip_min_values_ = None
            self.clip_max_values_ = None

        # Compute median and IQR
        self.median_values_ = torch.median(features_processed, dim=0)[
            0
        ]  # Shape: (n_features,)

        q1 = torch.quantile(features_processed, self.quantile_range[0], dim=0)
        q3 = torch.quantile(features_processed, self.quantile_range[1], dim=0)
        self.iqr_values_ = q3 - q1

        # Handle features with zero/small IQR (near-constant features)
        self.constant_features_mask_ = self.iqr_values_ <= self.constant_threshold

        if self.constant_features_mask_.sum() > 0:
            num_constant = self.constant_features_mask_.sum().item()
            logger.info(
                f"Found {num_constant} near-constant features with very small IQR"
            )

            # For constant features, set IQR to 1 to avoid division by zero
            self.iqr_values_[self.constant_features_mask_] = 1.0

            # Log constant feature names if available
            if feature_names is not None:
                constant_indices = cast(
                    list[int], torch.where(self.constant_features_mask_)[0].tolist() # type: ignore
                )  
                constant_names = [
                    feature_names[i] for i in constant_indices if i < len(feature_names)
                ]
                if constant_names:
                    logger.info(f"Near-constant features: {constant_names}")

        logger.info(
            f"Computed robust scaling parameters: "
            f"median range = {self.median_values_.min():.6f} to {self.median_values_.max():.6f}, "
            f"IQR range = {self.iqr_values_.min():.6f} to {self.iqr_values_.max():.6f}"
        )

        self.is_fitted = True
        return self

    def _transform_impl(self, features: torch.Tensor) -> torch.Tensor:
        """
        Apply the robust scaling to features.

        Args:
            features: Input features tensor

        Returns:
            Normalized features tensor
        """
        if features.size(1) != self.num_features_:
            raise ValueError(
                f"Feature dimension mismatch: expected {self.num_features_} features, "
                f"got {features.size(1)}"
            )

        # Apply the same preprocessing as during fitting
        if self.apply_log_transform:
            # Clip outliers using fitted values
            features_processed = torch.clamp(
                features, min=self.clip_min_values_, max=self.clip_max_values_
            )

            # Apply log transformation
            epsilon = 1e-8
            features_processed = torch.sign(features_processed) * torch.log1p(
                torch.abs(features_processed) + epsilon
            )
        else:
            features_processed = features.clone()

        if self.median_values_ is None or self.iqr_values_ is None:
            raise RuntimeError("Transform has not been fitted yet")

        # Apply robust scaling: (x - median) / IQR
        scaled_features = (features_processed - self.median_values_) / self.iqr_values_

        return scaled_features

    def get_config(self) -> Dict[str, Any]:
        """Get the configuration dictionary for this transform."""
        config: dict[str, Any] = {
            "name": self.name,
            "apply_log_transform": self.apply_log_transform,
            "quantile_range": self.quantile_range,
            "clip_quantiles": self.clip_quantiles,
            "constant_threshold": self.constant_threshold,
        }

        if self.is_fitted:
            config.update(
                { 
                    "median_values": self.median_values_.tolist() # type: ignore
                    if self.median_values_ is not None
                    else None,  
                    "iqr_values": self.iqr_values_.tolist() # type: ignore
                    if self.iqr_values_ is not None
                    else None,  
                    "clip_min_values": self.clip_min_values_.tolist() # type: ignore
                    if self.clip_min_values_ is not None
                    else None,  
                    "clip_max_values": self.clip_max_values_.tolist() # type: ignore
                    if self.clip_max_values_ is not None
                    else None,  
                    "num_features": self.num_features_,
                    "constant_features_mask": self.constant_features_mask_.tolist() # type: ignore
                    if self.constant_features_mask_ is not None
                    else None,  
                    "is_fitted": True,
                }
            )
        else:
            config["is_fitted"] = False

        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "RobustScalerTransform":
        """Create transform instance from configuration dictionary."""
        # Create instance with hyperparameters
        transform = cls(
            apply_log_transform=config["apply_log_transform"],
            quantile_range=tuple(config["quantile_range"]),
            clip_quantiles=tuple(config["clip_quantiles"]),
            constant_threshold=config.get("constant_threshold", 1e-8),
        )

        # Restore fitted state if available
        if config.get("is_fitted", False):
            transform.median_values_ = (
                torch.tensor(config["median_values"], dtype=torch.float32)
                if config["median_values"] is not None
                else None
            )
            transform.iqr_values_ = (
                torch.tensor(config["iqr_values"], dtype=torch.float32)
                if config["iqr_values"] is not None
                else None
            )
            transform.clip_min_values_ = (
                torch.tensor(config["clip_min_values"], dtype=torch.float32)
                if config["clip_min_values"] is not None
                else None
            )
            transform.clip_max_values_ = (
                torch.tensor(config["clip_max_values"], dtype=torch.float32)
                if config["clip_max_values"] is not None
                else None
            )
            transform.num_features_ = config["num_features"]
            transform.constant_features_mask_ = (
                torch.tensor(config["constant_features_mask"], dtype=torch.bool)
                if config["constant_features_mask"] is not None
                else None
            )
            transform.is_fitted = True

        return transform

    def get_scaling_parameters(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get the scaling parameters (median, IQR) computed from training data.

        Returns:
            Tuple of (median_values, iqr_values) if fitted, None otherwise
        """

        if not self.is_fitted:
            return None

        if self.median_values_ is None or self.iqr_values_ is None:
            return None

        return self.median_values_, self.iqr_values_

    def get_constant_features_mask(self) -> Optional[torch.Tensor]:
        """
        Get a boolean mask indicating which features were considered constant.

        Returns:
            Boolean tensor of shape (n_features,) where True indicates
            the feature was considered constant during fitting.
        """
        if not self.is_fitted:
            return None
        return self.constant_features_mask_
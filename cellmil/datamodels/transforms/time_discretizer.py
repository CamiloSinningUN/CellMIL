"""
Survival discretization transform for converting continuous survival times to discrete bins.
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, cast, Union, Optional
from .base_label_transform import FittableLabelTransform


class TimeDiscretizerTransform(FittableLabelTransform):
    """
    Transform for discretizing continuous survival times into bins.
    
    This transform bins survival times using quantiles computed on uncensored patients only.
    It's designed for discrete-time survival analysis where we convert the regression 
    problem into a classification problem.
    
    Args:
        n_bins (int): Number of bins to create (e.g., 4 for quartiles)
        eps (float): Small epsilon value to adjust bin boundaries
    """
    
    def __init__(self, n_bins: int = 4, eps: float = 1e-8):
        super().__init__(name="survival_discretizer")
        self.n_bins = n_bins
        self.eps = eps
        self.bins: Optional[np.ndarray[Any, Any]] = None
        
    def fit(
        self, 
        labels: Dict[str, Union[int, Tuple[float, int]]],
        **kwargs: Any
    ) -> "TimeDiscretizerTransform":
        """
        Fit the discretizer on survival data.
        
        Args:
            labels: Dictionary mapping slide IDs to (duration, event) tuples
            **kwargs: Additional keyword arguments (not used, for API compatibility)
            
        Returns:
            Self for method chaining
        """
        # Extract uncensored patients for computing quantiles
        
        uncensored_durations: List[float] = []
        all_durations: List[float] = []
        
        for duration, event in labels.values():  # type: ignore
            all_durations.append(duration)  # type: ignore
            if event == 1:  # Uncensored (event occurred)
                uncensored_durations.append(duration) # type: ignore
        
        if len(uncensored_durations) < self.n_bins:
            raise ValueError(
                f"Not enough uncensored samples ({len(uncensored_durations)}) "
                f"to create {self.n_bins} bins"
            )
        
        # Compute quantile bins on uncensored data
        _, bins = pd.qcut(uncensored_durations, q=self.n_bins, retbins=True, labels=False) # type: ignore
        
        # Adjust bin boundaries to ensure all data is included
        bins = np.array(bins)
        bins[-1] = max(all_durations) + self.eps
        bins[0] = min(all_durations) - self.eps
        
        self.bins = bins
        self.is_fitted = True
        
        return self
    
    def _transform_labels_impl(
        self, labels: Dict[str, Union[int, Tuple[float, int]]]
    ) -> Dict[str, Union[int, Tuple[float, int]]]:
        """
        Implementation of the label transform operation.
        
        Args:
            labels: Dictionary mapping slide IDs to (duration, event) tuples
            
        Returns:
            Dictionary mapping slide IDs to (bin_index, event) tuples
        """
        if self.bins is None:
            raise RuntimeError("Transform must be fitted before use")
        
        discretized_labels: Dict[str, Tuple[int, int]] = {}
        
        for slide_id, label in labels.items():
            # Extract duration and event from label
            if isinstance(label, tuple):
                duration, event = label
            else:
                raise ValueError(f"Expected tuple label for survival analysis, got {type(label)}")
            
            # Use pd.cut to assign to bins (right=False means left-inclusive)
            bin_index = cast(int, pd.cut( # type: ignore
                [duration], 
                bins=self.bins, # type: ignore
                labels=False, 
                right=False, 
                include_lowest=True
            )[0])
            
            # Convert to int (pd.cut may return float with NaN)
            if pd.isna(bin_index): # type: ignore
                # Handle edge case: assign to closest bin
                if duration <= self.bins[0]:
                    bin_index = 0
                elif duration >= self.bins[-1]:
                    bin_index = self.n_bins - 1
                else:
                    # Find the appropriate bin
                    bin_index = np.searchsorted(self.bins, duration, side='right') - 1
                    bin_index = max(0, min(bin_index, self.n_bins - 1))
            
            discretized_labels[slide_id] = (int(bin_index), int(event))
        
        return discretized_labels  # type: ignore
    
    def get_config(self) -> Dict[str, Any]:
        """Get configuration for saving."""
        config: dict[str, Any] = {
            "name": self.name,
            "n_bins": self.n_bins,
            "eps": self.eps,
        }
        
        if self.bins is not None:
            config["bins"] = self.bins.tolist()
            
        return config
    
    @classmethod
    def from_config(cls, config: dict[str, Any]):
        """Create instance from configuration."""
        transform = cls(
            n_bins=config["n_bins"],
            eps=config["eps"]
        )
        
        if "bins" in config:
            transform.bins = np.array(config["bins"])
            transform.is_fitted = True
            
        return transform
    
    def save(self, path: Path) -> None:
        """Save transform to disk."""
        
        config = self.get_config()
        config["transform_class"] = self.__class__.__name__
        
        with open(path, "w") as f:
            json.dump(config, f, indent=2)
    
    @classmethod
    def load(cls, path: Path):
        """Load transform from disk."""
        
        with open(path, "r") as f:
            config = json.load(f)
        
        config.pop("transform_class", None)
        return cls.from_config(config)

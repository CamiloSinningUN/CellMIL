import os
from .k_fold_cross_validation import KFoldCrossValidation

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

__all__ = ["KFoldCrossValidation"]
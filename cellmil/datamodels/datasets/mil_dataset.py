import pandas as pd
from typing import Any, Literal
from pathlib import Path
from .cell_mil_dataset import CellMILDataset
from .patch_mil_dataset import PatchMILDataset
from cellmil.interfaces.FeatureExtractorConfig import (
    ExtractorType,
    FeatureExtractionType,
)


def MILDataset(
    root: Path,
    label: str | tuple[str, str],
    folder: Path, 
    data: pd.DataFrame, 
    extractor: ExtractorType | list[ExtractorType],
    split: Literal["train", "val", "test", "all"] = "all",
    **kwargs: Any,
) -> CellMILDataset | PatchMILDataset:
    if isinstance(extractor, list) or extractor not in FeatureExtractionType.Embedding:
        return CellMILDataset(
            root=root,
            label=label,
            folder=folder,
            data=data,
            extractor=extractor,
            split=split,
            graph_creator=kwargs.get("graph_creator", None),
            segmentation_model=kwargs.get("segmentation_model", None),
            cell_type=kwargs.get("cell_type", False),
            cell_types_to_keep=kwargs.get("cell_types_to_keep", None),
            roi_folder=kwargs.get("roi_folder", None),
            force_reload=kwargs.get("force_reload", False),
            transforms=kwargs.get("transforms", None),
            label_transforms=kwargs.get("label_transforms", None),
            return_cell_types=kwargs.get("return_cell_types", True),
        )
    else:
        return PatchMILDataset(
            root=root,
            label=label,
            folder=folder,
            data=data,
            extractor=extractor,
            split=split,
            force_reload=kwargs.get("force_reload", False),
            transforms=None,  # Safety measure
            label_transforms=kwargs.get("label_transforms", None),
        )

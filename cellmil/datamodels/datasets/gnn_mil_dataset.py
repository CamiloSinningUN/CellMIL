import pandas as pd
from typing import Any, Literal, Union, Callable, Optional
from torch_geometric.data import Data  # type: ignore
from pathlib import Path
from .cell_gnn_mil_dataset import CellGNNMILDataset
from .patch_gnn_mil_dataset import PatchGNNMILDataset
from cellmil.interfaces.FeatureExtractorConfig import ExtractorType, FeatureExtractionType

def GNNMILDataset(
    root: Union[str, Path], 
    folder: Union[str, Path],
    label: str, 
    data: pd.DataFrame, 
    extractor: ExtractorType | list[ExtractorType],
    split: Literal["train", "val", "test", "all"] = "all",
    transform: Optional[Callable[[Data], Data]] = None,
    pre_transform: Optional[Callable[[Data], Data]] = None,
    pre_filter: Optional[Callable[[Data], bool]] = None,
    force_reload: bool = False,
    label_transforms: Optional[Any] = None,
    **kwargs: Any
) -> CellGNNMILDataset | PatchGNNMILDataset:
    if isinstance(extractor, list) or extractor not in FeatureExtractionType.Embedding:
        graph_creator = kwargs.get("graph_creator", None)
        if graph_creator is None:
            raise ValueError("Graph creator must be specified when using a list of extractors.")

        segmentation_model = kwargs.get("segmentation_model", None)
        if segmentation_model is None:
            raise ValueError("Segmentation model must be specified when using a list of extractors.")

        return CellGNNMILDataset(
            root=root,
            folder=folder,
            label=label,
            data=data,
            extractor=extractor,
            split=split,
            graph_creator=graph_creator,
            segmentation_model=segmentation_model,
            cell_type=kwargs.get("cell_type", False),
            cell_types_to_keep=kwargs.get("cell_types_to_keep", None),
            centroid=kwargs.get("centroid", False),
            roi_folder=kwargs.get("roi_folder", None),
            transforms=kwargs.get("transforms", kwargs.get("transform_pipeline", None)),
            label_transforms=label_transforms,
            return_cell_types= kwargs.get("return_cell_types", True),
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
            force_reload=force_reload
        )
    else:
        return PatchGNNMILDataset(
            root=root,
            folder=folder,
            label=label,
            data=data,
            extractor=extractor,
            split=split,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
            force_reload=force_reload,
            transforms=None, # Safety measure
            label_transforms=label_transforms
        )
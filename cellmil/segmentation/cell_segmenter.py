# -*- coding: utf-8 -*-
# Cell Segmenter
# This module was inspired by the CellViT project and adapted for the CellMIL framework.
#
# References:
# CellViT: Vision Transformers for precise cell segmentation and classification
# Fabian Hörst et al., Medical Image Analysis, 2024
# DOI: https://doi.org/10.1016/j.media.2024.103143

import yaml
import os
import torch
import ujson
import numpy as np
import logging
import pandas as pd
from tqdm import tqdm
from collections import deque
from typing import Any, Literal
from pathlib import Path
from torch.utils.data import DataLoader
import urllib.request
import shutil
from shapely import strtree
from shapely.geometry import Polygon, MultiPolygon
from torchvision import transforms as T  # type: ignore
from torch.nn import functional as F
from cellmil.interfaces import CellSegmenterConfig, PatchExtractorConfig
from cellmil.utils import logger
from cellmil.utils.tools import unflatten_dict, to_float, to_float_normalized
from cellmil.datamodels import WSI
from cellmil.models.segmentation import HoVerNet, CellViTSAM, CellposeSAM
from .datasets.patched_wsi_inference import PatchedWSIInference
from .utils.geojson import convert_geojson

class CellSegmenter:
    """Class for cell instance segmentation in whole slide images.

    Features reliable multiprocessing with Pool.imap and automatic fallback to
    sequential processing if multiprocessing fails.
    """

    def __init__(self, config: CellSegmenterConfig) -> None:
        self.config = config

        # TODO: Make this configurable from config
        self.geojson = True
        self.batch_size = 64
        self.mixed_precision = False
        # TODO: ---

        self.device = f"cuda:{self.config.gpu}" if torch.cuda.is_available() else "cpu"
        self._setup_wsi()
        self._setup_patch_config()
        self._download_models()
        self._load_model()
        self._load_inference_transforms()

    def set_wsi(self, wsi_path: str | Path, patched_slide_path: str | Path) -> None:
        """Set the WSI path and patched slide path for segmentation."""
        self.config.wsi_path = Path(wsi_path)
        self.config.patched_slide_path = Path(patched_slide_path)
        logger.info("WSI and patched slide paths set")
        self._setup_wsi()
        self._setup_patch_config()

    def _setup_wsi(self) -> None:
        """Setup the whole slide image for segmentation."""
        logger.info("Processing single WSI file")
        wsi_path = Path(self.config.wsi_path)
        wsi_name = wsi_path.stem
        wsi_file = WSI(
            name=wsi_name,
            patient=wsi_name,
            slide_path=wsi_path,
            patched_slide_path=self.config.patched_slide_path,
        )
        self.wsi = wsi_file

    def _setup_patch_config(self) -> None:
        metadata_path = Path(self.config.patched_slide_path) / "metadata.yaml"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        # Load and parse the YAML file
        with open(metadata_path, "r") as file:
            metadata = yaml.safe_load(file)

        # Extract the patch configuration values
        # Use defaults if the values are not found in the metadata
        patch_size = metadata.get("patch_size", 1024)
        patch_overlap = metadata.get("patch_overlap", 64)
        target_mag = metadata.get("magnification", 40.0)

        # Create a PatchExtractorConfig with the extracted values
        self.patch_config = PatchExtractorConfig(
            wsi_path=self.config.wsi_path,
            patch_size=patch_size,
            patch_overlap=patch_overlap,
            target_mag=target_mag,
            output_path=self.config.patched_slide_path,
        )

        # Validate patch size for CellposeSAM model
        if self.config.model == "cellpose_sam" and patch_size != 256:
            logger.error(
                f"CellposeSAM model requires patches of size 256, but found patch size {patch_size}. "
                f"Please re-extract patches with patch_size=256."
            )
            raise ValueError(
                f"CellposeSAM model requires patches of size 256, but found patch size {patch_size}"
            )

        logger.info(f"Patch configuration: size={patch_size}, overlap={patch_overlap}")

    def _download_models(self) -> None:
        """Download model checkpoints if they don't exist."""
        # Skip downloading for CellposeSAM as it handles its own model downloads
        if self.config.model == "cellpose_sam":
            # Create the checkpoints directory but don't download anything
            script_dir = Path(__file__).resolve()
            checkpoints_dir = script_dir.parent.parent / "checkpoints"
            checkpoints_dir.mkdir(exist_ok=True, parents=True)
            self.checkpoints_dir = checkpoints_dir
            logger.info("CellposeSAM handles its own model downloads")
            return

        # Get the parent of parent directory of this script
        script_dir = Path(__file__).resolve()
        checkpoints_dir = script_dir.parent.parent / "checkpoints"

        # Create the checkpoints directory if it doesn't exist
        checkpoints_dir.mkdir(exist_ok=True, parents=True)

        # Define model URLs and their local paths
        model_urls = {
            "cellvit_20": "https://drive.usercontent.google.com/download?id=1wP4WhHLNwyJv97AK42pWK8kPoWlrqi30&export=download&authuser=0&confirm=t&uuid=f3ea9433-f877-47d3-8a8c-b50234c4b085&at=AN8xHopWwiqwCFH-QzM8xrPx4JKY:1750773930681",
            "cellvit_40": "https://drive.usercontent.google.com/download?id=1MvRKNzDW2eHbQb5rAgTEp6s2zAXHixRV&export=download&authuser=0&confirm=t&uuid=549a273b-ba4b-43d4-b113-bdcb311b8f5f&at=AN8xHoqzbI-y5SjVdwrtaHnFpnEd:1750773931839",
            "hovernet": "https://drive.usercontent.google.com/download?id=1vHOsIASmAmOmmmFXeE4C2CA6-6LOMiCh&export=download&confirm=t&uuid=a9470991-3391-4c1d-bfb3-17c95fb3fbc6",
        }

        # Check and download each model if not present
        for model_name, url in model_urls.items():
            if self.config.model not in model_name:
                continue

            _name = f"{model_name}.pth"
            model_path = checkpoints_dir / _name
            if not model_path.exists():
                logger.info(f"Downloading {model_name} checkpoint...")
                try:
                    # Create a request with a User-Agent to avoid some download restrictions
                    req = urllib.request.Request(
                        url, headers={"User-Agent": "Mozilla/5.0"}
                    )
                    with (
                        urllib.request.urlopen(req) as response,
                        open(model_path, "wb") as out_file,
                    ):
                        shutil.copyfileobj(response, out_file)
                    logger.info(f"Downloaded {model_name} checkpoint to {model_path}")
                except Exception as e:
                    logger.error(f"Error downloading {model_name} checkpoint: {e}")
            else:
                logger.info(f"{model_name} checkpoint already exists at {model_path}")

        # Store the checkpoints directory
        self.checkpoints_dir = checkpoints_dir

    def _load_model(self) -> None:
        """Load model and checkpoint"""
        # Use the downloaded model based on config.model
        if not hasattr(self, "checkpoints_dir"):
            self._download_models()

        if self.config.model == "cellvit":
            _path = (
                "cellvit_20.pth"
                if self.wsi.metadata["magnification"] == 20
                else "cellvit_40.pth"
            )
            model_path = self.checkpoints_dir / _path
        elif self.config.model == "cellpose_sam":
            # CellposeSAM doesn't use a checkpoint file, it downloads models automatically
            model_path = None
        else:
            model_path = self.checkpoints_dir / f"{self.config.model}.pth"

        # Load checkpoint if model_path exists
        model_checkpoint = None
        if model_path is not None:
            if not model_path.exists():
                raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
            logger.info(f"Loading model: {model_path}")
            model_checkpoint = torch.load(model_path, map_location="cpu")
        else:
            logger.info("Initializing CellposeSAM model (no checkpoint needed)")

        if self.config.model in ("cellvit", "hovernet"):
            self.nuclei_types = {
                "Background": 0,
                "Neoplastic": 1,
                "Inflammatory": 2,
                "Connective": 3,
                "Dead": 4,
                "Epithelial": 5,
            }
        else:
            self.nuclei_types = {"Background": 0, "Cell": 1}

        if self.config.model == "cellvit":
            assert model_checkpoint is not None, (
                "Model checkpoint is required for cellvit"
            )
            self.run_conf = unflatten_dict(model_checkpoint["config"], ".")
            self.model = CellViTSAM(
                model_path=None,
                num_nuclei_classes=self.run_conf["data"]["num_nuclei_classes"],
                num_tissue_classes=self.run_conf["data"]["num_tissue_classes"],
                vit_structure=self.run_conf["model"]["backbone"],
                regression_loss=self.run_conf["model"].get("regression_loss", False),
            )
            logger.info(
                self.model.load_state_dict(model_checkpoint["model_state_dict"])
            )
        elif self.config.model == "hovernet":
            assert model_checkpoint is not None, (
                "Model checkpoint is required for hovernet"
            )
            self.model = HoVerNet()
            logger.info(self.model.load_state_dict(model_checkpoint))
        elif self.config.model == "cellpose_sam":
            # Initialize CellposeSAM with device parameter
            device = torch.device(self.device)
            self.model = CellposeSAM(pretrained_model="cpsam", device=device)
            logger.info("CellposeSAM model initialized successfully")
        else:
            raise NotImplementedError(
                f"Model {self.config.model} is not implemented or not supported."
            )

        self.model.eval()
        self.model.to(self.device)

    def _load_inference_transforms(self):
        """Load the inference transformations from the run_configuration"""
        logger.info("Loading inference transformations")

        if self.config.model in ("cellvit"):
            transform_settings: dict[str, Any] = self.run_conf["transformations"]
            if "normalize" in transform_settings:
                mean: tuple[float, float, float] = tuple(
                    transform_settings["normalize"].get("mean", (0.5, 0.5, 0.5))
                )  # type: ignore
                std: tuple[float, float, float] = tuple(
                    transform_settings["normalize"].get("std", (0.5, 0.5, 0.5))
                )  # type: ignore
            else:
                mean = (0.5, 0.5, 0.5)
                std = (0.5, 0.5, 0.5)
            self.inference_transforms = T.Compose(
                [
                    T.ToTensor(),
                    T.Normalize(mean=mean, std=std),  # type: ignore
                ]
            )
        elif self.config.model in ("hovernet"):
            self.inference_transforms = T.Compose(
                [
                    T.PILToTensor(),  # Converts to CHW, uint8
                    T.Lambda(to_float),  # Convert to float32, keep [0, 255] values
                ]
            )
        elif self.config.model == "cellpose_sam":
            # CellposeSAM expects RGB images in range [0, 255] as uint8 or [0, 1] as float
            self.inference_transforms = T.Compose(
                [
                    T.PILToTensor(),  # Converts to CHW, uint8
                    T.Lambda(
                        to_float_normalized
                    ),  # Convert to float32 and normalize to [0, 1]
                ]
            )

    def _get_wsi_dimensions(self) -> tuple[int, int]:
        """Get the dimensions (width, height) of the WSI at level 0.

        Returns:
            tuple[int, int]: WSI dimensions as (width, height)
        """
        try:
            # Import openslide here to avoid dependency issues if not installed
            import openslide

            # Open the WSI to get its dimensions
            slide = openslide.OpenSlide(str(self.config.wsi_path))
            # dimensions returns (width, height) at level 0
            wsi_width, wsi_height = slide.dimensions
            slide.close()
            return wsi_width, wsi_height
        except ImportError:
            logger.error(
                "OpenSlide is required to validate contour coordinates. Install python-openslide."
            )
            raise
        except Exception as e:
            logger.error(f"Failed to get WSI dimensions: {e}")
            raise

    def _are_contour_coordinates_valid(
        self, contour_global: np.ndarray[Any, Any]
    ) -> bool:
        """Check if contour coordinates are within the WSI bounds.

        Args:
            contour_global (np.ndarray): Global contour coordinates, shape (N, 2) where
                                       each point is [x, y]

        Returns:
            bool: True if all contour points are within WSI bounds, False otherwise
        """
        if not hasattr(self, "_wsi_width") or not hasattr(self, "_wsi_height"):
            # Cache WSI dimensions for performance
            self._wsi_width, self._wsi_height = self._get_wsi_dimensions()

        # Check all contour points
        # contour_global is in (x, y) format where x is horizontal and y is vertical
        x_coords = contour_global[:, 0]
        y_coords = contour_global[:, 1]

        # Check if any coordinates are outside bounds
        x_valid = np.all((x_coords >= 0) & (x_coords < self._wsi_width))
        y_valid = np.all((y_coords >= 0) & (y_coords < self._wsi_height))

        return bool(x_valid and y_valid)

    def process(self) -> None:
        """Process WSI file"""

        logger.info(f"Processing WSI: {self.wsi.name}")

        wsi_inference_dataset = PatchedWSIInference(
            self.wsi, transform=self.inference_transforms
        )

        cpu_count = os.cpu_count()
        if cpu_count is None:
            num_workers = 16
        else:
            num_workers = int(3 / 4 * cpu_count)

        num_workers = int(np.clip(num_workers, 1, 2 * self.batch_size))

        wsi_inference_dataloader = DataLoader(
            dataset=wsi_inference_dataset,
            batch_size=self.batch_size,
            # num_workers=num_workers,
            shuffle=False,
            collate_fn=wsi_inference_dataset.collate_batch,
            pin_memory=False,
        )

        if self.wsi.patched_slide_path is None:
            raise ValueError(
                "Patched slide path is not set. Cannot process WSI without patched slide path."
            )

        outdir = (
            Path(self.wsi.patched_slide_path) / "cell_detection" / self.config.model
        )
        outdir.mkdir(exist_ok=True, parents=True)

        # Log dataset info for debugging
        logger.info(f"Total patches in dataset: {len(wsi_inference_dataset)}")
        logger.info(f"WSI dimensions: {self._get_wsi_dimensions()}")
        logger.info(f"Batch size: {self.batch_size}")

        cell_dict_wsi: list[dict[str, Any]] = []  # for storing all cell information
        cell_dict_detection: list[dict[str, Any]] = []  # for storing only the centroids

        processed_patches: list[str] = []

        cell_count: int = 0

        with torch.no_grad():
            pbar = tqdm(wsi_inference_dataloader, total=len(wsi_inference_dataset))

            for batch in wsi_inference_dataloader:
                patches = batch[0].to(self.device)
                metadata = batch[1]
                if self.mixed_precision:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        predictions = self.model.forward(patches)
                else:
                    predictions = self.model.forward(patches)

                # print("DEBUGG")
                # print(predictions["hv_map"].shape)
                # reshape, apply softmax to segmentation maps
                # predictions = self.model.reshape_model_output(predictions_, self.device)

                # TODO: This might become custom for every model
                instance_types = self.get_cell_predictions(
                    predictions, magnification=self.wsi.metadata["magnification"]
                )

                # unpack each patch from batch
                for _, (patch_instance_types, patch_metadata) in enumerate(
                    zip(instance_types, metadata)
                ):
                    pbar.update(1)

                    # Log patch being processed
                    logger.debug(
                        f"Processing patch [{patch_metadata['row']}, {patch_metadata['col']}]"
                    )

                    # add global patch metadata
                    patch_cell_detection: dict[str, Any] = {}
                    patch_cell_detection["patch_metadata"] = patch_metadata
                    patch_cell_detection["type_map"] = self.nuclei_types

                    processed_patches.append(
                        f"{patch_metadata['row']}_{patch_metadata['col']}"
                    )

                    # calculate coordinate on highest magnifications
                    # wsi_scaling_factor = patch_metadata["wsi_metadata"]["downsampling"]
                    # patch_size = patch_metadata["wsi_metadata"]["patch_size"]
                    wsi_scaling_factor = self.wsi.metadata["downsampling"]
                    patch_size = self.wsi.metadata["patch_size"]
                    x_global = int(
                        patch_metadata["row"] * patch_size * wsi_scaling_factor
                        - (patch_metadata["row"] + 0.5)
                        * self.patch_config.patch_overlap
                    )
                    y_global = int(
                        patch_metadata["col"] * patch_size * wsi_scaling_factor
                        - (patch_metadata["col"] + 0.5)
                        * self.patch_config.patch_overlap
                    )

                    # extract cell information
                    for cell in patch_instance_types.values():
                        if cell["type"] == self.nuclei_types["Background"]:
                            continue
                        offset_global = np.array([x_global, y_global])
                        centroid_global = cell["centroid"] + np.flip(offset_global)
                        contour_global = cell["contour"] + np.flip(offset_global)
                        bbox_global = cell["bbox"] + offset_global

                        # Debug coordinate calculation for problematic cases
                        if not self._are_contour_coordinates_valid(contour_global):
                            logger.info(
                                f"COORDINATE DEBUG - Patch [{patch_metadata['row']}, {patch_metadata['col']}]: "
                                f"x_global={x_global}, y_global={y_global}, "
                                f"offset_global={offset_global}, "
                                f"flipped_offset={np.flip(offset_global)}, "
                                f"cell_contour_range=x[{cell['contour'][:, 0].min():.1f}, {cell['contour'][:, 0].max():.1f}], "
                                f"y[{cell['contour'][:, 1].min():.1f}, {cell['contour'][:, 1].max():.1f}], "
                                f"contour_global_range=x[{contour_global[:, 0].min():.1f}, {contour_global[:, 0].max():.1f}], "
                                f"y[{contour_global[:, 1].min():.1f}, {contour_global[:, 1].max():.1f}]"
                            )

                        # Check if contour coordinates are within WSI limits
                        if not self._are_contour_coordinates_valid(contour_global):
                            logger.warning(
                                f"Cell contour coordinates exceed WSI bounds for patch "
                                f"[{patch_metadata['row']}, {patch_metadata['col']}]. Skipping cell. "
                                f"WSI dims: [{self._wsi_width}, {self._wsi_height}], "
                                f"Contour range: x=[{contour_global[:, 0].min():.1f}, {contour_global[:, 0].max():.1f}], "
                                f"y=[{contour_global[:, 1].min():.1f}, {contour_global[:, 1].max():.1f}], "
                                f"Offset: [{x_global}, {y_global}], Patch size: {patch_size}, Scaling: {wsi_scaling_factor}"
                            )
                            continue

                        cell_dict: dict[str, Any] = {
                            "bbox": bbox_global.tolist(),
                            "centroid": centroid_global.tolist(),
                            "contour": contour_global.tolist(),
                            "type_prob": cell["type_prob"],
                            "type": cell["type"],
                            "patch_coordinates": [
                                patch_metadata["row"],
                                patch_metadata["col"],
                            ],
                            "cell_status": get_cell_position_marging(
                                cell["bbox"],
                                self.patch_config.patch_size,
                                int(self.patch_config.patch_overlap),
                            ),
                            "offset_global": offset_global.tolist(),
                        }
                        cell_detection: dict[str, Any] = {
                            "bbox": bbox_global.tolist(),
                            "centroid": centroid_global.tolist(),
                            "type": cell["type"],
                        }
                        if (
                            np.max(cell["bbox"]) == self.patch_config.patch_size
                            or np.min(cell["bbox"]) == 0
                        ):
                            position = get_cell_position(
                                cell["bbox"], self.patch_config.patch_size
                            )
                            cell_dict["edge_position"] = True
                            cell_dict["edge_information"] = {}
                            cell_dict["edge_information"]["position"] = position
                            cell_dict["edge_information"]["edge_patches"] = (
                                get_edge_patch(
                                    position,
                                    patch_metadata["row"],
                                    patch_metadata["col"],
                                )
                            )
                        else:
                            cell_dict["edge_position"] = False

                        cell_dict_wsi.append(cell_dict)
                        cell_dict_detection.append(cell_detection)

                        # get the cell token
                        bb_index = cell["bbox"] / self.patch_config.patch_size
                        bb_index[0, :] = np.floor(bb_index[0, :])
                        bb_index[1, :] = np.ceil(bb_index[1, :])
                        bb_index = bb_index.astype(np.uint8)

                        cell_count = cell_count + 1

                    pbar.set_postfix(Cells=cell_count)  # type: ignore

        # post processing
        logger.info(f"Detected cells before cleaning: {len(cell_dict_wsi)}")
        if len(cell_dict_wsi) > 0:
            keep_idx = self.post_process_edge_cells(cell_list=cell_dict_wsi)
            cell_dict_wsi = [
                {"cell_id": i, **cell_dict_wsi[idx_c]} for i, idx_c in enumerate(keep_idx)
            ]
            cell_dict_detection = [
                {"cell_id": i, **cell_dict_detection[idx_c]}
                for i, idx_c in enumerate(keep_idx)
            ]
            logger.info(f"Detected cells after cleaning: {len(keep_idx)}")
        else:
            logger.warning("No cells detected.")
            cell_dict_wsi = []
            cell_dict_detection = []
            
        

        logger.info(
            f"Processed all patches. Storing final results: {str(outdir / 'cells.json')} and cell_detection.json"
        )
        _cell_dict_wsi: dict[str, Any] = {
            "wsi_metadata": self.wsi.metadata,
            "processed_patches": processed_patches,
            "type_map": self.nuclei_types,
            "cells": cell_dict_wsi,
        }
        with open(str(outdir / "cells.json"), "w") as outfile:
            ujson.dump(_cell_dict_wsi, outfile, indent=2)

        logger.info("Converting segmentation to geojson")
        geojson_collection = convert_geojson(
            _cell_dict_wsi["cells"], True, self.config.model
        )
        with open(str(str(outdir / "cells.geojson")), "w") as outfile:
            ujson.dump(geojson_collection, outfile, indent=2)

        _cell_dict_detection: dict[str, Any] = {
            "wsi_metadata": self.wsi.metadata,
            "processed_patches": processed_patches,
            "type_map": self.nuclei_types,
            "cells": cell_dict_detection,
        }
        with open(str(outdir / "cell_detection.json"), "w") as outfile:
            ujson.dump(_cell_dict_detection, outfile, indent=2)

        logger.info("Converting detection to geojson")
        geojson_collection = convert_geojson(
            _cell_dict_wsi["cells"], False, self.config.model
        )
        with open(str(str(outdir / "cell_detection.geojson")), "w") as outfile:
            ujson.dump(geojson_collection, outfile, indent=2)

        # Generate statistics only if cells were detected
        if len(_cell_dict_wsi["cells"]) > 0:
            cell_stats_df = pd.DataFrame(_cell_dict_wsi["cells"])
            cell_stats = dict(cell_stats_df.value_counts("type"))  # type: ignore
            nuclei_types_inverse = {v: k for k, v in self.nuclei_types.items()}
            verbose_stats = {nuclei_types_inverse[k]: v for k, v in cell_stats.items()}  # type: ignore
            logger.info(f"Finished with cell detection for WSI {self.wsi.name}")
            logger.info("Stats:")
            logger.info(f"{verbose_stats}")
        else:
            logger.info(f"Finished with cell detection for WSI {self.wsi.name}")
            logger.info("Stats: No cells detected")

    def get_cell_predictions(
        self, predictions: dict[str, torch.Tensor], magnification: float | int = 40
    ) -> list[dict[np.int32, dict[str, Any]]] | list[dict[int, dict[str, Any]]]:
        """Get cell predictions from model output

        Args:
            predictions (torch.Tensor): Model output
            magnification (float, optional): Magnification of the WSI. Defaults to 40.0.

        Returns:
            dict[str, Any]: Dictionary with cell predictions
        """
        if self.config.model in ("hovernet", "cellvit"):
            predictions["nuclei_binary_map"] = F.softmax(
                predictions["nuclei_binary_map"], dim=1
            )  # shape: (batch_size, 2, H, W)
            predictions["nuclei_type_map"] = F.softmax(
                predictions["nuclei_type_map"], dim=1
            )  # shape: (batch_size, num_nuclei_classes, H, W)
            # get the instance types

            _, instance_types = self.model.calculate_instance_map(
                predictions, magnification=magnification
            )
            return instance_types
        elif self.config.model == "cellpose_sam":
            # For Cellpose-SAM, the predictions already contain the final masks
            # We just need to extract the instance information
            _, instance_types = self.model.calculate_instance_map(
                predictions, magnification=magnification
            )
            return instance_types
        else:
            raise NotImplementedError(f"Model {self.config.model} not implemented")

    def post_process_edge_cells(self, cell_list: list[dict[str, Any]]) -> list[Any]:
        """Use the CellPostProcessor to remove multiple cells and merge due to overlap

        Args:
            cell_list (List[dict]): List with cell-dictionaries. Required keys:
                * bbox
                * centroid
                * contour
                * type_prob
                * type
                * patch_coordinates
                * cell_status
                * offset_global

        Returns:
            List[int]: List with integers of cells that should be kept
        """
        cell_processor = CellPostProcessor(cell_list, logger)
        cleaned_cells = cell_processor.post_process_cells()

        return list(cleaned_cells.index.values)  # type: ignore


def get_cell_position(bbox: np.ndarray[Any, Any], patch_size: int = 1024) -> list[int]:
    """Get cell position as a list

    Entry is 1, if cell touches the border: [top, right, down, left]

    Args:
        bbox (np.ndarray): Bounding-Box of cell
        patch_size (int, optional): Patch-size. Defaults to 1024.

    Returns:
        List[int]: List with 4 integers for each position
    """
    # bbox = 2x2 array in h, w style
    # bbox[0,0] = upper position (height)
    # bbox[1,0] = lower dimension (height)
    # boox[0,1] = left position (width)
    # bbox[1,1] = right position (width)
    # bbox[:,0] -> x dimensions
    top, left, down, right = False, False, False, False
    if bbox[0, 0] == 0:
        top = True
    if bbox[0, 1] == 0:
        left = True
    if bbox[1, 0] == patch_size:
        down = True
    if bbox[1, 1] == patch_size:
        right = True
    position = [top, right, down, left]
    position = [int(pos) for pos in position]

    return position


def get_cell_position_marging(
    bbox: np.ndarray[Any, Any], patch_size: int = 1024, margin: int = 64
) -> Literal[0, 1, 2, 3, 4, 5, 6, 7, 8]:
    """Get the status of the cell, describing the cell position

    A cell is either in the mid (0) or at one of the borders (1-8)

    # Numbers are assigned clockwise, starting from top left
    # i.e., top left = 1, top = 2, top right = 3, right = 4, bottom right = 5 bottom = 6, bottom left = 7, left = 8
    # Mid status is denoted by 0

    Args:
        bbox (np.ndarray): Bounding Box of cell
        patch_size (int, optional): Patch-Size. Defaults to 1024.
        margin (int, optional): Margin-Size. Defaults to 64.

    Returns:
        int: Cell Status
    """
    cell_status = None
    if np.max(bbox) > patch_size - margin or np.min(bbox) < margin:
        if bbox[0, 0] < margin:
            # top left, top or top right
            if bbox[0, 1] < margin:
                # top left
                cell_status = 1
            elif bbox[1, 1] > patch_size - margin:
                # top right
                cell_status = 3
            else:
                # top
                cell_status = 2
        elif bbox[1, 1] > patch_size - margin:
            # top right, right or bottom right
            if bbox[1, 0] > patch_size - margin:
                # bottom right
                cell_status = 5
            else:
                # right
                cell_status = 4
        elif bbox[1, 0] > patch_size - margin:
            # bottom right, bottom, bottom left
            if bbox[0, 1] < margin:
                # bottom left
                cell_status = 7
            else:
                # bottom
                cell_status = 6
        elif bbox[0, 1] < margin:
            # bottom left, left, top left, but only left is left
            cell_status = 8
        else:
            cell_status = 0
    else:
        cell_status = 0

    return cell_status


def get_edge_patch(position: list[int], row: int, col: int):
    # row starting on bottom or on top?
    if position == [1, 0, 0, 0]:
        # top
        return [[row - 1, col]]
    if position == [1, 1, 0, 0]:
        # top and right
        return [[row - 1, col], [row - 1, col + 1], [row, col + 1]]
    if position == [0, 1, 0, 0]:
        # right
        return [[row, col + 1]]
    if position == [0, 1, 1, 0]:
        # right and down
        return [[row, col + 1], [row + 1, col + 1], [row + 1, col]]
    if position == [0, 0, 1, 0]:
        # down
        return [[row + 1, col]]
    if position == [0, 0, 1, 1]:
        # down and left
        return [[row + 1, col], [row + 1, col - 1], [row, col - 1]]
    if position == [0, 0, 0, 1]:
        # left
        return [[row, col - 1]]
    if position == [1, 0, 0, 1]:
        # left and top
        return [[row, col - 1], [row - 1, col - 1], [row - 1, col]]


class CellPostProcessor:
    def __init__(
        self,
        cell_list: list[dict[str, Any]],
        logger: logging.Logger,
    ) -> None:
        """Post-Processing a list of cells from one WSI

        Args:
            cell_list (List[dict]): List with cell-dictionaries. Required keys:
                * bbox
                * centroid
                * contour
                * type_prob
                * type
                * patch_coordinates
                * cell_status
                * offset_global
            logger (logging.Logger): Logger
        """
        self.logger = logger
        self.logger.info("Initializing Cell-Postprocessor")
        self.cell_df: pd.DataFrame = pd.DataFrame(cell_list)
        
        # Fast vectorized coordinate conversion
        self.logger.info(f"DataFrame has {len(self.cell_df)} rows, using vectorized processing")
        # Extract x, y coordinates from patch_coordinates column
        coords = pd.DataFrame(self.cell_df['patch_coordinates'].tolist(), columns=['x', 'y'], index=self.cell_df.index)
        # Convert to string format using vectorized operations
        self.cell_df['patch_coordinates'] = coords['x'].astype(str) + '_' + coords['y'].astype(str)

        self.mid_cells = self.cell_df[  # type: ignore
            self.cell_df["cell_status"] == 0  # type: ignore
        ]  # cells in the mid
        self.cell_df_margin = self.cell_df[  # type: ignore
            self.cell_df["cell_status"] != 0  # type: ignore
        ]  # cells either torching the border or margin

    def post_process_cells(self) -> pd.DataFrame:
        """Main Post-Processing coordinator, entry point

        Returns:
            pd.DataFrame: DataFrame with post-processed and cleaned cells
        """
        self.logger.info("Finding edge-cells for merging")
        cleaned_edge_cells = self._clean_edge_cells()
        self.logger.info("Removal of cells detected multiple times")
        cleaned_edge_cells = self._remove_overlap(cleaned_edge_cells)

        # merge with mid cells
        postprocessed_cells = pd.concat(
            [self.mid_cells, cleaned_edge_cells]  # type: ignore
        ).sort_index()  # type: ignore
        return postprocessed_cells

    def _clean_edge_cells(self) -> pd.DataFrame:
        """Create a DataFrame that just contains all margin cells (cells inside the margin, not touching the border)
        and border/edge cells (touching border) with no overlapping equivalent (e.g, if patch has no neighbour)

        Returns:
            pd.DataFrame: Cleaned DataFrame
        """

        margin_cells = self.cell_df_margin[  # type: ignore
            self.cell_df_margin["edge_position"] == 0  # type: ignore
        ]  # cells at the margin, but not touching the border
        edge_cells = self.cell_df_margin[  # type: ignore
            self.cell_df_margin["edge_position"] == 1  # type: ignore
        ]  # cells touching the border
        existing_patches = list(set(self.cell_df_margin["patch_coordinates"].to_list()))  # type: ignore

        edge_cells_unique = pd.DataFrame(
            columns=self.cell_df_margin.columns  # type: ignore
        )  # cells torching the border without having an overlap from other patches 

        # Add progress bar for processing edge cells
        edge_cells_pbar = tqdm( # type: ignore
            edge_cells.iterrows(), total=len(edge_cells), desc="Processing edge cells" # type: ignore
        )  
        for idx, cell_info in edge_cells_pbar:  # type: ignore
            edge_information = dict(cell_info["edge_information"])  # type: ignore
            try:
                edge_patch = edge_information["edge_patches"][0]  # type: ignore
                edge_patch = f"{edge_patch[0]}_{edge_patch[1]}"
                if edge_patch not in existing_patches:
                    edge_cells_unique.loc[idx, :] = cell_info  # type: ignore
            except (TypeError, IndexError, KeyError) as e:
                self.logger.warning(
                    f"Skipping edge cell {idx} due to invalid edge_patches data: {e}. "
                    f"Edge information: {edge_information}"
                )
                continue

        cleaned_edge_cells = pd.concat([margin_cells, edge_cells_unique])  # type: ignore

        return cleaned_edge_cells.sort_index()  # type: ignore

    def _remove_overlap(self, cleaned_edge_cells: pd.DataFrame) -> pd.DataFrame:
        """Remove overlapping cells from provided DataFrame

        Args:
            cleaned_edge_cells (pd.DataFrame): DataFrame that should be cleaned

        Returns:
            pd.DataFrame: Cleaned DataFrame
        """
        merged_cells = cleaned_edge_cells

        for iteration in range(20):
            # Create list of polygons and corresponding indices
            poly_list: list[Polygon] = []
            poly_indices: list[Any] = []
            for idx, cell_info in merged_cells.iterrows():  # type: ignore
                poly: Polygon = Polygon(cell_info["contour"])  # type: ignore
                if not poly.is_valid:
                    self.logger.debug("Found invalid polygon - Fixing with buffer 0")
                    multi = poly.buffer(0)
                    if isinstance(multi, MultiPolygon):
                        if len(multi.geoms) > 1:
                            poly_idx = np.argmax([p.area for p in multi.geoms])
                            poly = multi.geoms[poly_idx]
                        else:
                            poly = multi.geoms[0]
                    else:
                        poly = multi
                poly_list.append(poly)
                poly_indices.append(idx)

            # Use STRtree with explicit indices rather than storing uid on the geometry
            tree = strtree.STRtree(poly_list)

            merged_idx: deque[Any] = deque()
            iterated_cells: set[Any] = set()
            overlaps = 0

            for i, query_poly in enumerate(poly_list):
                query_idx = poly_indices[i]
                if query_idx not in iterated_cells:
                    # Get indices of intersecting polygons
                    intersected_indices = tree.query(query_poly, predicate="intersects")
                    intersected_polygons: list[Polygon] = [
                        poly_list[j] for j in intersected_indices
                    ]
                    intersected_ids = [poly_indices[j] for j in intersected_indices]

                    if (
                        len(intersected_polygons) > 1
                    ):  # we have at least one intersection with another cell
                        submergers: list[
                            Polygon
                        ] = []  # all cells that overlap with query
                        submerger_indices: list[Any] = []  # corresponding indices

                        for j, inter_poly in enumerate(intersected_polygons):
                            inter_idx = intersected_ids[j]
                            if (
                                inter_idx != query_idx
                                and inter_idx not in iterated_cells
                            ):
                                intersection = query_poly.intersection(inter_poly)
                                query_area = query_poly.area
                                inter_area = inter_poly.area

                                if (
                                    intersection.area / query_area > 0.01
                                    or intersection.area / inter_area > 0.01
                                ):
                                    overlaps = overlaps + 1
                                    submergers.append(inter_poly)
                                    submerger_indices.append(inter_idx)
                                    iterated_cells.add(inter_idx)

                        # catch block: empty list -> some cells are touching, but not overlapping strongly enough
                        if len(submergers) == 0:
                            merged_idx.append(query_idx)
                        else:  # merging strategy: take the biggest cell, other merging strategies needs to get implemented
                            areas = [poly.area for poly in submergers]
                            selected_poly_index = np.argmax(areas)
                            selected_poly_uid = submerger_indices[selected_poly_index]
                            merged_idx.append(selected_poly_uid)
                    else:
                        # no intersection, just add
                        merged_idx.append(query_idx)
                    iterated_cells.add(query_idx)

            self.logger.info(
                f"Iteration {iteration}: Found overlap of # cells: {overlaps}"
            )
            if overlaps == 0:
                self.logger.info("Found all overlapping cells")
                break
            elif iteration == 20:
                self.logger.info(
                    f"Not all doubled cells removed, still {overlaps} to remove. For perfomance issues, we stop iterations now. Please raise an issue in git or increase number of iterations."
                )
            merged_cells = cleaned_edge_cells.loc[
                cleaned_edge_cells.index.isin(merged_idx)  # type: ignore
            ].sort_index()

        return merged_cells.sort_index()  # type: ignore



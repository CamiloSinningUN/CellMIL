import json
import cv2
import traceback
import torch
import radiomics  # type: ignore
import numpy as np
import yaml
from typing import Any, Optional, Generator, cast
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool
from torch.utils.data import Dataset, DataLoader

from .extractor import Extractor
from cellmil.utils import logger
from cellmil.interfaces import FeatureExtractorConfig
from cellmil.interfaces.FeatureExtractorConfig import FeatureExtractionType, ExtractorType
from .extractor import MorphologicalExtractor, TopologicalExtractor, EmbeddingExtractor
from cellmil.utils.tools import get_cpu_count

pyradiomics_logger = radiomics.logger
pyradiomics_logger.setLevel("ERROR")

# Per-process extractor for multiprocessing workers to avoid recreating it per task
_PROCESS_EXTRACTOR: Optional[Extractor] = None

def _pool_init(extractor_type: ExtractorType) -> None:
    """Initializer for multiprocessing Pool to create one Extractor per process."""
    global _PROCESS_EXTRACTOR
    _PROCESS_EXTRACTOR = Extractor.create(extractor_type)  # type: ignore


class FeatureExtractor:
    """Feature extractor for whole slide images."""

    def __init__(self, config: FeatureExtractorConfig):
        self.config = config
        self.config.patched_slide_path = Path(self.config.patched_slide_path)
        self.config.wsi_path = Path(self.config.wsi_path) if self.config.wsi_path else None

        # TODO: Make configurable
        self.parallel = True
        self.test = False
        # TODO: -----

        if self.test or self.config.extractor in FeatureExtractionType.Embedding :
            self.n_workers = 1
        elif self.config.extractor in FeatureExtractionType.Morphological:
            self.n_workers = max(1, min(get_cpu_count() - 1, 61))
        elif self.config.extractor in FeatureExtractionType.Topological:
            self.n_workers = max(1, min(get_cpu_count() - 1, 8))

        logger.info(f"Feature extractor initialized with config: {self.config}")
        logger.info(f"Number of workers: {self.n_workers}")

        self._load_metadata()
        if self.config.extractor in FeatureExtractionType.Morphological or self.config.extractor in FeatureExtractionType.Topological:
            self._load_cells()
        if self.config.extractor in FeatureExtractionType.Topological:
            self._load_graph()
        if self.config.extractor in FeatureExtractionType.Embedding:
            self._load_patches()

    def _load_cells(self):
        """Load cell data from the specified path."""
        if self.config.segmentation_model is None:
            raise ValueError("Segmentation model must be specified.")

        json_path = (
            self.config.patched_slide_path
            / "cell_detection"
            / str(self.config.segmentation_model)
            / "cells.json"
        )
        
        if not json_path.exists():
            raise FileNotFoundError(f"Cells path {json_path} does not exist.")

        with open(json_path, "r") as f:
            self.json = json.load(f)
        self.cells: list[dict[str, Any]] = self.json["cells"]

        logger.info(f"Loaded {len(self.cells)} cells from {json_path}")

    def _load_metadata(self):
        """Load metadata from the patched slide path to get patch size and overlap."""
        metadata_path = self.config.patched_slide_path / "metadata.yaml"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")    
        
        with open(metadata_path, "r") as file:
            self.metadata = yaml.safe_load(file)
            
        # Extract key parameters
        self.patch_size = self.metadata.get("patch_size", 256)
        self.patch_overlap = self.metadata.get("patch_overlap", 0)
        self.target_mag = self.metadata.get("magnification", 40.0)

        logger.info(f"Loaded metadata: patch_size={self.patch_size}, patch_overlap={self.patch_overlap}, target_mag={self.target_mag}")

    def _load_graph(self):
        """Load graph from the specified path"""
        if self.config.segmentation_model is None:
            raise ValueError("Segmentation model must be specified.")
        if self.config.graph_method is None:
            raise ValueError("Graph method must be specified.")
        
        graph_path = (
            self.config.patched_slide_path
            / "graphs"
            / str(self.config.graph_method)
            / str(self.config.segmentation_model)
            / "graph.pt"
        )

        if not graph_path.exists():
            raise FileNotFoundError(f"Graph path {graph_path} does not exist.")

        _graph = torch.load(graph_path, map_location="cpu", weights_only=False)
        self.graph = {
            "node_features": _graph["node_features"],
            "edge_indices": _graph["edge_indices"],
            "edge_features": _graph["edge_features"],
        }

        logger.info(f"Loaded graph from {graph_path}")
        
    def _load_patches(self):
        folder = (
            self.config.patched_slide_path
            / "patches"
        )
        
        if not folder.exists() or not folder.is_dir():
            raise FileNotFoundError(f"Patches folder {folder} does not exist or is not a directory.")
        
        # Get all PNG files in the patches folder
        self.patches = list(folder.glob("*.png"))
        logger.info(f"Loaded {len(self.patches)} patch paths from {folder}")

    def get_features(self):
        """Extract features from the whole slide image using parallel processing."""
        
        logger.info(
            f"Extracting features for {self.config.patched_slide_path}..."
        )
        
        output_path = None
        
        if self.config.extractor in FeatureExtractionType.Morphological:
            def _arg_iter() -> Generator[dict[str, Any], None, None]:
                for cell in self.cells:
                    yield {
                        "cell_data": cell,
                        "patch_size": self.patch_size,
                        "patch_overlap": self.patch_overlap,
                        "extractor_name": str(self.config.extractor),
                        "patched_slide_path": str(self.config.patched_slide_path),
                    }
            
            total_tasks = len(self.cells)
            chunksize = max(1, min(1000, total_tasks // max(1, self.n_workers * 8)))
            logger.info(f"Multiprocessing chunksize: {chunksize}")

            results: list[dict[str, Any]] = []
            with Pool(
                self.n_workers,
                initializer=_pool_init,
                initargs=(str(self.config.extractor),),
                maxtasksperchild=500,  
            ) as pool:
                for res in tqdm(
                    pool.imap_unordered(
                        FeatureExtractor._get_morphological_features, _arg_iter(), chunksize=chunksize
                    ),
                    total=total_tasks,
                    desc="Extracting features",
                    mininterval=0.5,
                ):
                    if res is not None:
                        results.append(res)

            logger.info(
                f"Successfully extracted features for {len(results)} cells out of {len(self.cells)} total cells"
            )
            
            output_path = self._save_pt(results)        
            
        elif self.config.extractor in FeatureExtractionType.Topological:
            # if self.config.extractor in [ExtractorType.connectivity, ExtractorType.structure]:
            # Initialize the extractor
            _pool_init(self.config.extractor)

            results: list[dict[str, Any] ] = []
            for node in tqdm(self.graph["node_features"], desc="Extracting topological features"):
                res = self._get_topological_features({
                    "cell_id": node, 
                    "graph": self.graph, 
                    "cells": self.cells, 
                    "extractor_name": str(self.config.extractor)
                })
                if res is not None:
                    results.append(res)

            output_path = self._save_pt(results)
        # else:
            #     def _arg_iter() -> Generator[dict[str, Any], None, None]:
            #         for node in self.graph["node_features"]:
            #             yield {
            #                 "cell_id": node,
            #                 "graph": self.graph.copy(),
            #                 "cells": self.cells.copy(),
            #                 "extractor_name": str(self.config.extractor),
            #             }

            #     total_tasks = len(self.graph["node_features"])
            #     chunksize = max(1, min(1000, total_tasks // max(1, self.n_workers * 8)))
            #     logger.info(f"Multiprocessing chunksize: {chunksize}")
                
            #     results: list[dict[str, Any]] = []
            #     with Pool(
            #         self.n_workers,
            #         initializer=_pool_init,
            #         initargs=(str(self.config.extractor),),
            #     ) as pool:
            #         for res in tqdm(
            #             pool.imap_unordered(
            #                 FeatureExtractor._get_topological_features, _arg_iter(), chunksize=chunksize
            #             ),
            #             total=total_tasks,
            #             desc="Extracting features",
            #             mininterval=0.5,
            #         ):
            #             if res is not None:
            #                 results.append(res)

            #     logger.info(
            #         f"Successfully extracted features for {len(results)} cells out of {len(self.cells)} total cells"
            #     )

            #     output_path = self._save_pt(results)

        elif self.config.extractor in FeatureExtractionType.Embedding:
            _pool_init(self.config.extractor)
            results = self._get_embedding_features({
                "patches": self.patches,
                "extractor_name": str(self.config.extractor)
            })

            output_path = self._save_pt(results)

        logger.info(
            f"Results saved to {output_path}" if output_path else "No results to save"
        )
        
    @staticmethod
    def _load_patch(patched_slide_path: Path, row: int, col: int) -> np.ndarray[Any, Any]:
        """Load a patch image from the file system.

        Args:
            patched_slide_path (Path): Path to the patched slide directory
            row (int): Row coordinate of the patch
            col (int): Column coordinate of the patch

        Returns:
            np.ndarray: The patch image as a numpy array
        
        Raises:
            FileNotFoundError: If the patch file doesn't exist
        """
        if isinstance(patched_slide_path, str):
            patched_slide_path = Path(patched_slide_path)
            
        patch_path = FeatureExtractor._get_patch_path(patched_slide_path, row, col)

        if patch_path.exists():
            patch = cv2.imread(str(patch_path))
                
            # Convert BGR to RGB
            patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
            return patch
        else:
            logger.error(
                f"Patch file {patch_path} does not exist."
            )
            raise FileNotFoundError(
                f"Patch file {patch_path} does not exist. Please ensure the patch extraction step was completed."
            )
    
    @staticmethod
    def _get_patch_path(patched_slide_path: Path, row: int, col: int) -> Path:
        """Get the path to a patch file based on row and column coordinates.

        Args:
            patched_slide_path (Path): Path to the patched slide directory
            row (int): Row coordinate of the patch
            col (int): Column coordinate of the patch

        Returns:
            Path: Full path to the patch file
        """
        wsi_name = patched_slide_path.name
        patch_filename = f"{wsi_name}_{row}_{col}.png"
        patch_path = patched_slide_path / "patches" / patch_filename
        return patch_path
    
    def _save_pt(self, results: list[dict[str, Any]]):
        """Save extracted features to a PyTorch file. (Faster than JSON)"""
        logger.info("Saving features to PyTorch file")
        
        # Create output directory if it doesn't exist
        if self.config.extractor in FeatureExtractionType.Morphological:
            output_dir = (
                self.config.patched_slide_path
                / "feature_extraction"
                / str(self.config.extractor)
                / str(self.config.segmentation_model)
            )
        elif self.config.extractor in FeatureExtractionType.Topological:
            output_dir = (
                self.config.patched_slide_path
                / "feature_extraction"
                / str(self.config.extractor)
                / str(self.config.graph_method)
                / str(self.config.segmentation_model)
            )
        else:
            output_dir = (
                self.config.patched_slide_path
                / "feature_extraction"
                / str(self.config.extractor)
            )

        output_dir.mkdir(parents=True, exist_ok=True)

        filename = "features.pt"
        output_path = output_dir / filename

        # Save to PyTorch file
        try:
            if not results or len(results) == 0:
                if self.config.extractor in FeatureExtractionType.Embedding:
                    output: dict[str, Any] = {
                        "features": torch.empty((0, 0), dtype=torch.float32),
                        "patch_indices": {},  # Empty dictionary
                        "feature_names": [],  # Empty list
                    }
                else:
                    output = {
                        "features": torch.empty((0, 0), dtype=torch.float32),
                        "cell_indices": {},  # Empty dictionary
                        "feature_names": [],  # Empty list
                    }
                torch.save(output, output_path)
                logger.info(f"Successfully saved empty features file to {output_path}")
                logger.info("Feature tensor shape: (0, 0)")
            else:
                # Create tensor from feature dictionaries
                features_tensor = self._results_to_tensor(results)

                feature_names = list(results[0]["features"].keys())
                feature_names.sort()  # Sort to ensure consistent order with tensor
                
                # Create identifier to index mapping dictionary
                if self.config.extractor in FeatureExtractionType.Embedding:
                    # For embeddings, use patch_name as identifier
                    patch_indices = {item["patch_name"]: i for i, item in enumerate(results)}
                    output: dict[str, Any] = {
                        "features": features_tensor,
                        "patch_indices": patch_indices,  # Dictionary mapping identifier to tensor index
                        "feature_names": feature_names,  # List of feature names in tensor order
                    }
                else:
                    # For morphological/topological, use cell_id as identifier
                    cell_indices = {item["cell_id"]: i for i, item in enumerate(results)}
                    output = {
                        "features": features_tensor,
                        "cell_indices": cell_indices,  # Dictionary mapping identifier to tensor index
                        "feature_names": feature_names,  # List of feature names in tensor order
                    }

                torch.save(output, output_path)
                logger.info(f"Successfully saved {len(results)} results to {output_path}")
                logger.info(f"Feature tensor shape: {features_tensor.shape}")
            return output_path
        except Exception as e:
            logger.info(f"Error saving results to {output_path}: {e}")
            return None
        
    def _results_to_tensor(self, data: list[dict[str, Any]]) -> torch.Tensor:
        """Convert a dictionary of features to a PyTorch tensor.

        Args:
            data: List of dictionaries containing cell_id and features

        Returns:
            torch.Tensor: Tensor of shape [N, D] where N is the number of cells
                         and D is the number of features
        """
        if not data or len(data) == 0:
            logger.warning("No data to convert to tensor")
            return torch.empty((0, 0), dtype=torch.float32)

        # Get feature names from the first element
        feature_names = list(data[0]["features"].keys())
        feature_names.sort()  # Sort to ensure consistent order

        # Create a tensor of the appropriate size
        n_cells = len(data)
        n_features = len(feature_names)
        features_tensor = torch.zeros((n_cells, n_features), dtype=torch.float32)

        # Fill in the tensor
        for i, item in enumerate(tqdm(data, desc="Converting features to tensor")):
            # Get identifier for logging - handle both cell_id and patch_name
            identifier = item.get("cell_id") or item.get("patch_name", f"item_{i}")
            
            for j, feature_name in enumerate(feature_names):
                # Get feature value, defaulting to 0.0 if not present
                value = item["features"].get(feature_name, 0.0)

                # Convert to float if it's a scalar, otherwise use first element
                try:
                    if isinstance(value, (int, float, np.ndarray)):
                        features_tensor[i, j] = float(value) # type: ignore
                    elif hasattr(value, "__len__") and len(value) > 0:
                        features_tensor[i, j] = float(value[0])
                    else:
                        features_tensor[i, j] = 0.0
                        logger.warning(
                            f"Using default value 0.0 for feature {feature_name} (identifier {identifier}): incompatible type {type(value)}"
                        )
                except (ValueError, TypeError) as e:
                    features_tensor[i, j] = 0.0
                    logger.warning(
                        f"Error converting feature {feature_name} for identifier {identifier}: {e}"
                    )

        logger.info(f"Created features tensor with shape {features_tensor.shape}")
        return features_tensor

    # ---- MORPHOLOGICAL EXTRACTION ----
    
    @staticmethod
    def _get_morphological_features(item: dict[str, Any]) -> Optional[dict[str, Any]]:
        """Process a single cell and extract features."""

        i = int(item["cell_data"]["cell_id"])  # type: ignore
        cell = dict(item["cell_data"])  # type: ignore
        patch_size = item["patch_size"]  # type: ignore
        extractor_name = item["extractor_name"]  # type: ignore
        patched_slide_path = Path(item.get("patched_slide_path", ""))  # type: ignore
        patch_overlap = item.get("patch_overlap", 0)  # type: ignore

        # Use per-process extractor if available to avoid re-initialization overhead
        global _PROCESS_EXTRACTOR
        extractor = _PROCESS_EXTRACTOR if _PROCESS_EXTRACTOR is not None else Extractor.create(extractor_name)  # type: ignore

        try:
            # Check if contour is valid
            if not cell["contour"] or len(cell["contour"]) < 3:
                logger.warning(
                    f"Skipping cell {i}: Invalid contour (need at least 3 points)"
                )
                return None
                
            # Check if we can use pre-extracted patch
            if patched_slide_path and patched_slide_path.exists() and "patch_coordinates" in cell:
                try:
                    row, col = cell["patch_coordinates"]
                except (ValueError, TypeError, KeyError):
                    raise ValueError(f"Invalid patch_coordinates for cell {i}")
            else:
                raise ValueError(f"Missing patch_coordinates for cell {i}")

            # Get bounding box from cell contour to determine patch region
            contour_points = np.array(cell["contour"])
            x_coords = contour_points[:, 0]
            y_coords = contour_points[:, 1]

            y_min, y_max = y_coords.min(), y_coords.max()
            x_min, x_max = x_coords.min(), x_coords.max()

            # Add some padding
            pad = patch_size // 2
            patch = None
            mask = None
            
            if x_max - x_min <= 0 or y_max - y_min <= 0:
                logger.warning(
                    f"Invalid patch size for cell {i}: x_min={x_min}, x_max={x_max}, y_min={y_min}, y_max={y_max}"
                )
                return None
                
            # Prepare patch and mask
            patch_shape = (patch_size, patch_size)

            # Use pre-extracted patch
            row, col = cell["patch_coordinates"]
            patch = FeatureExtractor._load_patch(patched_slide_path, row, col)

            x_global = int(
                row * patch_size
                - (row + 0.5)
                * patch_overlap
            )
            y_global = int(
                col * patch_size
                - (col + 0.5)
                * patch_overlap
            )
            
            # Create mask
            mask, mask_area = FeatureExtractor.create_cell_mask(
                cell["contour"],
                (x_global, y_global, x_global + patch_shape[1], y_global + patch_shape[0]),
                patch_shape,
            )
            
            if mask_area == 0:
                logger.warning(f"Skipping cell {i}: Empty mask created")
                return None
            
            # feature extraction
            if not isinstance(extractor, MorphologicalExtractor):
                raise TypeError(f"Expected MorphologicalExtractor, got {type(extractor)}")

            features = extractor.extract_features(patch, mask)

            # Time result preparation
            # Clean features (remove metadata)
            clean_features = {
                k: v
                for k, v in features.items()
                if not k.startswith("diagnostics_")
            }  # type: ignore

            # Add cell metadata
            result: dict[str, Any] = {
                "cell_id": i,
                "features": clean_features,
            }
            
            # Clean up memory
            del patch, mask, features, clean_features
            del contour_points, x_coords, y_coords
            del x_min, y_min, x_max, y_max, pad

            return result

        except Exception as e:
            logger.warning(f"Error processing cell {i}: {e}")
            traceback.print_exc()
            return None

    @staticmethod
    def create_cell_mask(
        cell_contour: list[list[int]],
        patch_bounds: tuple[int, int, int, int],
        patch_shape: tuple[int, int],
    ) -> tuple[np.ndarray[Any, Any], int]:
        """
        Create a binary mask from cell contour for a specific patch region only.

        Args:
            cell_contour: List of [x, y] coordinates defining the cell boundary
            patch_bounds: Tuple of (x_min, y_min, x_max, y_max) defining the patch region
            patch_shape: Tuple of (height, width) for the output mask

        Returns:
            Binary mask array with shape patch_shape
        """
        x_min, y_min, _, _ = patch_bounds

        mask = np.zeros(patch_shape, dtype=np.uint8)

        if not cell_contour or len(cell_contour) < 3:
            logger.warning(
                f"Invalid contour with {len(cell_contour) if cell_contour else 0} points"
            )
            return mask, 0

        # Translate contour coordinates to patch-relative coordinates (vectorized)
        contour_array = np.array(cell_contour)
        translated_contour_array = contour_array - np.array([y_min, x_min])
        
        # Filter points that are within patch bounds (vectorized)
        x_valid = (translated_contour_array[:, 0] >= 0) & (translated_contour_array[:, 0] < patch_shape[1])
        y_valid = (translated_contour_array[:, 1] >= 0) & (translated_contour_array[:, 1] < patch_shape[0])
        valid_mask = x_valid & y_valid
        
        # Get valid translated points
        translated_contour_array = translated_contour_array[valid_mask]
        
        # Log warnings for invalid points if any
        if not np.all(valid_mask):
            invalid_points = contour_array[~valid_mask]
            invalid_translated = (contour_array - np.array([x_min, y_min]))[~valid_mask]
            for orig_point, trans_point in zip(invalid_points, invalid_translated):
                logger.warning(
                    f"Point {orig_point} translated to ({trans_point[0]}, {trans_point[1]}) is outside patch bounds"
                )

        # If we have enough points after translation, create the mask
        if len(translated_contour_array) >= 3:
            contour_array = translated_contour_array.astype(np.int32)
            cv2.fillPoly(mask, [contour_array], 1)  # type: ignore

            area = np.sum(mask)
            if area == 0:
                logger.warning(
                    f"Empty mask created for contour with {len(translated_contour_array)} translated points"
                )
            return mask, int(area)
        else:
            logger.warning(
                f"Not enough points ({len(translated_contour_array)}) in patch region for contour"
            )

        return mask, 0

    # ---- TOPOLOGICAL EXTRACTION ----

    @staticmethod
    def _get_topological_features(item: dict[str, Any]) -> Optional[dict[str, Any]]:
        """Process a single cell and extract features."""
        cell_id = item["cell_id"]
        graph = item["graph"]
        cells = item["cells"]
        extractor_name = item["extractor_name"]

        global _PROCESS_EXTRACTOR
        extractor = _PROCESS_EXTRACTOR if _PROCESS_EXTRACTOR is not None else Extractor.create(extractor_name)
        
        try:
            # feature extraction
            if not isinstance(extractor, TopologicalExtractor):
                raise TypeError(f"Expected TopologicalExtractor, got {type(extractor)}")
            
            features = extractor.extract_features(cell_id, graph, cells)
        
            result: dict[str, Any] = {
                "cell_id": cell_id.item(),
                "features": features,
            }
            
            return result

        except Exception as e:
            logger.warning(f"Error processing node/cell {cell_id.item()}: {e}")
            traceback.print_exc()
            return None

    # ---- EMBEDDING EXTRACTION ----
    
    @staticmethod
    def _get_embedding_features(item: dict[str, Any]) -> list[dict[str, Any]]:
        patches = cast(list[str], item["patches"])
        extractor_name = item["extractor_name"]

        global _PROCESS_EXTRACTOR
        extractor = _PROCESS_EXTRACTOR if _PROCESS_EXTRACTOR is not None else Extractor.create(extractor_name)

        if not isinstance(extractor, EmbeddingExtractor):
            raise TypeError(f"Expected EmbeddingExtractor, got {type(extractor)}")

        class PatchesDataset(Dataset[torch.Tensor]):
            def __init__(self, patches: list[str]):
                self.patches = patches

            def __len__(self):
                return len(self.patches)

            def __getitem__(self, idx: int) -> torch.Tensor:
                patch = self.patches[idx]
                image = cv2.imread(str(patch))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # Convert to tensor for consistency with extractor expectations
                image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()  # type: ignore
                return image_tensor

        dataset = PatchesDataset(patches)
        dataloader = DataLoader(
            dataset, 
            batch_size=32, 
            shuffle=False, 
            num_workers=4
        )

        features: list[torch.Tensor] = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Extracting features"):
                batch_features = extractor.extract_features(batch)
                features.append(batch_features)
                
        # Concatenate all features from batches
        all_features = torch.cat(features, dim=0)

        # Build list of dictionaries mapping patch name to features
        results:list[dict[str, Any]] = []
        for i, patch_path in tqdm(enumerate(patches), desc="Mapping patches to features"):
            patch_name = Path(patch_path).stem
            # Extract just the row_col coordinates from the patch name
            # Assuming format: slidename_row_col.png -> extract "row_col"
            parts = patch_name.split('_')
            if len(parts) >= 2:
                # Take the last two parts as row_col
                patch_coordinates = f"{parts[-2]}_{parts[-1]}"
            else:
                # Fallback to full name if format is unexpected
                patch_coordinates = patch_name
            
            feature_dict = {str(j): all_features[i, j].item() for j in range(all_features.shape[1])}
            results.append({
                "patch_name": patch_coordinates,
                "features": feature_dict
            })

        return results

        

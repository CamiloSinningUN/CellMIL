# -*- coding: utf-8 -*-
# Patch Extractor Module
# 
# References:
# CellViT: Vision Transformers for precise cell segmentation and classification
# Fabian Hörst et al., Medical Image Analysis, 2024
# DOI: https://doi.org/10.1016/j.media.2024.103143


import numpy as np
import random
import os
import json
import multiprocessing
from multiprocessing import Process
from tqdm import tqdm
from shutil import rmtree
from typing import Union, Any, List, Type
from pathlib import Path
from openslide import OpenSlide
from shapely.geometry import Polygon
from natsort import natsorted
from PIL.Image import Image as ImageType
from PIL import Image

from cellmil.interfaces import PatchExtractorConfig
from cellmil.utils import logger
from cellmil.utils.tools import module_exists, start_timer, end_timer
from .storage import Storage
from .utils.patch_util import (
    get_files_from_dir,
    is_power_of_two,
    patch_to_tile_size,
    target_mag_to_downsample,
    compute_interesting_patches,
    generate_thumbnails,
    pad_tile,
    calculate_background_ratio,
    get_intersected_labels,
    DeepZoomGeneratorOS
)
from .utils.exceptions import WrongParameterException



def queue_worker(q: Any, store: Storage, processed_count: Any) -> None:
    """Queue Worker to save patches with metadata

    Args:
        q (Any): Queue for input
        store (Storage): Storage object
        processed_count (multiprocessing.Value): Processed element count for tqdm, shared between processes
    """
    while True:
        item: Any = q.get()
        if item is None:
            break

        # check if size matches, otherwise rescale in multiprocessing
        # TODO: check for context patches and masks!
        item = list(item)
        tile = item[0]
        tile_size = tile.shape[0]
        target_tile_size = item[-1]
        if tile_size != target_tile_size:
            tile = Image.fromarray(tile)
            if tile_size > target_tile_size:
                tile.thumbnail(
                    (target_tile_size, target_tile_size),
                    getattr(Image, "Resampling", Image).LANCZOS,  # type: ignore[call-arg]
                )
            else:
                tile = tile.resize(
                    (target_tile_size, target_tile_size),
                    getattr(Image, "Resampling", Image).LANCZOS,  # type: ignore[call-arg]
                )
            tile = np.array(tile, dtype=np.uint8)
            item[0] = tile
        item.pop()
        item = tuple(item)
        store.save_elem_to_disk(item)
        processed_count.value += 1

class PatchExtractor:
    """Class for preparing data from WSI"""

    def __init__(self, config: PatchExtractorConfig) -> None:
        self.config = config

        self.files: list[Path] = []
        self.annotation_files: list[Path] = []
        self.num_files: int = 0
        self.rescaling_factor: float = 1.0

        # TODO: Make this configurable from config
        self.downsample: int = 1
        self.min_intersection_ratio: float = 0.01
        self.store_masks: bool = False
        self.context_scales: list[int] | None = None
        self.masked_otsu: bool = False
        self.label_map: dict[str | int, int] = {"background": 0}
        self.otsu_annotation = None
        self.tissue_annotation_intersection_ratio = 0.01
        self.apply_prefilter = False
        self.normalize_stains: bool = False
        self.processes = 24
        self.save_only_annotated_patches: bool = False
        self.adjust_brightness: bool = False
        self.save_context: bool = True if self.context_scales is not None else False
        self.overlapping_labels: bool = False
        # TODO: ---

        # paths
        self.setup_output_path(self.config.output_path)
        self._set_wsi_path(self.config.wsi_path)

        # hardware
        self._set_hardware()

        # convert overlap from percentage to pixels
        self.config.patch_overlap = int(
            np.floor(self.config.patch_size / 2 * self.config.patch_overlap / 100)
        )

        # set seed
        random.seed(42)

        logger.info(f"Data store directory: {str(self.config.output_path)}")
        logger.info(f"Images found: {self.num_files}")
        logger.info(f"Annotations found: {len(self.annotation_files)}")

    def _set_hardware(self, hardware_selection: str = "cucim") -> None:
        """Either load CuCIM (GPU-accelerated) or OpenSlide


        Args:
            hardware_selection (str, optional): Specify hardware. Just for experiments. Must be either "openslide", or "cucim".
                Defaults to cucim.
        """
        if (
            module_exists("cucim", error="ignore")
            and hardware_selection.lower() == "cucim"
        ):
            logger.info("Using CuCIM")
            from cucim import CuImage  # type: ignore[import]

            from cellmil.data.cucim_deepzoom import DeepZoomGeneratorCucim

            self.deepzoomgenerator: (
                Type[DeepZoomGeneratorCucim] | Type[DeepZoomGeneratorOS]
            ) = DeepZoomGeneratorCucim
            self.image_loader: Any = CuImage
        else:
            logger.info("Using OpenSlide")
            self.deepzoomgenerator = DeepZoomGeneratorOS
            self.image_loader = OpenSlide

    def _set_wsi_path(self, wsi_path: str | Path) -> None:
        """Set the path to the WSI file.

        Args:
            wsi_paths (Union[str, Path, List]): Path to the folder where all WSI are stored or path to a single WSI-file.
        """
        if isinstance(wsi_path, str):
            wsi_path = Path(wsi_path)

        wsi_extension = wsi_path.suffix.lower()[1:]
        self.files = get_files_from_dir(wsi_path, wsi_extension)
        self.num_files = len(self.files)

        def key(x: Path) -> str:
            return x.name

        self.files = natsorted(self.files, key=key)

    def get_patches(self) -> None:
        """Main functiuon to create a dataset. Sample the complete dataset.

        This function acts as an entrypoint to the patch-processing.
        When this function is called, all WSI that have been detected are processed.
        Depending on the selected configuration, either already processed WSI will be removed or newly processed.
        The processed WSI are stored in the file `processed.json` in the output-folder.
        """
        # perform logical check
        self._check_patch_params(
            patch_size=self.config.patch_size,
            patch_overlap=int(self.config.patch_overlap),
            downsample=self.downsample,
            min_background_ratio=self.min_intersection_ratio,
        )
        # remove database or check to continue from checkpoint
        self._check_overwrite()

        total_count = 0
        start_time = start_timer()
        for i, wsi_file in enumerate(self.files):
            try:
                logger.info(f"{(os.get_terminal_size()[0] - 33) * '*'}")
            except Exception:
                pass
            logger.info(f"{i + 1}/{len(self.files)}: {wsi_file.name}")

            # prepare wsi, espeically find patches
            (
                _,
                (wsi_metadata, mask_images, mask_images_annotations, thumbnails),
                (
                    interesting_coords_wsi,
                    level_wsi,
                    polygons_downsampled_wsi,
                    region_labels_wsi,
                ),
            ) = self._prepare_wsi(wsi_file)

            # setup storage
            store = Storage(
                wsi_name=wsi_file.stem,
                output_path=self.config.output_path,
                metadata=wsi_metadata,
                mask_images=mask_images,
                mask_images_annotations=mask_images_annotations,
                thumbnails=thumbnails,
                store_masks=self.store_masks,
                save_context=self.context_scales is not None,
                context_scales=self.context_scales,
            )
            logger.info("Start extracting patches...")

            patch_count, patch_distribution, patch_result_metadata = self.process_queue(
                batch=interesting_coords_wsi,
                wsi_file=wsi_file,
                wsi_metadata=wsi_metadata,
                level=level_wsi,
                polygons=polygons_downsampled_wsi,
                region_labels=region_labels_wsi,
                store=store,
            )

            if patch_count == 0:
                logger.warning(f"No patches sampled from {wsi_file.name}")
            logger.info(f"Total patches sampled: {patch_count}")
            store.clean_up(patch_distribution, patch_result_metadata)

            total_count += patch_count

        logger.info(f"Patches saved to: {self.config.output_path.resolve()}")
        logger.info(f"Total patches sampled for all WSI: {total_count}")

        end_timer(start_time)

    def _prepare_wsi(
        self, wsi_file: Path
    ) -> tuple[
        tuple[int, int],
        tuple[
            dict[str, Any],
            dict[str, ImageType],
            dict[str, ImageType],
            dict[str, ImageType],
        ],
        tuple[list[Any], Any, list[Polygon], list[str]],
    ]:
        """Prepare a WSI for preprocessing

        First, some sanity checks are performed and the target level for DeepZoomGenerator is calculated.
        We are not using OpenSlides default DeepZoomGenerator, but rather one based on the cupy library which is much faster
        (cf https://github.com/rapidsai/cucim). One core element is to find all patches that are non-background patches.
        For this, a tissue mask is generated. At this stage, no patches are extracted!

        For further documentation (i.e., configuration settings), see the class documentation [link].

        Args:
            wsi_file (str): Name of the wsi file

        Raises:
            WrongParameterException: The level resulting from target magnification or downsampling factor must exists to extract patches.

        Returns:
            Tuple[Tuple[int, int], Tuple[dict, dict, dict, dict], Callable, List[List[Tuple]]]:

            - Tuple[int, int]: Number of rows, cols of the WSI at the given level
            - dict: Dictionary with Metadata of the WSI
            - dict[str, Image]: Masks generated during tissue detection stored in dict with keys equals the mask name and values equals the PIL image
            - dict[str, Image]: Annotation masks for provided annotations for the complete WSI. Masks are equal to the tissue masks sizes.
                Keys are the mask names and values are the PIL images.
            - dict[str, Image]: Thumbnail images with different downsampling and resolutions.
                Keys are the thumbnail names and values are the PIL images.
            - callable: Batch-Processing function performing the actual patch-extraction task
            - List[List[Tuple]]: Divided List with batches of batch-size. Each batch-element contains the row, col position of a patch together with bg-ratio.
        """
        logger.info(f"Computing patches for {wsi_file.name}")

        # load slide (OS and CuImage/OS)
        slide = OpenSlide(str(wsi_file))
        slide_cu = self.image_loader(str(wsi_file))

        mpp_value = slide.properties.get("openslide.mpp-x")
        if mpp_value is not None:
            slide_mpp = float(mpp_value)
        else:
            raise NotImplementedError("MPP must be in metadata of the WSI file!")

        mag_value = slide.properties.get("openslide.objective-power")
        if mag_value is not None:
            slide_mag = float(mag_value)
        else:
            raise NotImplementedError("MPP must be in metadata of the WSI file!")

        slide_properties = {"mpp": slide_mpp, "magnification": slide_mag}
        # Generate thumbnails
        logger.info("Generate thumbnails")
        thumbnails = generate_thumbnails(
            slide, slide_properties["mpp"], sample_factors=[128]
        )

        # target mag has precedence before downsample!
        self.downsample = target_mag_to_downsample(
            slide_properties["magnification"],
            self.config.target_mag,
        )

        # Zoom Recap:
        # - Row and column of the tile within the Deep Zoom level (t_)
        # - Pixel coordinates within the Deep Zoom level (z_)
        # - Pixel coordinates within the slide level (l_)
        # - Pixel coordinates within slide level 0 (l0_)
        # Tile size is the amount of pixels that are taken from the image (without overlaps)
        tile_size, overlap = patch_to_tile_size(
            self.config.patch_size,
            int(self.config.patch_overlap),
            self.rescaling_factor,
        )

        tiles = self.deepzoomgenerator(
            osr=slide,
            cucim_slide=slide_cu,
            tile_size=tile_size,
            overlap=overlap,
            limit_bounds=True,
        )

        # Each level is downsampled by a factor of 2
        # downsample expresses the desired downsampling, we need to count how many times the
        # downsampling is performed to find the level
        # e.g. downsampling of 8 means 2 * 2 * 2 = 3 times
        # we always need to remove 1 level more than necessary, so 4
        # so we can just use the bit length of the numbers, since 8 = 1000 and len(1000) = 4
        level = tiles.level_count - self.downsample.bit_length()

        if level >= tiles.level_count:
            raise WrongParameterException(
                "Requested level does not exist. Number of slide levels:",
                tiles.level_count,
            )

        # store level!
        self.curr_wsi_level = level

        # initialize annotation objects
        region_labels: List[str] = []
        polygons: List[Polygon] = []
        polygons_downsampled: List[Polygon] = []
        tissue_region: List[Polygon] = []

        # get the interesting coordinates: no background, filtered by annotation etc.
        # original number of tiles of total wsi
        n_cols, n_rows = tiles.level_tiles[level]

        (
            interesting_coords,
            mask_images,
            mask_images_annotations,
        ) = compute_interesting_patches(
            polygons=polygons,
            slide=slide,
            tiles=tiles,
            target_level=level,
            target_patch_size=tile_size,  # self.config.patch_size,
            target_overlap=overlap,  # self.config.patch_overlap,
            rescaling_factor=self.rescaling_factor,
            mask_otsu=self.masked_otsu,
            label_map=self.label_map,
            region_labels=region_labels,
            tissue_annotation=tissue_region,
            otsu_annotation=self.otsu_annotation,
            tissue_annotation_intersection_ratio=self.tissue_annotation_intersection_ratio,
            apply_prefilter=self.apply_prefilter,
        )

        if len(interesting_coords) == 0:
            logger.warning(f"No patches sampled from {wsi_file.name}")
        wsi_metadata: dict[str, Any] = {
            "orig_n_tiles_cols": n_cols,
            "orig_n_tiles_rows": n_rows,
            "base_magnification": slide_mag,
            "downsampling": self.downsample,
            "label_map": self.label_map,
            "patch_overlap": self.config.patch_overlap * 2,
            "patch_size": self.config.patch_size,
            "base_mpp": slide_mpp,
            "target_patch_mpp": slide_mpp * self.rescaling_factor,
            "stain_normalization": self.normalize_stains,
            "magnification": slide_mag / (self.downsample * self.rescaling_factor),
            "level": level,
        }

        logger.info(f"{wsi_file.name}: Processing {len(interesting_coords)} patches.")
        return (
            (n_cols, n_rows),
            (wsi_metadata, mask_images, mask_images_annotations, thumbnails),
            (list(interesting_coords), level, polygons_downsampled, region_labels),
        )

    def process_queue(
        self,
        batch: List[tuple[int, int, float]],
        wsi_file: Path,
        wsi_metadata: dict[str, Any],
        level: int,
        polygons: List[Polygon],
        region_labels: List[str],
        store: Storage,
    ) -> tuple[int, dict[int, int], list[dict[str, dict[str, Any]]]]:
        """Extract patches for a list of coordinates by using multiprocessing queues

        Patches are extracted according to their coordinate with given patch-settings (size, overlap).
        Patch annotation masks can be stored, as well as context patches with the same shape retrieved.
        Optionally, stains can be nornalized according to macenko normalization.

        Args:
            batch (List[Tuple[int, int, float]]): A batch of patch coordinates (row, col, backgropund ratio)
            wsi_file (Union[Path, str]): Path to the WSI file from which the patches should be extracted from
            wsi_metadata (dict): Dictionary with important WSI metadata
            level (int): The tile level for sampling.
            polygons (List[Polygon]): Annotations of this WSI as a list of polygons (referenced to highest level of WSI).
                If no annotations, pass an empty list [].
            region_labels (List[str]): List of labels for the annotations provided as polygons parameter.
            If no annotations, pass an empty list [].
            store (Storage): Storage object passed to each worker to store the files

        Returns:
            int: Number of processed patches
        """
        logger.debug(f"Started process {multiprocessing.current_process().name}")

        # reload image
        slide = OpenSlide(str(wsi_file))
        slide_cu = self.image_loader(str(wsi_file))

        tile_size, overlap = patch_to_tile_size(
            self.config.patch_size,
            int(self.config.patch_overlap),
            self.rescaling_factor,
        )

        tiles = self.deepzoomgenerator(
            osr=slide,
            cucim_slide=slide_cu,
            tile_size=tile_size,
            overlap=overlap,
            limit_bounds=True,
        )

        # queue setup
        queue: Any = multiprocessing.Queue() # type: ignore[assignment]
        processes: list[Process] = []
        processed_count = multiprocessing.Value("i", 0)

        pbar = tqdm(total=len(batch), desc="Retrieving patches")

        for _ in range(self.processes):
            p = multiprocessing.Process(
                target=queue_worker, args=(queue, store, processed_count)
            )
            p.start()
            processes.append(p)

        patches_count = 0
        patch_result_list: list[dict[str, dict[str, Any]]] = []
        patch_distribution = self.label_map
        patch_distribution = {v: 0 for _, v in patch_distribution.items()}

        start_time = start_timer()
        for row, col, _ in batch:
            pbar.update()
            # set name
            patch_fname = f"{wsi_file.stem}_{row}_{col}.png"
            patch_yaml_name = f"{wsi_file.stem}_{row}_{col}.yaml"

            # OpenSlide: Address of the tile within the level as a (column, row) tuple
            new_tile = np.array(tiles.get_tile(level, (col, row)), dtype=np.uint8)
            patch = pad_tile(new_tile, tile_size + 2 * overlap, col, row)

            # calculate background ratio for every patch
            background_ratio = calculate_background_ratio(
                new_tile, self.config.patch_size
            )

            # patch_label
            if background_ratio > 1 - self.min_intersection_ratio:
                logger.debug(
                    f"Removing file {patch_fname} because of intersection ratio with background is too big"
                )
                intersected_labels: List[int] = []  # Zero means background
                ratio: dict[int, float] = {}
            else:
                intersected_labels, _ratio, _ = get_intersected_labels(
                    tile_size=tile_size,
                    patch_overlap=int(self.config.patch_overlap),
                    col=col,
                    row=row,
                    polygons=polygons,
                    label_map=self.label_map,
                    min_intersection_ratio=self.min_intersection_ratio,
                    region_labels=region_labels,
                    overlapping_labels=self.overlapping_labels,
                    store_masks=self.store_masks,
                )
                ratio = {k: v for k, v in zip(intersected_labels, _ratio)}
            if len(intersected_labels) == 0 and self.save_only_annotated_patches:
                continue

            patch_metadata: dict[str, Any] = {
                "row": row,
                "col": col,
                "background_ratio": float(background_ratio),
                "intersected_labels": intersected_labels,
                "label_ratio": ratio,
                "wsi_metadata": wsi_metadata,
            }

            # increase patch_distribution count
            for patch_label in patch_metadata["intersected_labels"]:
                patch_distribution[patch_label] += 1

            patches_count = patches_count + 1

            queue_elem: tuple[np.ndarray[Any, Any], dict[str, Any], None, dict[str, Any], int] = (
                patch,
                patch_metadata,
                None,
                {},
                self.config.patch_size,
            )
            queue.put(queue_elem)
            # store metadata for all patches
            patch_metadata.pop("wsi_metadata")
            patch_metadata["metadata_path"] = f"./metadata/{patch_yaml_name}"

            patch_result_list.append({patch_fname: patch_metadata})

        # Add termination markers to the queue
        for _ in range(self.processes):
            queue.put(None)

        pbar.close()
        # wait for the queue to end
        while not queue.empty():
            print(f"Progress: {processed_count.value}/{len(batch)}", end="\r")
            print("", end="", flush=True)

        # Wait for all workers to finish
        for p in processes:
            p.join()
            p.close()
        pbar.close()

        logger.info("Finished Processing and Storing. Took:")
        end_timer(start_time)
        return patches_count, patch_distribution, patch_result_list

    def _drop_processed_files(self, processed_files: list[str]) -> None:
        """Drop processed file from `processed.json` file from dataset.

        Args:
            processed_files (list[str]): List with processed filenames
        """
        self.files = [file for file in self.files if file.stem not in processed_files]

    def _check_overwrite(self, overwrite: bool = False) -> None:
        """Performs data cleanage, depending on overwrite.

        If true, overwrites the patches that have already been created in
        case they already exist. If false, skips already processed files from `processed.json`
        in the provided output path (created during class initialization)

        Args:
            overwrite (bool, optional): Overwrite flag. Defaults to False.
        """
        if overwrite:
            logger.info("Removing complete dataset! This may take a while.")
            subdirs = [f for f in Path(self.config.output_path).iterdir() if f.is_dir()]
            for subdir in subdirs:
                rmtree(subdir.resolve(), ignore_errors=True)
            if (Path(self.config.output_path) / "processed.json").exists():
                os.remove(Path(self.config.output_path) / "processed.json")
            self.setup_output_path(self.config.output_path)
        else:
            try:
                with open(
                    str(Path(self.config.output_path) / "processed.json"), "r"
                ) as processed_list:
                    processed_files = json.load(processed_list)[
                        "processed_files"
                    ]  # TODO: check
                    logger.info(
                        f"Found {len(processed_files)} files. Continue to process {len(self.files) - len(processed_files)}/{len(self.files)} files."
                    )
                    self._drop_processed_files(processed_files)
            except FileNotFoundError:
                logger.info("Empty output folder. Processing all files")

    @staticmethod
    def _check_patch_params(
        patch_size: int,
        patch_overlap: int,
        downsample: int | None = None,
        target_mag: float | None = None,
        level: int | None = None,
        min_background_ratio: float = 0.01,
    ) -> None:
        """Sanity Check for parameters

        See `Raises`section for further comments about the sanity check.

        Args:
            patch_size (int): The size of the patches in pixel that will be retrieved from the WSI, e.g. 256 for 256px
            patch_overlap (int): The amount pixels that should overlap between two different patches.
            downsample (int, optional): Downsampling factor from the highest level (largest resolution). Defaults to None.
            target_mag (float, optional): If this parameter is provided, the output level of the wsi
            corresponds to the level that is at the target magnification of the wsi.
            Alternative to downsaple and level. Defaults to None.
            level (int, optional): The tile level for sampling, alternative to downsample. Defaults to None.
            min_background_ratio (float, optional): Minimum background selection ratio. Defaults to 1.0.

        Raises:
            WrongParameterException: Either downsample, level, or target_magnification must have been selected.
            WrongParameterException: Downsampling must be a power of two.
            WrongParameterException: Negative overlap is not allowed.
            WrongParameterException: Overlap should not be larger than half of the patch size.
            WrongParameterException: Background Percentage must be between 0 and 1.
        """
        if downsample is None and level is None and target_mag is None:
            raise WrongParameterException(
                "Both downsample and level are none, "
                "please fill one of the two parameters."
            )
        if downsample is not None and not is_power_of_two(downsample):
            raise WrongParameterException("Downsample can only be a power of two.")

        if downsample is not None and level is not None:
            logger.warning(
                "Both downsample and level are set, "
                "we will use downsample and ignore level."
            )

        if patch_overlap < 0:
            raise WrongParameterException("Negative overlap not allowed.")
        if patch_overlap > patch_size // 2:
            raise WrongParameterException(
                "An overlap greater than half the patch size yields a tile size of zero."
            )

        if min_background_ratio < 0.0 or min_background_ratio > 1.0:
            raise WrongParameterException(
                "The parameter min_background_ratio should be a "
                "float value between 0 and 1 representing the "
                "maximum percentage of background allowed."
            )

    @staticmethod
    def setup_output_path(output_path: Union[str, Path]) -> None:
        """Create output path

        Args:
            output_path (Union[str, Path]): Output path for WSI
        """
        output_path = Path(output_path)
        output_path.mkdir(exist_ok=True, parents=True)

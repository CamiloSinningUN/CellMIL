# -*- coding: utf-8 -*-
# Masking function to generate tissue masks
#
# References:
# CellViT: Vision Transformers for precise cell segmentation and classification
# Fabian Hörst et al., Medical Image Analysis, 2024
# DOI: https://doi.org/10.1016/j.media.2024.103143

import os
import warnings
from typing import List, Tuple, Union, Any

import cv2
import numpy as np
import rasterio # type: ignore
import skimage.color as sk_color
import skimage.filters as sk_filters
import skimage.morphology as sk_morphology
from PIL import Image
from rasterio.mask import mask as rasterio_mask # type: ignore
from shapely.affinity import scale
from shapely.geometry import Polygon

def generate_tissue_mask(
    tissue_tile: np.ndarray[Any, Any],
    mask_otsu: bool = False,
    polygons: List[Polygon] | None = None,
    region_labels: List[str] | None = None,
    otsu_annotation: Union[List[str], str] | None = "object",
    downsample: int = 1,
    apply_prefilter: bool = False,
) -> np.ndarray[Any, Any]:
    """Generate a tissue mask using otsu thresholding.

    Per Default, otsu-thresholding is performed. If mask_otsu is true, first a masked image is calculate
    using the annotation matching the otsu_annotation label.

    Args:
        tissue_tile (np.ndarray): Tissue tile as numpy array with shape (height, width, 3)
        mask_otsu (bool, optional): If masking is applied before thresholding. Defaults to False.
        polygons (List[Polygon], optional):  Annotations of this WSI as a list of polygons (referenced to highest level of WSI). Defaults to None.
        region_labels (List[str], optional): List of labels for the annotations provided as polygons parameter. Defaults to None.
        otsu_annotation (Union[List[str], str], optional):  List with annotation names or string with annotation name to use for a masked otsu thresholding.
            Defaults to "object".
        downsample (int, optional): Downsampling of the tissue tile compared to highest WSI level. Used for matching annotations with tissue-tile size.
            Defaults to 1.
        apply_prefilter (bool, optional): If a prefilter should be used to remove markers before applying otsu. Defaults to False.

    Returns:
        np.ndarray: Binary tissue mask with shape (height, width)
    """
    if polygons is not None and region_labels is not None:
        assert len(polygons) == len(
            region_labels
        ), "Polygon list and polygon labels are not having the same length"


    tissue_mask = apply_otsu_thresholding(tile=tissue_tile)
    assert len(np.unique(tissue_mask)) <= 2, "Mask is not binary"

    return tissue_mask


def convert_polygons_to_mask(
    polygons: Union[List[Polygon], Polygon],
    reference_size: tuple[int, int, int],
    downsample: int = 1,
) -> np.ndarray[Any, Any]:
    """Convert a polygon to a mask

    The function is assuming that polygons have already been filtered (see get_filtered_polygon).

    Args:
        polygons (Tuple[List[Polygon], Polygon]): List of polygons converted to a mask. Can work with Polygons with holes inside
        reference_size (tuple[int]): Shape of resulting mask image. Shape should be (height, width, channels).
        downsample (int, optional): Set the factor by which the polygon should be scaled down. Defaults to 1.

    Returns:
        np.ndarray: Binary mask with shape (height, width)
    """

    if type(polygons) is not List:
        polygons = list(polygons) # type: ignore
    polygons_downsampled = [
        scale(
            poly,
            xfact=1 / downsample,
            yfact=1 / downsample,
            origin=(0, 0),
        )
        for poly in polygons
    ]
    src = 255 * np.ones(shape=reference_size, dtype=np.uint8)
    im = Image.fromarray(src)
    im.save("tmp.tif")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        with rasterio.open("tmp.tif") as src: # type: ignore
            out_image: Tuple[np.ndarray[Any, Any], Any] = rasterio_mask(src, polygons_downsampled, crop=False) # type: ignore
            mask = out_image[0].transpose(1, 2, 0) 
            mask = np.invert(mask) 
    os.remove("tmp.tif")
    mask = (mask / 255).astype(np.uint8)

    assert len(np.unique(mask)) <= 2, "Mask is not binary"

    return mask[:, :, 0]

def apply_otsu_thresholding(tile: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Generate a binary tissue mask by using Otsu thresholding

    Args:
        tile (np.ndarray): Tile with tissue with shape (height, width, 3)

    Returns:
        np.ndarray: Binary mask with shape (height, width)
    """
    hsv_img = cv2.cvtColor(tile.astype(np.uint8), cv2.COLOR_RGB2HSV)
    gray_mask = cv2.inRange(hsv_img, np.array((0, 0, 70)), np.array((180, 10, 255)))
    black_mask = cv2.inRange(hsv_img, np.array((0, 0, 0)), np.array((180, 255, 85)))
    # Set all grey/black pixels to white
    full_tile_bg = np.copy(tile)
    combined_mask = cv2.bitwise_or(gray_mask, black_mask)
    full_tile_bg[np.where(combined_mask)] = 255

    # apply otsu mask first time for removing larger artifacts
    masked_image_gray = 255 * sk_color.rgb2gray(full_tile_bg) # type: ignore
    thresh = sk_filters.threshold_otsu(masked_image_gray) # type: ignore
    otsu_masking = masked_image_gray < thresh # type: ignore
    # improving mask
    otsu_masking = sk_morphology.remove_small_objects(otsu_masking, 60) # type: ignore
    otsu_masking = sk_morphology.dilation(otsu_masking, sk_morphology.square(12)) # type: ignore
    otsu_masking = sk_morphology.closing(otsu_masking, sk_morphology.square(5)) # type: ignore
    otsu_masking = sk_morphology.remove_small_holes(otsu_masking, 250) # type: ignore
    tile = mask_rgb(tile, otsu_masking).astype(np.uint8) # type: ignore

    # apply otsu mask second time for removing small artifacts
    masked_image_gray = 255 * sk_color.rgb2gray(tile) # type: ignore
    thresh = sk_filters.threshold_otsu(masked_image_gray) # type: ignore
    otsu_masking = masked_image_gray < thresh # type: ignore
    otsu_masking = sk_morphology.remove_small_holes(otsu_masking, 5000) # type: ignore
    otsu_thr: np.ndarray = ~otsu_masking # type: ignore
    otsu_thr = otsu_thr.astype(np.uint8) # type: ignore

    return otsu_thr # type: ignore

def mask_rgb(rgb: np.ndarray[Any, Any], mask: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Mask an RGB image

    Args:
        rgb (np.ndarray): RGB image to mask with shape (height, width, 3)
        mask (np.ndarray): Binary mask with shape (height, width)

    Returns:
        np.ndarray: Masked image
    """
    assert (
        rgb.shape[:-1] == mask.shape
    ), "Mask and RGB shape are different. Cannot mask when source and mask have different dimension."
    mask_positive = np.dstack([mask, mask, mask])
    mask_negative = np.dstack([~mask, ~mask, ~mask])
    positive = rgb * mask_positive
    negative = rgb * mask_negative
    negative = 255 * (negative > 0.0001).astype(int)

    masked_image = positive + negative

    return np.clip(masked_image, a_min=0, a_max=255)

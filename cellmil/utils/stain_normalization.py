# -*- coding: utf-8 -*-
# Patch Extractor Module
# 
# References:
# CellViT: Vision Transformers for precise cell segmentation and classification
# Fabian Hörst et al., Medical Image Analysis, 2024
# DOI: https://doi.org/10.1016/j.media.2024.103143

import numpy as np
from typing import List, Tuple, Union, Any, cast
from cellmil.utils import logger


class NormalizeParameters:
    """Macenko Stain Normalization.

    Contains vectors used as reference for the Macenko normalization.
    The H&E reference matrix should have shape (3, 2) where the rows are RGB positions and the
    columns are the two components (first column Haematoxylin, second column Eosin).
    The maximum saturation (stain concentration) should have shape (2, 1) and it represents the
    maximum saturation/concentration that the Haematoxylin and Eosin will have in optical density
    before the conversion to RGB.
    """

    def __init__(self) -> None:
        self._MAX_SAT = np.array([[1.12129866], [0.86893238]])
        self._HE_REF = np.array(
            [
                [0.52656958, 0.43845532],
                [0.79612922, 0.64025238],
                [0.29816563, 0.6307407],
            ]
        )

    def get_he_ref(self) -> np.ndarray[Any, Any]:
        """Returns the reference H&E vector.


        Returns:
            np.ndarray: A (3, 2) ndarray representing some reference values for a H&E WSI
        """
        return self._HE_REF

    def get_max_sat(self) -> np.ndarray[Any, Any]:
        """Returns the maximum saturation.


        Returns:
            np.ndarray: A (2, 1) ndarray representing the desired maximum saturation.
        """
        return self._MAX_SAT

def macenko_normalization(
    patches: List[np.ndarray[Any, Any]],
    beta: float = 0.15,
    alpha: int = 1,
    light_intensity: int = 255,
) -> Tuple[List[np.ndarray[Any, Any]], Union[None, np.ndarray[Any, Any]], Union[None, np.ndarray[Any, Any]]]:
    """Normalizes the stain appareance of the patches using Macenko.

    References:
    ---------
    - A method for normalizing histology slides for quantitative analysis. M. Macenko et al.,
    ISBI 2009
    - https://github.com/schaugf/HEnorm_python/blob/master/normalizeStaining.py
    - https://github.com/CODAIT/deep-histopath/blob/c8baf8d47b6c08c0f6c7b1fb6d5dd6b77e711c33/
    deephistopath/preprocessing.py
    - https://towardsdatascience.com/stain-estimation-on-microscopy-whole-slide-images-2b5a57062268

    Args:
        patches (List[np.ndarray]): A list of patches
        beta (float, optional):  Which portion of the transparent values should be removed. Defaults to 0.15.
        alpha (int, optional): Which percentiles of the values to consider (alpha, 1 - alpha). Defaults to 1.
        light_intensity (int, optional):  The highest luminosity value. Defaults to 255.
        normalization_vector_path (Union[Path, str], optional): Path to a normalization file (needs to be a json file). Defaults to None.

    Returns:
        Tuple[List[np.ndarray], Union[None, np.ndarray], Union[None, np.ndarray]]: A stained-normalized list of patches,
            the mixing stain vectors and the 99th percentile saturation vectors
    """
    # Load normalization parameters
    normalization_vector_patch = NormalizeParameters()

    # Get all the patches from the dictionary
    h, w, c = patches[0].shape
    # Stack all the patches on a new dimension and reshape it such that it is just a vector of RGB
    # intensities
    stacked_patches = np.stack(patches, axis=0).reshape(-1, 3)
    # Histo images: The more stain there is, the darker the color
    # According to Beer-Lambert law the light will be attenuated exponentially with the density of the tissue

    # Convert to OD (optical density)
    OD = RGB_to_OD(stacked_patches, light_intensity)

    # Remove transparent pixels up to the value beta
    non_trasparent = (OD > beta).any(axis=1)

    # Compute eigenvectors
    try:
        _, eig_vecs = np.linalg.eigh(np.cov(OD[non_trasparent], rowvar=False))
    except np.linalg.LinAlgError:
        logger.warning(
            "numpy.linalg.LinAlgError: Eigenvalues did not converge. The normalization "
            "could not be performed, try using a higher level with more patches."
        )
        return patches, None, None

    # Take the two largest eigenvectors (strip off the residual stain component)
    eig_vecs = eig_vecs[:, 1:3]
    # We assume they need to be positive
    if eig_vecs[0, 0] < 0:
        eig_vecs[:, 0] *= -1
    if eig_vecs[0, 1] < 0:
        eig_vecs[:, 1] *= -1
    # Project on the plane spanned by the eigenvectors corresponding to the two
    # largest eigenvalues
    # N x 3 * 3 x 2 = N x 2
    T_hat = OD[non_trasparent].dot(eig_vecs)
    # The stain vectors are represented by the maximum and minimum angle
    # We first compute the angle: phi = arctan2(y, x) -> angle between x axis and line passing through (y, x)
    phi = np.arctan2(T_hat[:, 1], T_hat[:, 0])
    # And estimate the maximum and minimum using percentiles
    min_phi = cast(float, np.percentile(phi, alpha))
    max_phi = cast(float, np.percentile(phi, 100 - alpha))

    # Trasform angles back to 3D coordinates
    v_min = eig_vecs.dot(np.array([(np.cos(min_phi), np.sin(min_phi))]).T)
    v_max = eig_vecs.dot(np.array([(np.cos(max_phi), np.sin(max_phi))]).T)

    # A heuristic to make the vector corresponding to Haematoxylin first and the
    # one corresponding to Eosin second (Haematoxylin is larger)
    if v_min[0] > v_max[0]:
        stain_vectors = np.array([v_min[:, 0], v_max[:, 0]]).T
    else:
        stain_vectors = np.array([v_max[:, 0], v_min[:, 0]]).T

    # Determine concentrations/saturations of the individual stains
    # Note: Here, we solve
    #    OD = VS
    #     S = V^{-1}OD
    # where 'OD' is the matrix of optical density values of our image,
    # 'V' is the matrix of stain vectors, and 'S' is the matrix of stain
    # saturations.  Since this is an overdetermined system, we use the
    # least squares solver, rather than a direct solve.
    # OD[non_trasparent].T: Rows correspond to channels (RGB), columns to OD values
    # 3 x (patch_size * patch_size)
    # We want to obtain a matrix with two values (H&E), but we have three
    # 2 x (patch_size * patch_size)
    sat = np.linalg.lstsq(stain_vectors, OD[non_trasparent].T, rcond=None)[0]

    # Find maximum saturation with percentiles
    # 2 values (H&E)
    max_sat = np.percentile(sat, 99, axis=1, keepdims=True)
    # Normalization: (sat / max_sat) * ref_sat
    # Divide by maximum and multiply by the new reference maximum
    sat = np.multiply(np.divide(sat, max_sat), normalization_vector_patch.get_max_sat())

    # Recreate the image using reference mixing matrix
    # Multiply our matrix by the reference matrix and convert back to RGB
    # Substitute the values that were not transparent with the new matrix
    OD[non_trasparent] = -normalization_vector_patch.get_he_ref().dot(sat).T
    i_norm = np.multiply(light_intensity, np.exp(OD)).reshape((len(patches), h, w, c))
    # Round and clip the values
    np.clip(np.round(i_norm), a_min=0, a_max=255, out=i_norm)
    # logger.debug(f'Restoring the original vector {i_norm.shape} took:'
    #              f' {timedelta(seconds=timer() - s)}.')
    return [p.astype(np.uint8) for p in i_norm], stain_vectors, max_sat


def RGB_to_OD(img: np.ndarray[Any, Any], light_intensity: int = 255) -> np.ndarray[Any, Any]:
    """Converts an RGB array to OD.

    Args:
        img (np.ndarray): An RGB array
        light_intensity (int, optional): The highest luminosity value. Defaults to 255.

    Returns:
        np.ndarray: The optical density array
    """
    img[img == 0] = 1  # Avoid 0 division
    return -np.log((img.astype(np.float64)) / light_intensity)
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Literal
from radiomics import featureextractor  # type: ignore
from skimage.measure import regionprops  # type: ignore
from skimage.morphology import convex_hull_image  # type: ignore
from cellmil.interfaces.FeatureExtractorConfig import ExtractorType
from scipy.ndimage import distance_transform_edt  # type: ignore
from skimage.color import rgb2hed, rgb2gray, rgb2hsv  # type: ignore

# TODO: Take into account the magnification factor when extracting features


class MorphologicalExtractor:
    """Class to handle feature extraction."""

    def __init__(self, extractor_name: ExtractorType) -> None:
        """Initialize the Extractor with a PyRadiomics feature extractor."""
        self.extractor_name = extractor_name

        # Ensure SimpleITK uses a single thread per process when multiprocessing
        try:
            sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)  # type: ignore
        except Exception:
            pass

        if self.extractor_name == "pyradiomics_gray":
            self.extractor = PyRadiomicsExtractor(enable_all_features=True, mode="gray")
        elif self.extractor_name == "pyradiomics_hed":
            self.extractor = PyRadiomicsExtractor(enable_all_features=True, mode="hed")
        elif self.extractor_name == "pyradiomics_hue":
            self.extractor = PyRadiomicsExtractor(enable_all_features=True, mode="hue")
        elif self.extractor_name == "morphometrics":
            self.extractor = MorphometricsExtractor()
        else:
            raise ValueError(
                f"Unsupported extractor type: {self.extractor}. Supported extractors are: pyradiomics_gray, pyradiomics_hed, pyradiomics_hue, morphometrics"
            )

    def extract_features(
        self, image: np.ndarray[Any, Any], mask: np.ndarray[Any, Any], debug: bool = False
    ) -> dict[str, Any]:
        """Extract features from the given image and mask."""
        if debug:
            fig, axs = plt.subplots(1, 3, figsize=(12, 4)) # type: ignore
            axs[0].imshow(image)
            axs[0].set_title(f"Image {image.shape}")
            axs[0].axis("off")

            axs[1].imshow(mask, cmap="gray")
            axs[1].set_title(f"Mask {mask.shape}")
            axs[1].axis("off")

            # Overlay: show image with mask as red transparent overlay
            overlay = image.copy()
            if overlay.ndim == 2 or overlay.shape[2] == 1:
                overlay = np.stack([overlay]*3, axis=-1)
            mask_rgb = np.zeros_like(overlay)
            mask_rgb[..., 0] = mask * 255  # Red channel
            alpha = 0.4
            overlay_img = overlay.copy()
            overlay_img = np.clip((1 - alpha) * overlay + alpha * mask_rgb, 0, 255).astype(np.uint8)
            axs[2].imshow(overlay_img)
            axs[2].set_title("Overlay")
            axs[2].axis("off")

            plt.tight_layout()
            plt.show() # type: ignore
            
            input("Press 'y' and Enter to continue...")
            
        try:
            features = self.extractor.extract_features(image, mask)
            return features
        except Exception as e:
            raise RuntimeError(
                f"Error during feature extraction with {self.extractor_name}: {e}"
            ) from e


class PyRadiomicsExtractor:
    """Class to handle PyRadiomics feature extraction."""

    def __init__(self, enable_all_features: bool = True, mode: Literal["gray", "hed", "hue"] = "gray") -> None:
        """Initialize the PyRadiomics feature extractor.

        Args:
            enable_all_features: Whether to enable all feature classes or just shape2D
            mode: 'gray' for grayscale, 'hed' for H channel from HED deconvolution
        """
        self.mode = mode
        # Create a new extractor instance
        self.extractor = featureextractor.RadiomicsFeatureExtractor()

        # Configure settings
        settings: dict[str, Any] = {
            # Discretization: choose ONE of these blocks

            # Option 1: fixed bin count (robust)
            "binCount": 64,          # <- use this
            # "binWidth": 25,        # <- REMOVE if using binCount

            # Option 2 (alternative): tighter bin width if you keep binWidth
            # "binWidth": 0.2,       # good with normalize=True (z-score range ~ -3..3)
            # (do not set binCount at the same time)

            # Small-ROI friendly textures
            "force2D": True,         # compute textures per-slice then aggregate
            "force2Ddimension": 0,   # axial; change if your data is sagittal/coronal
            "distances": [1],        # avoid larger neighborhoods

            # Resampling helps tiny masks form neighborhoods
            "resampledPixelSpacing": [0.5, 0.5],  # (use 3 values for 3D data)
            "interpolator": sitk.sitkBSpline,     # images (masks use NN internally) # type: ignore

            # Intensity handling
            "normalize": True,       # ok to keep; just don't use a huge binWidth
            "normalizeScale": 1,
            # If intensities can be ≤0 and you compute entropy-like features:
            # "voxelArrayShift": 1.0,

            "correctMask": True,
            "label": 1,
            "verbose": False,
        }

        # Apply settings
        for key, value in settings.items():
            self.extractor.settings[key] = value  # type: ignore

        # Enable image types and features based on parameter
        self.extractor.enableImageTypeByName("Original")  # type: ignore

        if enable_all_features:
            # self.extractor.enableImageTypeByName("Wavelet")  # type: ignore
            self.extractor.enableImageTypeByName("LoG")  # type: ignore
            self.extractor.enableFeatureClassByName("firstorder")  # type: ignore
            self.extractor.enableFeatureClassByName("shape2D")  # type: ignore
            self.extractor.enableFeatureClassByName("glcm")  # type: ignore
            self.extractor.enableFeatureClassByName("glrlm")  # type: ignore
            self.extractor.enableFeatureClassByName("glszm")  # type: ignore
            self.extractor.enableFeatureClassByName("gldm")  # type: ignore
            self.extractor.enableFeatureClassByName("ngtdm")  # type: ignore
        else:
            self.extractor.enableFeatureClassByName("shape2D")  # type: ignore

    def _preprocess(
        self, image: np.ndarray[Any, Any], mask: np.ndarray[Any, Any]
    ) -> tuple[sitk.Image, sitk.Image]:
        """Convert numpy arrays to SimpleITK images for PyRadiomics, with support for grayscale or HED-H channel."""
        if self.mode == "gray":
            image_proc = (rgb2gray(image) * 255).astype(np.uint8) # type: ignore
            image_proc = 255 - image_proc # type: ignore
        elif self.mode == "hue":
            hsv = rgb2hsv(image)  # type: ignore
            image_proc = (hsv[:, :, 0] * 255).astype(np.uint8)  # type: ignore
        elif self.mode == "hed":
            hed = rgb2hed(image)  # type: ignore
            image_proc = hed[:, :, 0]  # type: ignore
        else:
            raise ValueError(f"Unknown preprocessing mode: {self.mode}")

        # Convert to SimpleITK
        sitk_image = sitk.GetImageFromArray(image_proc.astype(np.float64))  # type: ignore
        sitk_mask = sitk.GetImageFromArray(mask.astype(np.int32))  # type: ignore

        return sitk_image, sitk_mask

    def extract_features(
        self, image: np.ndarray[Any, Any], mask: np.ndarray[Any, Any]
    ) -> dict[str, float]:
        """Extract features from the given image and mask."""
        sitk_image, sitk_mask = self._preprocess(image, mask)
        _features = self.extractor.execute(sitk_image, sitk_mask)  # type: ignore
        features: dict[str, float] = {
            k: v for k, v in _features.items() if not k.startswith("diagnostics_") # type: ignore
        }  
        return features  # type: ignore


class MorphometricsExtractor:
    """Placeholder class for morphometric feature extraction."""

    def extract_features(
        self,
        image: np.ndarray[Any, Any],
        mask: np.ndarray[Any, Any],
        magnification: float = 0.25,
    ) -> dict[str, float]:
        """Extract morphometrics features from the given image and mask."""
        # Get basic shape features from PyRadiomics
        props = regionprops(mask.astype(np.uint8), cache=False)  # type: ignore
        major_axis_length: float = props[0].major_axis_length  # type: ignore
        minor_axis_length: float = props[0].minor_axis_length  # type: ignore
        area = props[0].area  # type: ignore
        perimeter = props[0].perimeter  # type: ignore
        d_max = props[0].feret_diameter_max  # type: ignore

        ch_mask = convex_hull_image(mask.astype(np.uint8))  # type: ignore
        ch_props = regionprops(ch_mask.astype(np.uint8), cache=False)  # type: ignore
        ch_area = ch_props[0].area  # type: ignore
        ch_perimeter = ch_props[0].perimeter  # type: ignore

        dist = distance_transform_edt(mask.astype(np.int8))  # type: ignore
        d_ins = 2 * dist.max()  # type: ignore

        features: dict[str, float] = {
            "axial_ratio": (props[0].bbox[2] - props[0].bbox[0]) / (props[0].bbox[3] - props[0].bbox[1]),  # type:ignore
            "aspect_ratio": major_axis_length / minor_axis_length,  # AR = a/b where a and b are axes of best fitted ellipse
            "eccentricity": minor_axis_length / major_axis_length,
            "rectangular_factor": area / (major_axis_length * minor_axis_length),
            "elongation_index": np.log2(major_axis_length / minor_axis_length),  # type: ignore
            "dispersion_index": np.log2(np.pi * major_axis_length * minor_axis_length),  # type: ignore
            "circularity": 4 * np.pi * area / (perimeter**2),
            "roundness": perimeter / np.sqrt(4 * np.pi * area),  # type: ignore
            "roundness_factor": 4 * area / (np.pi * d_max**2),
            "convexity": ch_perimeter / perimeter,
            "spreading_index": (np.pi * ch_perimeter) / (4 * ch_area),
            "irregularity_index": d_max / d_ins,
            "solidity": area / ch_area,
        }

        return features

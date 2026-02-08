import torch
import timm
import torch.nn as nn
import torchvision.transforms as transforms  # type: ignore
from timm.data.config import resolve_data_config  # type: ignore
from timm.data.transforms_factory import create_transform
from timm.layers.mlp import SwiGLUPacked
from typing import cast
from torchvision.models import resnet50  # type: ignore
from cellmil.interfaces.FeatureExtractorConfig import ExtractorType
from cellmil.utils import logger
from typing import Any
from transformers import AutoModel

class EmbeddingExtractor:
    def __init__(self, extractor_name: ExtractorType):
        self.extractor_name = extractor_name

        if self.extractor_name == ExtractorType.resnet50:
            self.extractor = ResNet50Extractor()
        elif self.extractor_name == ExtractorType.gigapath:
            self.extractor = GigapathExtractor()
        elif self.extractor_name == ExtractorType.uni:
            self.extractor = UNIExtractor()
        elif self.extractor_name == ExtractorType.titan:
            self.extractor = TITANExtractor()
        else:
            raise ValueError(f"Unknown extractor type: {self.extractor_name}")

    def extract_features(self, batch: torch.Tensor) -> torch.Tensor:
        try:
            features = self.extractor.extract_features(batch)
            return features
        except Exception as e:
            raise RuntimeError(f"Failed to extract features from {batch}: {e}")


class ResNet50Extractor:
    """ResNet50 feature extractor with adaptive mean pooling after 3rd residual block."""

    def __init__(self):
        # Check if GPU is available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load pretrained ResNet50
        full_model = resnet50(weights="DEFAULT")

        # Extract layers up to the 3rd residual block
        self.features = nn.Sequential(
            full_model.conv1,  # 7x7 conv, 64 channels
            full_model.bn1,
            full_model.relu,
            full_model.maxpool,  # 3x3 max pool
            full_model.layer1,  # 1st residual block (64 -> 256 channels)
            full_model.layer2,  # 2nd residual block (256 -> 512 channels)
            full_model.layer3,  # 3rd residual block (512 -> 1024 channels)
        )

        # Add adaptive mean pooling to get 1024-dimensional features
        # After layer3, we have 1024 channels, so adaptive pooling to 1x1 gives us 1024 features
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Move model to GPU and set to eval mode
        self.features = self.features.to(self.device)
        self.adaptive_pool = self.adaptive_pool.to(self.device)
        self.features.eval()

        # Transform for 256x256 patches
        self.transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def extract_features(self, batch: torch.Tensor) -> torch.Tensor:
        try:
            # Normalize input to [0, 1] range if needed
            if batch.max() > 1.0:
                batch = batch.float() / 255.0

            # Apply transforms (resize and normalize)
            _batch = cast(torch.Tensor, self.transform(batch))

            with torch.no_grad():
                # Move input tensor to GPU
                _batch = _batch.to(self.device)

                # Extract features up to 3rd residual block
                features = self.features(_batch)

                # Apply adaptive mean pooling to get 1024-dimensional features
                pooled_features = self.adaptive_pool(features)

                # Flatten to get 1024-dimensional vector
                feature_vector = pooled_features.view(
                    pooled_features.size(0), -1
                )  # Ensure batch dimension is preserved

                # Move back to CPU for further processing
                feature_vector = feature_vector.cpu()

            return feature_vector

        except Exception:
            # Return zero vector with 1024 dimensions in case of error
            logger.error("Error in ResNet50 feature extraction, returning zero vector.")
            batch_size = batch.shape[0] if hasattr(batch, "shape") else 1
            return torch.zeros(batch_size, 1024)


class GigapathExtractor:
    """Gigapath feature extractor using a pretrained EfficientNet model from timm."""

    def __init__(self):
        # Check if GPU is available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load pretrained EfficientNet model from timm
        self.model = timm.create_model(
            "hf_hub:prov-gigapath/prov-gigapath", pretrained=True
        )

        # Move model to GPU and set to eval mode
        self.model = self.model.to(self.device)
        self.model.eval()

        # Transform for 224x224 patches
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    256, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.CenterCrop(224),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )

    def extract_features(self, batch: torch.Tensor) -> torch.Tensor:
        try:
            # Normalize input to [0, 1] range if needed
            if batch.max() > 1.0:
                batch = batch.float() / 255.0

            # Apply transforms (resize and normalize)
            _batch = cast(torch.Tensor, self.transform(batch))

            with torch.no_grad():
                # Move input tensor to GPU
                _batch = _batch.to(self.device)

                # Extract features using EfficientNet
                features = self.model(_batch)

                # Move back to CPU for further processing
                features = features.cpu()

            return features

        except Exception as e:
            logger.error(f"Error in Gigapath feature extraction: {e}")
            raise RuntimeError("Error in Gigapath feature extraction.")


class UNIExtractor:
    """UNI feature extractor using a pretrained model from timm.

    Note: Requires huggingface_hub login with access token before first use.
    Run: from huggingface_hub import login; login()
    """

    def __init__(self):
        # Check if GPU is available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Model configuration for UNI2-h
        timm_kwargs: dict[str, Any] = {
            "model_name": "vit_giant_patch14_224",
            "img_size": 224,
            "patch_size": 14,
            "depth": 24,
            "num_heads": 24,
            "init_values": 1e-5,
            "embed_dim": 1536,
            "mlp_ratio": 2.66667 * 2,
            "num_classes": 0,
            "no_embed_class": True,
            "mlp_layer": SwiGLUPacked,
            "act_layer": torch.nn.SiLU,
            "reg_tokens": 8,
            "dynamic_img_size": True,
        }

        # Load pretrained UNI2-h model from timm
        self.model = timm.create_model(
            "hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs
        )

        # Move model to GPU and set to eval mode
        self.model = self.model.to(self.device)
        self.model.eval()

        # Transform for 224x224 patches with ImageNet normalization
        self.transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )

    def extract_features(self, batch: torch.Tensor) -> torch.Tensor:
        try:
            # Normalize input to [0, 1] range if needed
            if batch.max() > 1.0:
                batch = batch.float() / 255.0

            # Apply transforms (resize and normalize)
            _batch = cast(torch.Tensor, self.transform(batch))

            with torch.no_grad():
                # Move input tensor to GPU
                _batch = _batch.to(self.device)

                # Extract features using UNI model
                features = self.model(_batch)

                # Move back to CPU for further processing
                features = features.cpu()

            return features

        except Exception as e:
            logger.error(f"Error in UNI feature extraction: {e}")
            raise RuntimeError("Error in UNI feature extraction.")
        
class TITANExtractor:
    """TITAN feature extractor using CONCH v1.5 for patch-level embeddings.
    
    Note: Requires huggingface_hub login with access token before first use.
    TITAN uses CONCH v1.5 for patch-level feature extraction at 512x512 pixels.
    """

    def __init__(self):

        # Check if GPU is available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load TITAN model to get CONCH v1.5
        logger.info("Loading TITAN model and CONCH v1.5 encoder...")
        try:
            titan = AutoModel.from_pretrained(
                'MahmoodLab/TITAN', 
                trust_remote_code=True
            )
            # Get CONCH v1.5 model and preprocessing transform
            self.model, self.transform = titan.return_conch()
        except Exception as e:
            raise RuntimeError(
                f"Failed to load TITAN/CONCH v1.5. Make sure you have access and are logged in to HuggingFace: {e}"
            )

        # Move model to GPU and set to eval mode
        self.model = self.model.to(self.device)
        self.model.eval()

        logger.info("TITAN/CONCH v1.5 model loaded successfully")

    def extract_features(self, batch: torch.Tensor) -> torch.Tensor:
        try:
            # Normalize input to [0, 1] range if needed
            if batch.max() > 1.0:
                batch = batch.float() / 255.0

            # Apply CONCH v1.5 preprocessing transform
            _batch = cast(torch.Tensor, self.transform(batch))

            with torch.no_grad():
                # Move input tensor to GPU
                _batch = _batch.to(self.device)

                # Extract patch-level features using CONCH v1.5
                # Use encode_image without projection for MIL tasks
                features = self.model.encode_image(_batch, proj_contrast=False, normalize=False)

                # Move back to CPU for further processing
                features = features.cpu()

            return features

        except Exception as e:
            logger.error(f"Error in TITAN/CONCH v1.5 feature extraction: {e}")
            raise RuntimeError("Error in TITAN/CONCH v1.5 feature extraction.")
        
class VirchowExtractor:
    """Virchow feature extractor using a custom model from timm."""

    def __init__(self):
        # Check if GPU is available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load custom model from timm
        self.model = timm.create_model(
            "hf-hub:paige-ai/Virchow2",
            pretrained=True,
            mlp_layer=SwiGLUPacked,
            act_layer=torch.nn.SiLU,
        )

        # Move model to GPU and set to eval mode
        self.model = self.model.to(self.device)
        self.model.eval()

        self.transform = cast(
            type[transforms.Compose],
            create_transform(
                **resolve_data_config(  # type: ignore
                    self.model.pretrained_cfg, model=self.model
                )
            ),
        )

    def extract_features(self, batch: torch.Tensor) -> torch.Tensor:
        try:
            # Normalize input to [0, 1] range if needed
            if batch.max() > 1.0:
                batch = batch.float() / 255.0

            # Apply transforms (resize and normalize)
            _batch = cast(torch.Tensor, self.transform(batch))

            with torch.no_grad():
                # Move input tensor to GPU
                _batch = _batch.to(self.device)

                # Extract features using the custom model
                output = self.model(_batch)

                class_token = output[:, 0]  # size: 1 x 1280
                patch_tokens = output[
                    :, 5:
                ]  # size: 1 x 256 x 1280, tokens 1-4 are register tokens so we ignore those

                # concatenate class token and average pool of patch tokens
                embedding = torch.cat(
                    [class_token, patch_tokens.mean(1)], dim=-1
                )  # size: 1 x 2560

                features = embedding.cpu()
            return features

        except Exception as e:
            logger.error(f"Error in Virchow feature extraction: {e}")
            raise RuntimeError("Error in Virchow feature extraction.")

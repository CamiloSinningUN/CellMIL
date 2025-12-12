import torch
import torch.nn as nn
import numpy as np
import os
import logging
from typing import Dict, Any, Optional, Union, Tuple, cast
from cellmil.utils import logger

# Suppress cellpose logs
logging.getLogger('cellpose.dynamics').setLevel(logging.WARNING)
logging.getLogger('cellpose').setLevel(logging.WARNING)

try:
    from cellpose import models, dynamics, transforms, utils  # type: ignore
    from cellpose.vit_sam import Transformer  # type: ignore
except ImportError:
    raise ImportError(
        "cellpose package is required. Install with: pip install cellpose"
    )


class CellposeSAM(nn.Module):
    """
    Simplified PyTorch wrapper around Cellpose that operates directly on torch tensors
    with true batching for improved performance. Designed for patch-based processing.
    """

    def __init__(
        self,
        pretrained_model: str = "cpsam",
        device: Optional[torch.device] = None,
        use_bfloat16: bool = True,
    ):
        """
        Initialize CellposeSAMV2 wrapper.

        Args:
            pretrained_model: Path to pretrained cellpose model or model name
            device: Device to run the model on
            use_bfloat16: Use bfloat16 precision for model weights
        """
        super().__init__()  # type: ignore

        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        ### create neural network
        if pretrained_model and not os.path.exists(pretrained_model):
            # check if pretrained model is in the models directory
            model_strings = cast(list[str], models.get_user_models())
            all_models = models.MODEL_NAMES.copy()
            all_models.extend(model_strings)
            if pretrained_model in all_models:
                pretrained_model = os.path.join(models.MODEL_DIR, pretrained_model)
            else:
                pretrained_model = os.path.join(models.MODEL_DIR, "cpsam")
                logger.warning(
                    f"pretrained model {pretrained_model} not found, using default model"
                )

        self.pretrained_model = pretrained_model
        self.use_bfloat16 = use_bfloat16

        # Initialize the underlying Cellpose network directly
        self._init_cellpose_network()

        # Set model to eval mode
        self.eval()

    def _init_cellpose_network(self):
        """Initialize the Cellpose network directly without the wrapper."""
        # Create the network architecture directly
        dtype = torch.bfloat16 if self.use_bfloat16 else torch.float32
        self.net = Transformer(dtype=dtype).to(self.device)

        if os.path.exists(self.pretrained_model):
            logger.info(f">>>> loading model {self.pretrained_model}")
            self.net.load_model(self.pretrained_model, device=self.device)  # type: ignore
        else:
            if os.path.split(self.pretrained_model)[-1] != "cpsam":
                raise FileNotFoundError("model file not recognized")
            models.cache_CPSAM_model_path()
            self.net.load_model(self.pretrained_model, device=self.device)  # type: ignore

    def forward(
        self, x: torch.Tensor, 
        normalize: bool = True, 
        resample: bool = True,
        niter: int = 200,
        flow_threshold: float = 0.4,
        cellprob_threshold: float = 0.0,
        min_size: int = 15,
        max_size_fraction: float = 0.4,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through Cellpose model with true batched processing.

        Args:
            x: Input tensor of shape (B, C, H, W) where C=3 (RGB channels)
            normalize: Whether to normalize input
            resample: Whether to resize flows and cellprob back to original image size
            niter: Number of iterations for mask refinement
            flow_threshold: Threshold for flow field
            cellprob_threshold: Threshold for cell probability map
            min_size: Minimum size of masks to keep
            max_size_fraction: Maximum size fraction of masks to keep

        Returns:
            Dictionary containing:
                - masks: Instance segmentation masks (B, H, W)
                - flows: Flow fields (B, H, W, 2)
                - cellprob: Cell probability maps (B, H, W)
                - styles: Style vectors (B, style_dim)
        """
        if x.dim() == 3:  # Add batch dimension if single image
            x = x.unsqueeze(0)
        elif x.dim() != 4:
            raise ValueError(
                f"Expected input tensor to have 3 or 4 dimensions, got {x.dim()}"
            )
        
        if self.use_bfloat16:
            x = x.to(torch.bfloat16)

        # Store original dimensions for resampling
        _, _, Ly_0, Lx_0 = x.shape

        # Normalize input if requested
        if normalize:
            x = self._normalize_batch(x)

        # Run network inference on entire batch
        with torch.no_grad():
            y, style = self.net(x)[:2]  # y: (B, 3, H, W), style: (B, style_dim)

        # Parse outputs
        batch_size, _, H, W = y.shape
        flows = y[:, :2].permute(0, 2, 3, 1)  # (B, H, W, 2)
        cellprob = y[:, 2]  # (B, H, W)

        # Resample flows and cellprob to original size if needed
        if resample and (H != Ly_0 or W != Lx_0):
            flows = self._resize_flows_batch(flows, Ly_0, Lx_0)
            cellprob = self._resize_cellprob_batch(cellprob, Ly_0, Lx_0)
        
        # Generate masks from flows and cellprob for each image in batch
        all_masks: list[torch.Tensor] = []
        for b in range(batch_size):
            batch_flows = cast(
                np.ndarray[Any, Any],
                flows[b].to(torch.float32).cpu().numpy() # type: ignore
            )  # (H, W, 2)
            batch_cellprob = cast(
                np.ndarray[Any, Any],
                cellprob[b].to(torch.float32).cpu().numpy()  # (H, W) # type: ignore
            )
            
            masks = self._compute_masks(
                batch_flows, 
                batch_cellprob, 
                flow_threshold=flow_threshold,
                cellprob_threshold=cellprob_threshold, min_size=min_size,
                max_size_fraction=max_size_fraction, 
                niter=niter
            )
            all_masks.append(torch.from_numpy(masks).to(self.device))  # type: ignore

        masks_tensor = torch.stack(all_masks, dim=0)  # (B, H, W)

        return {
            "masks": masks_tensor,
            "flows": flows,
            "cellprob": cellprob,
            "styles": style,
        }
    
    def _compute_masks(
        self,
        flows: np.ndarray[Any, Any],
        cellprob: np.ndarray[Any, Any],
        flow_threshold: float = 0.4,
        cellprob_threshold: float = 0.0,
        min_size: int = 15,
        max_size_fraction: float = 0.4,
        niter: int = 200,
    ) -> np.ndarray[Any, Any]:
        """Compute masks from flows and cell probabilities using cellpose dynamics."""

        masks = cast(
            np.ndarray[Any, Any], 
            dynamics.compute_masks( # type: ignore
                flows.transpose(2, 0, 1),  # (2, H, W)
                cellprob,  # (H, W)
                niter=niter,
                flow_threshold=flow_threshold,
                cellprob_threshold=cellprob_threshold,
                min_size=min_size,
                max_size_fraction=max_size_fraction,
            )
        )
        
        masks = cast(
            np.ndarray[Any, Any], 
            utils.fill_holes_and_remove_small_masks( # type: ignore
                masks, 
                min_size=min_size
            )
        )

        return masks

    def _normalize_batch(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize batch of images using torch operations."""
        # Normalize each image in the batch independently
        normalized = torch.zeros_like(x)

        for b in range(x.shape[0]):
            img = x[b]
            # Compute percentiles for normalization
            img_flat = img.view(img.shape[0], -1)  # (C, H*W)

            # Calculate 1st and 99th percentiles per channel
            percentile_1 = torch.kthvalue(
                img_flat, max(1, int(0.01 * img_flat.shape[1])), dim=1
            )[0]
            percentile_99 = torch.kthvalue(
                img_flat, max(1, int(0.99 * img_flat.shape[1])), dim=1
            )[0]

            # Normalize
            percentile_1 = percentile_1.view(-1, 1, 1)  # (C, 1, 1)
            percentile_99 = percentile_99.view(-1, 1, 1)  # (C, 1, 1)

            img_norm = (img - percentile_1) / (percentile_99 - percentile_1 + 1e-6)
            img_norm = torch.clamp(img_norm, 0, 1)

            normalized[b] = img_norm

        return normalized

    def _resize_flows_batch(self, flows: torch.Tensor, Ly: int, Lx: int) -> torch.Tensor:
        """Resize flow fields to target dimensions."""
        try:
            # Convert to numpy for resize operation
            flows_np = cast(
                np.ndarray[Any, Any],
                flows.cpu().numpy() # type: ignore
            )  # (B, H, W, 2)
            B, _, _, _ = flows_np.shape
            
            resized_flows: list[np.ndarray[Any, Any]] = []
            for b in range(B):
                # Move channels to first dimension for transforms.resize_image
                flow = flows_np[b].transpose(2, 0, 1)  # (2, H, W)
                # Resize using cellpose transforms
                flow_resized = cast(
                    np.ndarray[Any, Any],
                    transforms.resize_image( # type: ignore
                        flow, 
                        Ly=Ly, 
                        Lx=Lx, 
                        no_channels=False
                    )
                )
                # Move channels back to last dimension
                flow_resized = flow_resized.transpose(1, 2, 0)  # (Ly, Lx, 2)
                resized_flows.append(flow_resized)
            
            resized_flows_np = np.stack(resized_flows, axis=0)  # (B, Ly, Lx, 2)
            return torch.from_numpy(resized_flows_np).to(flows.device) # type: ignore
        except Exception as e:
            logger.warning(f"Failed to resize flows: {e}")
            return flows 


    def _resize_cellprob_batch(self, cellprob: torch.Tensor, Ly: int, Lx: int) -> torch.Tensor:
        """Resize cell probability maps to target dimensions."""
        try:
            # Convert to numpy for resize operation
            cellprob_np = cast(
                np.ndarray[Any, Any],
                cellprob.cpu().numpy()  # type: ignore
            )  # (B, H, W)
            B, _, _ = cellprob_np.shape
            
            resized_cellprob: list[np.ndarray[Any, Any]] = []
            for b in range(B):
                # Resize using cellpose transforms
                prob_resized = cast(
                    np.ndarray[Any, Any],
                    transforms.resize_image(  # type: ignore
                        cellprob_np[b], 
                        Ly=Ly,
                        Lx=Lx, 
                        no_channels=True
                    )
                )
                resized_cellprob.append(prob_resized)
            
            resized_cellprob_np = np.stack(resized_cellprob, axis=0)  # (B, Ly, Lx)
            return torch.from_numpy(resized_cellprob_np).to(cellprob.device) # type: ignore
        except Exception as e:
            logger.warning(f"Failed to resize cellprob: {e}")
            return cellprob  # Return original if resizing fails

    def eval(self):
        """Set model to evaluation mode."""
        super().eval()
        if hasattr(self, "net"):
            self.net.eval()
        return self

    def train(self, mode: bool = True):
        """Set model to training mode (cellpose doesn't support training)."""
        # Always keep in eval mode for cellpose
        super().train(False)
        if hasattr(self, "net"):
            self.net.eval()
        return self

    def to(self, device: Union[torch.device, str]):  # type: ignore
        """Move model to specified device."""
        super().to(device)
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device

        if hasattr(self, "net"):
            self.net.to(device)

        return self

    def cuda(self, device: Optional[Union[int, torch.device]] = None):
        """Move model to CUDA device."""
        if device is None:
            device = torch.device("cuda")
        elif isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        return self.to(device)

    def cpu(self):
        """Move model to CPU."""
        return self.to(torch.device("cpu"))

    def calculate_instance_map(
        self, predictions: Dict[str, torch.Tensor], magnification: float = 40.0
    ) -> Tuple[torch.Tensor, list[Dict[int, Dict[str, Any]]]]:
        """
        Calculate instance map and extract cell information from predictions.

        Args:
            predictions: Dictionary containing model outputs
            magnification: Magnification level

        Returns:
            Tuple containing instance map and cell information
        """
        masks = predictions["masks"]
        batch_size = masks.shape[0] if masks.dim() > 2 else 1

        if masks.dim() == 2:
            masks = masks.unsqueeze(0)

        instance_types: list[Dict[int, Dict[str, Any]]] = []

        for b in range(batch_size):
            mask = cast(
                np.ndarray[Any, Any],
                masks[b].cpu().numpy(),  # type: ignore
            )
            batch_cells: Dict[int, Dict[str, Any]] = {}

            # Get unique cell IDs
            unique_ids = np.unique(mask)
            unique_ids = unique_ids[unique_ids > 0]  # Remove background

            for cell_id in unique_ids:
                cell_mask = (mask == cell_id).astype(np.uint8)

                # Calculate properties
                coords = np.where(cell_mask)
                if len(coords[0]) > 0:
                    centroid = np.array([coords[0].mean(), coords[1].mean()])
                    bbox = np.array(
                        [
                            [coords[0].min(), coords[1].min()],
                            [coords[0].max(), coords[1].max()],
                        ]
                    )

                    # Extract contour (simplified)
                    contour = self._extract_contour(cell_mask)

                    batch_cells[int(cell_id)] = {
                        "centroid": centroid,
                        "bbox": bbox,
                        "contour": contour,
                        "type": 1,  # Default cell type
                        "type_prob": 1.0,  # Default probability
                    }

            instance_types.append(batch_cells)

        return masks, instance_types

    def _extract_contour(self, mask: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Extract contour from binary mask."""
        try:
            import cv2

            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                return largest_contour.squeeze()
            else:
                coords = np.where(mask)
                if len(coords[0]) > 0:
                    return np.column_stack((coords[1], coords[0]))  # x, y format
                else:
                    return np.array([[0, 0]])
        except ImportError:
            # Fallback without OpenCV
            coords = np.where(mask)
            if len(coords[0]) > 0:
                return np.column_stack((coords[1], coords[0]))  # x, y format
            else:
                return np.array([[0, 0]])

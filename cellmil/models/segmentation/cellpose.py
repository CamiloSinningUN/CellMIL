import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Union, List

try:
    from cellpose import models  # type: ignore
except ImportError:
    raise ImportError(
        "cellpose package is required. Install with: pip install cellpose"
    )


class CellposeSAM(nn.Module):
    """
    PyTorch nn.Module wrapper around Cellpose model for cell instance segmentation.
    This wrapper allows Cellpose to be used like other PyTorch models in the segmentation pipeline.
    """

    def __init__(
        self,
        pretrained_model: str = "cpsam",
        device: Optional[torch.device] = None,
        **kwargs: Any,
    ):
        """
        Initialize CellposeSAM wrapper.

        Args:
            pretrained_model: Path to pretrained cellpose model or model name
            device: Device to run the model on
            **kwargs: Additional arguments passed to CellposeModel
        """
        super().__init__() # type: ignore

        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.pretrained_model = pretrained_model

        # Initialize Cellpose model
        gpu = self.device.type in ["cuda", "mps"]
        self.cellpose_model = models.CellposeModel(
            gpu=gpu, pretrained_model=pretrained_model, device=self.device, **kwargs
        )

        # Set model to eval mode
        self.eval()

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through Cellpose model.

        Args:
            x: Input tensor of shape (B, C, H, W) where C=3 (RGB channels)

        Returns:
            Dictionary containing:
                - masks: Instance segmentation masks
                - flows: Flow fields
                - styles: Style vectors
                - cellprob: Cell probability maps
        """
        # Convert tensor to numpy array for Cellpose
        # Handle batch dimension
        if x.dim() == 4:  # (B, C, H, W)
            batch_size = x.shape[0]
            results = []

            for i in range(batch_size):
                img = x[i].cpu().numpy()
                # Convert from (C, H, W) to (H, W, C)
                img = np.transpose(img, (1, 2, 0)) 

                # Ensure image is in the right format (0-255 uint8 or 0-1 float)
                if img.dtype == np.float32 or img.dtype == np.float64: 
                    if img.max() <= 1.0:
                        img = (img * 255).astype(np.uint8)
                    else:
                        img = img.astype(np.uint8)

                # Run Cellpose inference
                masks, flows, styles = self.cellpose_model.eval(
                    img, batch_size=1, compute_masks=True, normalize=True
                )

                # Convert results back to tensors
                result = self._convert_outputs_to_tensors(
                    masks, flows, styles, img.shape[:2]
                )
                results.append(result)

            # Stack results for batch
            return self._stack_batch_results(results)

        elif x.dim() == 3:  # (C, H, W) - single image
            img = x.cpu().numpy()
            img = np.transpose(img, (1, 2, 0))

            # Ensure image is in the right format
            if img.dtype == np.float32 or img.dtype == np.float64:
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                else:
                    img = img.astype(np.uint8)

            # Run Cellpose inference
            masks, flows, styles = self.cellpose_model.eval(
                img, batch_size=1, compute_masks=True, normalize=True
            )

            return self._convert_outputs_to_tensors(
                masks, flows, styles, img.shape[:2]
            )

        else:
            raise ValueError(
                f"Expected input tensor to have 3 or 4 dimensions, got {x.dim()}"
            )

    def _convert_outputs_to_tensors(
        self, masks: Any, flows: Any, styles: Any, image_shape: tuple[int, int]
    ) -> Dict[str, torch.Tensor]:
        """
        Convert Cellpose outputs to PyTorch tensors.

        Args:
            masks: Instance masks from Cellpose
            flows: Flow outputs from Cellpose
            styles: Style vectors from Cellpose
            image_shape: Original image shape (H, W)

        Returns:
            Dictionary of converted tensors
        """
        result = {}

        # Convert masks
        if isinstance(masks, np.ndarray):
            result["masks"] = torch.from_numpy(masks).to(self.device)
        else:
            result["masks"] = torch.zeros(
                image_shape, dtype=torch.long, device=self.device
            )

        # Convert flows
        if flows is not None and len(flows) > 0:
            if len(flows) >= 3:
                # flows[0]: RGB flow visualization
                # flows[1]: XY flows
                # flows[2]: cell probability
                if flows[1] is not None:
                    result["flows"] = torch.from_numpy(flows[1]).to(self.device)
                if flows[2] is not None:
                    result["cellprob"] = torch.from_numpy(flows[2]).to(self.device)

        # Set default flows if not available
        if "flows" not in result:
            result["flows"] = torch.zeros((*image_shape, 2), device=self.device)
        if "cellprob" not in result:
            result["cellprob"] = torch.zeros(image_shape, device=self.device)

        # Convert styles
        if isinstance(styles, np.ndarray):
            result["styles"] = torch.from_numpy(styles).to(self.device)
        else:
            result["styles"] = torch.zeros(
                256, device=self.device
            )  # Default style vector size

        return result

    def _stack_batch_results(
        self, results: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        Stack batch results into single tensors.

        Args:
            results: List of result dictionaries from individual images

        Returns:
            Dictionary with stacked tensors
        """
        if not results:
            return {}

        stacked: dict[str, torch.Tensor] = {}
        for key in results[0].keys():
            tensors = [r[key] for r in results]

            if key == "styles":
                # Stack style vectors along batch dimension
                stacked[key] = torch.stack(tensors, dim=0)
            else:
                # For spatial outputs, add batch dimension
                stacked[key] = torch.stack(tensors, dim=0)

        return stacked

    def eval(self):
        """Set model to evaluation mode."""
        super().eval()
        if hasattr(self.cellpose_model, "net"):
            if hasattr(self.cellpose_model.net, "eval"):
                self.cellpose_model.net.eval()
        return self

    def train(self, mode: bool = True):
        """Set model to training mode (not implemented for Cellpose)."""
        # Cellpose doesn't support training mode in this wrapper
        # Always keep in eval mode, but properly call super().train() to avoid recursion
        super().train(False)  # Always set to eval mode (False = eval, True = train)
        if hasattr(self.cellpose_model, "net"):
            if hasattr(self.cellpose_model.net, "eval"):
                self.cellpose_model.net.eval()
        return self

    def to(self, device: Union[torch.device, str]):
        """Move model to specified device."""
        super().to(device)
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device

        # Update Cellpose model device
        if hasattr(self.cellpose_model, "device"):
            self.cellpose_model.device = device
        if hasattr(self.cellpose_model, "net"):
            if hasattr(self.cellpose_model.net, "to"):
                self.cellpose_model.net.to(device)

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

    def parameters(self):
        """Return model parameters (for compatibility with PyTorch optimizers)."""
        # Cellpose models don't expose parameters in the standard way
        # Return empty iterator for compatibility
        return iter([])

    def state_dict(self):
        """Return state dictionary (for compatibility with PyTorch save/load)."""
        return {"pretrained_model": self.pretrained_model, "device": str(self.device)}

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True):
        """Load state dictionary (for compatibility with PyTorch save/load)."""
        if "pretrained_model" in state_dict:
            self.pretrained_model = state_dict["pretrained_model"]
        if "device" in state_dict:
            self.device = torch.device(state_dict["device"])

        # Reinitialize Cellpose model with new parameters
        gpu = self.device.type in ["cuda", "mps"]
        self.cellpose_model = models.CellposeModel(
            gpu=gpu, pretrained_model=self.pretrained_model, device=self.device
        )

    def calculate_instance_map(
        self, predictions: Dict[str, torch.Tensor], magnification: float = 40.0
    ) -> tuple[torch.Tensor, list[dict[np.int32, dict[str, Any]]]]:
        """
        Calculate instance map and extract cell information from Cellpose predictions.

        Args:
            predictions: Dictionary containing model outputs (masks, flows, cellprob, styles)
            magnification: Magnification level of the image

        Returns:
            Tuple containing:
                - instance_map: Tensor with instance segmentation
                - instance_types: List of dictionaries with cell information for each image in batch
        """
        masks = predictions["masks"]
        batch_size = masks.shape[0] if masks.dim() > 2 else 1

        if masks.dim() == 2:
            masks = masks.unsqueeze(0)  # Add batch dimension

        instance_types = []

        for b in range(batch_size):
            mask = masks[b].cpu().numpy()
            batch_cells = {}

            # Get unique cell IDs (excluding background = 0)
            unique_ids = np.unique(mask)
            unique_ids = unique_ids[unique_ids > 0]  # Remove background

            for cell_id in unique_ids:
                cell_mask = (mask == cell_id).astype(np.uint8)

                # Calculate bounding box
                coords = np.where(cell_mask)
                if len(coords[0]) == 0:
                    continue

                y_min, y_max = coords[0].min(), coords[0].max()
                x_min, x_max = coords[1].min(), coords[1].max()
                bbox = np.array([[y_min, x_min], [y_max + 1, x_max + 1]])

                # Calculate centroid
                centroid_y = coords[0].mean()
                centroid_x = coords[1].mean()
                centroid = np.array([centroid_y, centroid_x])

                # Extract contour
                contour = self._extract_contour(cell_mask)

                # For Cellpose, we don't have type classification, so all cells are type 1 (Cell)
                cell_info = {
                    "bbox": bbox,
                    "centroid": centroid,
                    "contour": contour,
                    "type_prob": 1.0,  # Cellpose doesn't provide type probabilities
                    "type": 1,  # All cells are classified as "Cell" type
                }

                batch_cells[np.int32(cell_id)] = cell_info

            instance_types.append(batch_cells)

        return masks, instance_types

    def _extract_contour(self, mask: np.ndarray) -> np.ndarray:
        """
        Extract contour from binary mask.

        Args:
            mask: Binary mask of the cell

        Returns:
            Contour points as numpy array
        """
        try:
            import cv2

            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if contours:
                # Get the largest contour
                contour = max(contours, key=cv2.contourArea)
                return contour.squeeze()
            else:
                # Fallback: return boundary points
                coords = np.where(mask)
                if len(coords[0]) > 0:
                    return np.column_stack((coords[0], coords[1]))
                else:
                    return np.array([])
        except ImportError:
            # Fallback if OpenCV is not available: return boundary points
            coords = np.where(mask)
            if len(coords[0]) > 0:
                return np.column_stack((coords[0], coords[1]))
            else:
                return np.array([])

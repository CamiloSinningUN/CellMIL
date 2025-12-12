import torch
import ujson
import pandas as pd
import numpy as np
from typing import cast, Tuple, List, Optional, Dict, Union, Any
from pathlib import Path
from cellmil.utils import logger
import matplotlib.pyplot as plt
from cellmil.interfaces.FeatureExtractorConfig import (
    ExtractorType,
    FeatureExtractionType,
)
from cellmil.interfaces.CellSegmenterConfig import ModelType, TYPE_NUCLEI_DICT
from cellmil.interfaces.GraphCreatorConfig import GraphCreatorType
from torch_geometric.data import Data  # type: ignore
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union


def column_sanity_check(data: pd.DataFrame | None, label: Union[str, Tuple[str, str]]) -> None:
    """Perform sanity checks on the input data."""
    if data is None or data.empty:
        raise ValueError("Input data is empty or None.")

    # Handle both single label and survival data (duration, event) tuple
    if isinstance(label, tuple):
        required_columns = ["FULL_PATH", label[0], label[1]]
    else:
        required_columns = ["FULL_PATH", label]
    
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"Missing required column: {col}")


def preprocess_row(
    row: pd.Series,
    label: Union[str, Tuple[str, str]],
    folder: Path,
    extractor: ExtractorType | List[ExtractorType],
    graph_creator: Optional[GraphCreatorType] = None,
    segmentation_model: Optional[ModelType] = None,
    do_validate_features: bool = True
) -> Tuple[str, Union[int, Tuple[float, int]]] | Tuple[None, ...]:
    """
    Process a single slide row to extract slide name and validate features.

    Args:
        row: A pandas Series representing a row from the Excel file
        label: Either a single string (classification) or tuple of (duration, event) strings (survival)

    Returns:
        For classification: Tuple of (slide_name, label)
        For survival: Tuple of (slide_name, (duration, event))
        On error: Tuple of (None, None)
    """
    try:
        file_path = Path(cast(str, row["FULL_PATH"]))
        slide_name = extract_slide_name(file_path)
        if do_validate_features:
            valid_features = validate_features(
                folder, slide_name, extractor, graph_creator, segmentation_model
            )
        else:
            valid_features = True
        if slide_name and valid_features:
            # Handle survival data (tuple of duration and event columns)
            if isinstance(label, tuple):
                duration_col, event_col = label
                duration = float(row[duration_col])
                event = int(row[event_col])
                return slide_name, (duration, event)
            else:
                # Regular classification label
                return slide_name, row[label]
        else:
            logger.warning(
                f"Skipping slide {slide_name}: slide has no valid features or slide name is invalid"
            )
            return None, None
    except Exception as e:
        logger.warning(f"Error processing slide row: {e}")
        return None, None


def validate_features(
    folder: Path,
    slide_name: str,
    extractor: ExtractorType | List[ExtractorType],
    graph_creator: Optional[GraphCreatorType] = None,
    segmentation_model: Optional[ModelType] = None,
):
    """
    Check if the feature file(s) exist and contain valid (non-empty) features.
    """

    def _check_single(
        extractor: ExtractorType,
    ) -> Tuple[bool, Optional[torch.Tensor], Dict[int, int]]:
        p = get_feature_path(
            folder, slide_name, extractor, graph_creator, segmentation_model
        )
        if not p.exists():
            logger.warning(f"Feature file does not exist for slide {slide_name}: {p}")
            return False, None, {}
        try:
            data = torch.load(p, map_location="cpu", weights_only=False)
            if "features" not in data:
                logger.warning(
                    f"No 'features' key in data for slide {slide_name} ({extractor})"
                )
                return False, None, {}
            ft = data["features"]
            if ft.numel() == 0 or ft.shape[0] == 0:
                logger.warning(
                    f"Empty features tensor for slide {slide_name} ({extractor}): shape {ft.shape}"
                )
                return False, None, {}
            if ft.shape[0] < 200:
                logger.warning(
                    f"Insufficient features for slide {slide_name} ({extractor}): {ft.shape[0]} < 200"
                )
                return False, None, {}
            ci = cast(Dict[int, int], data.get("cell_indices", {}))
            return True, ft, ci
        except Exception as e:
            logger.warning(
                f"Error loading features for slide {slide_name} ({extractor}): {e}"
            )
            return False, None, {}

    if isinstance(extractor, list):
        results = [_check_single(ext) for ext in extractor]
        if not all(ok for ok, _, _ in results):
            logger.warning(f"Some extractors failed validation for slide {slide_name}")
            return False
        fts = [cast(torch.Tensor, ft) for _, ft, _ in results]
        cis = [ci for _, _, ci in results]
        have_maps = all(len(m) > 0 for m in cis)
        if have_maps:
            common = set(cis[0].keys())
            original_counts = [len(m) for m in cis]

            # Track which cells are missing from which extractors
            all_cells = cast(set[int], set().union(*[set(m.keys()) for m in cis]))  # type: ignore

            for _, m in enumerate(cis[1:], 1):
                common &= set(m.keys())

            if len(common) == 0:
                logger.warning(
                    f"No overlapping cell ids across extractors for slide {slide_name}. "
                    f"Extractor cell counts: {original_counts}"
                )
                return False

            # Log detailed information about missing cells per extractor
            total_unique_cells = len(all_cells)
            excluded_cells = total_unique_cells - len(common)

            if excluded_cells > 0:
                logger.warning(
                    f"Validation: {excluded_cells} cells will be excluded from slide {slide_name} "
                    f"due to missing features in some extractors. "
                    f"Will use {len(common)} common cells out of {total_unique_cells} total unique cells."
                )

                # Log which extractors are missing which cells
                for _, (ext, cell_map) in enumerate(zip(extractor, cis)):
                    missing_cells = all_cells - set(cell_map.keys())
                    if missing_cells:
                        logger.warning(
                            f"  Extractor {ext} is missing {len(missing_cells)} cells: "
                            f"cell IDs {sorted(list(missing_cells))[:10]}{'...' if len(missing_cells) > 10 else ''}"
                        )
                    else:
                        logger.info(f"  Extractor {ext} has all {len(cell_map)} cells")

            return True
        # Without mappings, require same number of instances to allow naive concat
        n0 = fts[0].size(0)
        if any(ft.size(0) != n0 for ft in fts):
            logger.warning(
                f"Mismatched instance counts without cell_indices for slide {slide_name}. "
                f"Counts: {[ft.size(0) for ft in fts]}"
            )
            return False
        return True
    else:
        ok, _, _ = _check_single(extractor)
        return ok


def get_feature_path(
    folder: Path,
    slide_name: str,
    extractor: ExtractorType,
    graph_creator: Optional[GraphCreatorType] = None,
    segmentation_model: Optional[ModelType] = None,
) -> Path:
    """
    Get the path to the feature file for the given slide.
    """
    if extractor in FeatureExtractionType.Embedding:
        return (
            folder / slide_name / "feature_extraction" / str(extractor) / "features.pt"
        )

    if segmentation_model is None:
        raise ValueError("Segmentation model is not set")

    if extractor in FeatureExtractionType.Morphological:
        return (
            folder
            / slide_name
            / "feature_extraction"
            / str(extractor)
            / str(segmentation_model)
            / "features.pt"
        )

    if graph_creator is None:
        raise ValueError("Graph creator is not set")

    if extractor in FeatureExtractionType.Topological:
        return (
            folder
            / slide_name
            / "feature_extraction"
            / str(extractor)
            / str(graph_creator)
            / str(segmentation_model)
            / "features.pt"
        )

    raise ValueError(f"Unknown extractor type: {extractor}")


def filter_split(data: pd.DataFrame, split: str) -> pd.DataFrame:
    """Filter the DataFrame by the specified split."""
    data = data[data["SPLIT"] == split]
    logger.info(f"Using {split} split: {len(data)} slides")
    return data


def apply_permutation(features: torch.Tensor) -> torch.Tensor:
    """
    Randomly permute the order of instances (rows).

    Args:
        features: Input feature tensor of shape (n_instances, n_features)

    Returns:
        Feature tensor with rows permuted
    """
    if features.size(0) > 1:  # Only shuffle if there are multiple instances
        # Generate random permutation indices
        perm_indices = torch.randperm(features.size(0))
        # Shuffle the rows using the permutation indices
        features = features[perm_indices]
    return features


def subsample_and_pad(
    features: torch.Tensor,
    target_size: int,
) -> torch.Tensor:
    """
    Randomly subsample or pad the bag to a fixed target size by replicating rows.

    Args:
        features: Input feature tensor of shape (n_instances, n_features)
        target_size: Desired number of instances per bag

    Returns:
        Tensor of shape (target_size, n_features)
    """
    n = features.size(0)
    if n == 0:
        raise ValueError("Features tensor is empty; cannot subsample/pad an empty bag.")
    if n == target_size:
        return features
    if n > target_size:
        idx = torch.randperm(n)[:target_size]
        return features[idx]
    # n < target_size: pad by replicating rows
    pad_count = target_size - n
    pad_idx = torch.randint(low=0, high=n, size=(pad_count,), dtype=torch.long)
    return torch.cat([features, features[pad_idx]], dim=0)


def wsl_preprocess(data: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data to ensure paths are correctly formatted."""
    data_copy = data.copy()
    data_copy["FULL_PATH"] = data_copy["FULL_PATH"].apply(  # type: ignore
        lambda path: path.replace("\\", "/").replace("D:", "/mnt/d")  # type: ignore
    )
    return data_copy


def extract_slide_name(file_path: Path) -> str:
    """Extract slide name from full path."""
    # Get the last part of the path (filename) and remove extension
    if not file_path:
        return ""
    return file_path.stem


def get_cell_detection_path(
    folder: Path, slide_name: str, segmentation_model: ModelType
) -> Path:
    """
    Get the path to the cell detection file for the given slide.
    """

    return (
        folder
        / slide_name
        / "cell_detection"
        / str(segmentation_model)
        / "cell_detection.json"
    )


def get_cell_types(
    folder: Path, slide_name: str, segmentation_model: ModelType
) -> Dict[int, int] | None:
    # Load cell detection data once
    cell_detection_path = get_cell_detection_path(
        folder, slide_name, segmentation_model
    )

    if not cell_detection_path.exists():
        return None

    with open(cell_detection_path, "r") as f:
        cell_data = ujson.load(f)

    cells = cell_data.get("cells", [])

    # Use dictionary comprehension for faster processing
    cell_type_dict = {
        cell["cell_id"]: cell.get("type", 0)
        for cell in cells
        if cell.get("cell_id") is not None
    }

    return cell_type_dict


def get_centroids(
    folder: Path, slide_name: str, segmentation_model: ModelType
) -> Dict[int, Tuple[float, float]] | None:
    """
    Get centroids for cells from the segmentation data.

    Args:
        folder: Path to the dataset folder
        slide_name: Name of the slide
        segmentation_model: Segmentation model used

    Returns:
        Dictionary mapping cell_id to (x, y) centroid coordinates, or None if data not found
    """
    # Load cell detection data once
    cell_detection_path = get_cell_detection_path(
        folder, slide_name, segmentation_model
    )

    if not cell_detection_path.exists():
        return None


    with open(cell_detection_path, "r") as f:
        cell_data = ujson.load(f)

    cells = cell_data.get("cells", [])

    # Use dictionary comprehension for faster processing
    # This is significantly faster than a for loop with dict.append
    centroid_dict = {
        cell["cell_id"]: (float(cell["centroid"][0]), float(cell["centroid"][1]))
        for cell in cells
        if cell.get("cell_id") is not None
        and cell.get("centroid") is not None
        and len(cell["centroid"]) >= 2
    }

    return centroid_dict


def centroids_to_tensor(
    centroids: Dict[int, Tuple[float, float]], cell_indices: Dict[int, int]
) -> torch.Tensor:
    """
    Convert centroids dictionary to a tensor.

    Args:
        centroids: Dictionary mapping cell_id to (x, y) centroid coordinates
        cell_indices: Dictionary mapping cell_id to its index in the tensor

    Returns:
        A tensor containing the centroid coordinates [num_cells, 2]
    """
    centroid_tensor = torch.zeros(len(cell_indices), 2, dtype=torch.float32)

    for cell_id, (x, y) in centroids.items():
        if cell_id in cell_indices:
            centroid_tensor[cell_indices[cell_id], 0] = x
            centroid_tensor[cell_indices[cell_id], 1] = y

    return centroid_tensor


def cell_types_to_tensor(
    cell_types: Dict[int, int], cell_indices: Dict[int, int]
) -> torch.Tensor:
    """
    Convert cell types dictionary to a tensor.

    Args:
        cell_types: Dictionary mapping cell_id to cell_type
        cell_indices: Dictionary mapping cell_id to its index in the tensor

    Returns:
        A tensor containing the cell types
    """
    cell_type_tensor = torch.zeros(
        len(cell_indices), len(TYPE_NUCLEI_DICT), dtype=torch.long
    )

    for cell_id, cell_type in cell_types.items():
        if cell_id in cell_indices:
            cell_type_tensor[cell_indices[cell_id], cell_type - 1] = 1.0
    return cell_type_tensor


def get_cell_features(
    folder: Path,
    slide_name: str,
    extractor: Union[ExtractorType, List[ExtractorType]],
    graph_creator: GraphCreatorType | None = None,
    segmentation_model: ModelType | None = None,
) -> Tuple[torch.Tensor | None, Dict[int, int] | None, List[str] | None]:
    """
    Get the features for a specific slide using the specified extractor.

    Args:
        folder: Path to the dataset folder
        slide_name: Name of the slide
        extractor: Feature extractor type or list of types to use for feature extraction.
        graph_creator: Optional graph creator type, needed for some extractors
        segmentation_model: Optional Segmentation model type, needed for some extractors

    Returns:
        A tensor containing the extracted features, or None if extraction failed.
    """

    if isinstance(extractor, list):
        tensors: List[torch.Tensor] = []
        maps: List[Dict[int, int]] = []
        feature_names: List[str] = []

        for ext in extractor:
            data = torch.load(
                get_feature_path(
                    folder, slide_name, ext, graph_creator, segmentation_model
                ),
                map_location="cpu",
                weights_only=False,
            )
            tensors.append(data["features"])  # (N_i, F_i)
            maps.append(cast(Dict[int, int], data.get("cell_indices", {})))

            ext_feature_names = cast(list[str] | None, data.get("feature_names", None))

            if ext_feature_names:
                feature_names.extend(ext_feature_names)

        have_maps = all(len(m) > 0 for m in maps)
        if have_maps:
            # Align by common cell ids
            common = set(maps[0].keys())
            original_counts = [len(m) for m in maps]

            # Track which cells are missing from which extractors
            all_cells = cast(set[int], set().union(*maps))  # type: ignore

            for m in maps[1:]:
                common &= set(m.keys())
            ordered = sorted(common)
            if len(ordered) == 0:
                logger.warning(
                    f"No overlapping cells across extractors for slide {slide_name}. "
                    f"Extractor cell counts: {original_counts}"
                )
                raise ValueError(
                    f"No overlapping cells across extractors for slide {slide_name}."
                )

            # Log cells that will be excluded with detailed per-extractor information
            total_unique_cells = len(all_cells)
            excluded_cells = total_unique_cells - len(ordered)
            if excluded_cells > 0:
                logger.warning(
                    f"Excluding {excluded_cells} cells from slide {slide_name} due to missing features in some extractors. "
                    f"Using {len(ordered)} common cells out of {total_unique_cells} total unique cells."
                )

                # Log which extractors are missing which cells
                for _, (ext, cell_map) in enumerate(zip(extractor, maps)):
                    missing_cells = all_cells - set(cell_map.keys())
                    if missing_cells:
                        logger.warning(
                            f"  Extractor {ext} is missing {len(missing_cells)} cells: "
                            f"cell IDs {sorted(list(missing_cells))[:10]}{'...' if len(missing_cells) > 10 else ''}"
                        )
                    else:
                        logger.info(f"  Extractor {ext} has all {len(cell_map)} cells")

            aligned: List[torch.Tensor] = []
            for t, m in zip(tensors, maps):
                idxs = torch.tensor([m[cid] for cid in ordered], dtype=torch.long)
                aligned.append(t[idxs])
            features = torch.cat(aligned, dim=1)
            cell_indices: Dict[int, int] = {cid: i for i, cid in enumerate(ordered)}
        else:
            # Fallback: naive concat if instance counts match
            n0 = tensors[0].size(0)
            if any(t.size(0) != n0 for t in tensors):
                raise ValueError(
                    f"Cannot align features without cell_indices for slide {slide_name}: mismatched instance counts."
                )
            features = torch.cat(tensors, dim=1)
            cell_indices = {}
    else:
        features_data = torch.load(
            get_feature_path(
                folder, slide_name, extractor, graph_creator, segmentation_model
            ),
            map_location="cpu",
            weights_only=False,
        )
        features = features_data["features"]
        cell_indices = cast(Dict[int, int], features_data.get("cell_indices", {}))
        feature_names = cast(List[str], features_data.get("feature_names", []))

    return features, cell_indices, feature_names


def compute_normalization(features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    logger.info(
        f"Computing robust scaling parameters for {features.shape[1]} features using {features.shape[0]} instances..."
    )
    # Compute median and IQR (75th percentile - 25th percentile) for each feature
    epsilon = 1e-8
    features = torch.sign(features) * torch.log1p(torch.abs(features) + epsilon)
    median_values = torch.median(features, dim=0)[0]  # Shape: (n_features,)
    q1 = torch.quantile(features, 0.25, dim=0)
    q3 = torch.quantile(features, 0.75, dim=0)
    iqr_values = q3 - q1

    # Handle features with zero/small IQR (near-constant features)
    constant_mask = iqr_values <= 1e-8

    if constant_mask.sum() > 0:
        logger.info(
            f"Found {constant_mask.sum()} near-constant features with very small IQR"
        )
        # For constant features, set IQR to 1 to avoid division by zero
        iqr_values[constant_mask] = 1.0

    return median_values, iqr_values


def correlation_filter(
    features: torch.Tensor, correlation_threshold: float, plot: bool = True
):
    # Store shape info before clearing memory
    total_features = features.shape[1]
    total_instances = features.shape[0]

    logger.info(
        f"Computing correlation matrix for {total_features} features using {total_instances} instances..."
    )

    feature_std = features.std(dim=0)
    non_constant_mask = feature_std > 1e-8

    # Clear feature_std as it's no longer needed
    del feature_std

    if non_constant_mask.sum() == 0:
        raise ValueError("All features are constant. Skipping correlation filter.")
    else:
        logger.info(
            f"Found {non_constant_mask.sum()} non-constant features out of {total_features} total features"
        )

    # Only compute correlation for non-constant features - extract and delete original
    valid_features = features[:, non_constant_mask]
    del features  # Free the large original tensor immediately

    # Compute correlation matrix with minimal intermediate tensors
    # Center the features in-place
    feature_means = valid_features.mean(dim=0)
    valid_features -= feature_means  # In-place subtraction to save memory
    del feature_means  # Clean up means

    # Compute covariance matrix directly
    n_samples = valid_features.shape[0]
    cov_matrix = torch.mm(valid_features.T, valid_features) / (n_samples - 1)
    del valid_features  # Free the centered features tensor

    # Compute correlation matrix in-place
    # Extract diagonal for standard deviations
    std_devs = torch.sqrt(torch.diag(cov_matrix))

    # Compute correlation matrix by modifying cov_matrix in-place
    corr_matrix = cov_matrix / torch.outer(std_devs, std_devs)
    del cov_matrix, std_devs  # Clean up intermediate tensors

    # Plot correlation matrix if requested
    if plot:
        try:
            # Convert to numpy for plotting (create a copy to avoid modifying original)
            corr_np = corr_matrix.detach().cpu().numpy()  # type: ignore

            # Create the plot
            _, ax = plt.subplots(figsize=(12, 10))  # type: ignore

            # Create heatmap
            im = ax.imshow(corr_np, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")  # type: ignore

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)  # type: ignore
            cbar.set_label("Correlation Coefficient", rotation=270, labelpad=20)  # type: ignore

            # Set title and labels
            ax.set_title(  # type: ignore
                f"Feature Correlation Matrix\n({corr_np.shape[0]} non-constant features)",
                fontsize=14,
                pad=20,
            )
            ax.set_xlabel("Feature Index", fontsize=12)  # type: ignore
            ax.set_ylabel("Feature Index", fontsize=12)  # type: ignore

            # Add grid for better readability
            ax.grid(True, alpha=0.3)  # type: ignore

            # Adjust layout and save
            plt.tight_layout()
            plt.show()  # type: ignore

            logger.info("Correlation matrix plot saved as 'correlation_matrix.png'")

        except Exception as e:
            logger.warning(f"Failed to create correlation matrix plot: {e}")

    # Find highly correlated pairs
    upper_triangle = torch.triu(torch.abs(corr_matrix), diagonal=1)
    high_corr_pairs = torch.where(upper_triangle > correlation_threshold)

    # Store the count before cleaning up
    num_high_corr_pairs = len(high_corr_pairs[0])

    # Clean up correlation matrix to save memory
    del corr_matrix, upper_triangle

    # Create mask for features to keep
    features_to_remove: set[int] = set()

    # Convert to int lists for iteration
    row_indices = [int(x) for x in high_corr_pairs[0]]
    col_indices = [int(x) for x in high_corr_pairs[1]]

    # Clean up high_corr_pairs tensors
    del high_corr_pairs

    for i, j in zip(row_indices, col_indices):
        if i not in features_to_remove and j not in features_to_remove:
            # Remove the second feature (j) of the pair
            features_to_remove.add(j)

    # Create final mask mapping back to original feature space
    keep_mask = torch.ones(total_features, dtype=torch.bool)

    # Map back to original indices
    valid_indices = torch.where(non_constant_mask)[0]
    for idx_to_remove in features_to_remove:
        original_idx = valid_indices[idx_to_remove]
        keep_mask[original_idx] = False

    # Also remove constant features
    keep_mask = keep_mask & non_constant_mask

    # Clean up valid_indices tensor
    del valid_indices

    features_removed = (~keep_mask).sum().item()
    features_kept = keep_mask.sum().item()

    logger.info(
        f"Correlation filter: removed {features_removed} features, kept {features_kept} features"
    )
    logger.info(
        f"Found {num_high_corr_pairs} highly correlated pairs (threshold: {correlation_threshold})"
    )

    return keep_mask, non_constant_mask


def weights_for_sampler(labels: list[int]) -> torch.Tensor:
    """
    Compute weights for WeightedRandomSampler to handle class imbalance.

    The weight for each sample is computed as 1 / (class_frequency * num_samples_in_class).
    This gives higher weights to samples from underrepresented classes.

    Returns:
        torch.Tensor: Weights for each sample in the dataset, with shape (len(dataset),).
                     These weights can be used directly with torch.utils.data.WeightedRandomSampler.

    """
    if len(labels) == 0:
        logger.warning("No labels found in dataset. Returning uniform weights.")
        return torch.ones(len(labels), dtype=torch.float32)

    # Convert labels to tensor for easier computation
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    # Count frequency of each class
    unique_labels, counts = cast(
        tuple[torch.Tensor, torch.Tensor],
        torch.unique(labels_tensor, return_counts=True),  # type: ignore
    )

    # Create a mapping from label to its frequency
    label_to_count = {
        label.item(): count.item() for label, count in zip(unique_labels, counts)
    }

    # Compute weight for each sample: 1 / count_of_its_class
    weights = torch.zeros(len(labels), dtype=torch.float32)
    for i, label in enumerate(labels):
        weights[i] = 1.0 / label_to_count[label]

    # Normalize weights so they sum to the number of samples
    weights = weights * len(weights) / weights.sum()

    logger.info(f"Computed sampling weights for {len(unique_labels)} classes:")
    for label, count in label_to_count.items():
        weight_per_sample = 1.0 / count
        logger.info(
            f"  Class {label}: {count} samples, weight per sample: {weight_per_sample:.4f}"
        )

    return weights


def load_precomputed_graph(
    folder: Path,
    slide_name: str,
    graph_creator: GraphCreatorType,
    segmentation_model: ModelType,
) -> Data:
    """
    Load pre-computed graph from disk.

    Args:
        folder: Base folder containing slide data
        slide_name: Name of the slide
        graph_creator: Graph creator type used (string or enum)
        segmentation_model: Segmentation model used (string or enum)

    Returns:
        Data object containing the loaded graph

    Raises:
        ValueError: If graph file doesn't exist or has invalid format
    """
    graph_path = (
        folder / slide_name / "graphs" / graph_creator / segmentation_model / "graph.pt"
    )

    if not graph_path.exists():
        raise ValueError(f"Pre-computed graph not found at {graph_path}")

    try:
        graph_dict = torch.load(graph_path, map_location="cpu", weights_only=False)

        if not isinstance(graph_dict, dict):
            raise ValueError(f"Expected graph dictionary, got {type(graph_dict)}")

        required_keys = ["node_features", "edge_indices", "edge_features"]
        missing_keys = [key for key in required_keys if key not in graph_dict]
        if missing_keys:
            raise ValueError(f"Graph missing required keys {missing_keys}")

        node_features = cast(torch.Tensor, graph_dict["node_features"])
        edge_indices = cast(torch.Tensor, graph_dict["edge_indices"])
        edge_features = cast(torch.Tensor, graph_dict["edge_features"])

        if edge_indices.dim() != 2 or edge_indices.shape[0] != 2:
            raise ValueError(
                f"edge_indices should have shape [2, num_edges], got {edge_indices.shape}"
            )

        if node_features.shape[1] < 1:
            raise ValueError("Graph node features seem empty")

        graph_data = Data(
            x=node_features,
            edge_index=edge_indices,
            edge_attr=edge_features,
            num_nodes=node_features.shape[0],
        )

        if "metadata" in graph_dict:
            graph_data.metadata = graph_dict["metadata"]

        return graph_data

    except Exception as e:
        raise ValueError(
            f"Failed to load pre-computed graph for slide {slide_name}: {e}"
        )


def merge_graph_with_features(
    graph_data: Data,
    features: torch.Tensor,
    cell_indices: Dict[int, int],
    cell_coordinates: torch.Tensor,
) -> Data:
    """
    Merge pre-computed graph structure with extracted features, ensuring proper alignment.

    This function ensures features are correctly assigned to their corresponding graph nodes
    based on cell IDs, creating a proper subgraph with aligned features.

    Args:
        graph_data: Data object containing graph structure
        features: Feature tensor
        cell_indices: Mapping from cell_id to feature tensor index
        cell_coordinates: Optional cell coordinates tensor [num_cells, 2]

    Returns:
        Data object with properly aligned features and graph structure
    """
    # Extract graph components from Data object
    node_features = graph_data.x
    edge_indices = graph_data.edge_index
    edge_features = getattr(graph_data, "edge_attr", None)

    if node_features is None:
        raise ValueError("Graph node features are None")

    if edge_indices is None:
        raise ValueError("Graph edge indices are None")

    # Extract cell_ids from graph node features (assuming first column contains cell IDs)
    graph_cell_ids = node_features[:, 0].long()

    # Create mapping from cell_id to graph node index
    cell_id_to_graph_idx = {
        cell_id.item(): idx for idx, cell_id in enumerate(graph_cell_ids)
    }

    # Find intersection of cell_ids between graph and features
    common_cell_ids: List[int] = []
    graph_indices: List[int] = []
    feature_indices: List[int] = []

    for cell_id, feature_idx in cell_indices.items():
        if cell_id in cell_id_to_graph_idx:
            common_cell_ids.append(cell_id)
            graph_indices.append(cell_id_to_graph_idx[cell_id])
            feature_indices.append(feature_idx)

    if not common_cell_ids:
        raise ValueError("No common cell_ids found between graph and features")

    logger.info(f"Found {len(common_cell_ids)} common cells between graph and features")

    graph_indices_tensor = torch.tensor(graph_indices, dtype=torch.long)
    feature_indices_tensor = torch.tensor(feature_indices, dtype=torch.long)

    # Create subgraph by filtering edges
    subgraph_edge_mask = torch.isin(edge_indices[0], graph_indices_tensor) & torch.isin(
        edge_indices[1], graph_indices_tensor
    )

    if not subgraph_edge_mask.any():
        logger.warning("No edges found in subgraph - creating isolated nodes")
        # Create empty edge_index for isolated nodes
        remapped_edges = torch.empty((2, 0), dtype=torch.long)
        subgraph_edge_attr = None
    else:
        # Remap edge indices to new node ordering
        old_to_new_idx = {
            old_idx.item(): new_idx
            for new_idx, old_idx in enumerate(graph_indices_tensor)
        }

        subgraph_edges = edge_indices[:, subgraph_edge_mask]
        remapped_edges = torch.zeros_like(subgraph_edges)

        for i in range(subgraph_edges.shape[1]):
            src_old = subgraph_edges[0, i].item()
            dst_old = subgraph_edges[1, i].item()
            remapped_edges[0, i] = old_to_new_idx[src_old]
            remapped_edges[1, i] = old_to_new_idx[dst_old]

        # Get edge attributes for subgraph
        subgraph_edge_attr = None
        if edge_features is not None:
            subgraph_edge_attr = edge_features[subgraph_edge_mask]

    # Select features for the common cells in the correct order
    selected_features = features[feature_indices_tensor]

    # Prepare centroids if available
    pos = cell_coordinates[feature_indices_tensor]

    # Create final merged Data object
    merged_data = Data(
        x=selected_features,
        edge_index=remapped_edges,
        edge_attr=subgraph_edge_attr,
        pos=pos,
        num_nodes=len(common_cell_ids),
    )

    # Store cell_ids for reference
    merged_data.cell_ids = torch.tensor(common_cell_ids, dtype=torch.long)

    return merged_data


def cell_type_name_to_index(cell_type_names: List[str]) -> List[int]:
    """
    Convert cell type names to their corresponding indices.

    Args:
        cell_type_names: List of cell type names (case-insensitive)

    Returns:
        List of cell type indices (1-based, as used in TYPE_NUCLEI_DICT)

    Raises:
        ValueError: If any cell type name is invalid
    """
    # Create a case-insensitive lookup dictionary
    name_to_index = {name.lower(): idx for idx, name in TYPE_NUCLEI_DICT.items()}

    indices: list[int] = []
    for name in cell_type_names:
        name_lower = name.lower()
        if name_lower not in name_to_index:
            valid_names = list(TYPE_NUCLEI_DICT.values())
            raise ValueError(
                f"Invalid cell type name: '{name}'. "
                f"Valid names are: {valid_names} (case-insensitive)"
            )
        indices.append(name_to_index[name_lower])

    return indices


def load_roi_for_slide(
    slide_name: str, roi_folder: Path, metadata: pd.DataFrame
) -> Optional[pd.DataFrame]:
    """
    Load ROI data for a specific slide.

    Args:
        slide_name: Name of the slide (DIG_PAT_XXXXXXXX format)
        roi_folder: Path to directory containing ROI CSV files
        metadata: DataFrame containing 'ID', 'I3LUNG_ID', and 'CENTER' columns

    Returns:
        DataFrame with ROI coordinates or None if not found
    """
    try:
        # Find the slide in metadata
        slide_row = metadata[metadata["ID"] == slide_name]
        if slide_row.empty:
            raise ValueError(f"Slide {slide_name} not found in metadata")

        i3lung_id = cast(str, slide_row["I3LUNG_ID"].values[0])  # type: ignore
        center = cast(str, slide_row["CENTER"].values[0])  # type: ignore

        # Map center to folder name
        center_to_folder: Dict[str, Union[List[str], str]] = {
            "GHD": "GHD_RoI_auto",
            "INT": "INT_RoI_auto",
            "MH": "MH_RoI_auto",
            "SZMC": ["SZMC_RoI_auto", "SZMC-unzipped_RoI_auto"],
            "UOC": "UOC_RoI_auto",
            "VHIO": "VHIO_RoI_auto",
        }

        folder = center_to_folder.get(center)
        if folder is None:
            raise ValueError(f"Unknown CENTER value '{center}' for slide {slide_name}")

        # Try to find the ROI file
        roi_path = None
        if isinstance(folder, list):
            for f in folder:
                potential_path = roi_folder / f / f"{i3lung_id}.csv"
                if potential_path.exists():
                    roi_path = potential_path
                    break
        else:
            potential_path = roi_folder / folder / f"{i3lung_id}.csv"
            if potential_path.exists():
                roi_path = potential_path

        if roi_path is None:
            raise FileNotFoundError(f"ROI file not found for slide {slide_name}")

        # Load ROI data
        roi_df = pd.read_csv(roi_path)  # type: ignore
        logger.debug(f"Loaded ROI for slide {slide_name}: {len(roi_df)} points")
        return roi_df

    except Exception as e:
        logger.error(f"Error loading ROI for slide {slide_name}: {e}")
        return None


def filter_cells_by_roi(
    centroids: Dict[int, Tuple[float, float]], roi_df: pd.DataFrame
) -> set[int]:
    """
    Filter cells to keep only those within ROI boundaries.

    Args:
        centroids: Dictionary mapping cell_id to (x, y) centroid coordinates
        roi_df: DataFrame with ROI coordinates (columns: roi_name, label, x_base, y_base)

    Returns:
        Set of cell IDs that are within the ROI boundaries
    """
    # Create polygons for each ROI
    roi_polygons: list[Polygon] = []
    for roi_name in roi_df["roi_name"].unique():  # type: ignore
        roi_points = cast(
            np.ndarray[Any, Any],
            roi_df[roi_df["roi_name"] == roi_name][["x_base", "y_base"]].values,  # type: ignore
        )
        if len(roi_points) >= 3:  # Need at least 3 points to form a polygon
            roi_polygons.append(Polygon(roi_points))

    if len(roi_polygons) == 0:
        logger.warning("No valid ROI polygons found")
        return set()

    # Merge all ROI polygons into a single geometry
    roi_union = unary_union(roi_polygons)

    # Filter cells by checking if centroid is within ROI
    cells_to_keep: set[int] = set()
    for cell_id, (x, y) in centroids.items():
        point = Point(x, y)
        if point.within(roi_union):
            cells_to_keep.add(cell_id)

    logger.debug(f"ROI filtering: kept {len(cells_to_keep)}/{len(centroids)} cells")
    return cells_to_keep

import pandas as pd
import torch
import json
import hashlib
from pathlib import Path
from typing import Tuple, List, Dict, Union, Literal, Any
from cellmil.utils import logger
from cellmil.datamodels.datasets.cell_gnn_mil_dataset import CellGNNMILDataset
from cellmil.datamodels.datasets.utils import column_sanity_check, filter_split, extract_slide_name
from torch_geometric.data import Data # type: ignore


def create_processed_dataset_files(
    root: Union[str, Path],
    label: str,
    pyg_datasets: List[CellGNNMILDataset],
    data: pd.DataFrame,
    split: Literal["train", "val", "test"] = "train",
    force_reload: bool = False,
) -> str:
    """
    Create processed dataset files directly compatible with GNNMILDataset.
    
    This function reuses existing processed datasets but with different labels,
    and creates the processed files exactly as GNNMILDataset would create them.
    After running this function, you can use GNNMILDataset normally with the new label.
    
    Args:
        root: Root directory where the processed dataset files will be saved
        label: New label column name for classification
        pyg_datasets: List of existing GNNMILDatasets [train, val, test]
        data: DataFrame containing metadata with the new labels
        split: Dataset split to create (train/val/test)
        force_reload: Whether to force reprocessing even if processed files exist
        
    Returns:
        Path to the created processed file
    """
    
    # Perform sanity check on the dataframe
    column_sanity_check(data, label)
    
    # Filter by split type
    split_data = filter_split(data, split)
    
    if len(split_data) == 0:
        raise ValueError(f"No data found for split '{split}'")
    
    # Get reference dataset for configuration
    ref_dataset = pyg_datasets[0]
    
    # Create the same hash that GNNMILDataset would create
    extractor_str = json.dumps(ref_dataset.extractor, sort_keys=True)
    
    config_dict: Dict[str, Any] = {
        'label': label,  # Use the NEW label here
        'extractor': extractor_str,
        'graph_creator': ref_dataset.graph_creator,
        'segmentation_model': ref_dataset.segmentation_model,
        'split': split,
        'cell_type': ref_dataset.cell_type,
        'centroid': ref_dataset.centroid,
        'correlation_filter_enabled': ref_dataset.correlation_filter_enabled,
        'correlation_threshold': ref_dataset.correlation_threshold,
        'normalize_feature': ref_dataset.normalize_feature,
    }
    
    # Create the same hash as GNNMILDataset
    config_str = json.dumps(config_dict, sort_keys=True)
    config_hash = hashlib.md5(config_str.encode('utf-8')).hexdigest()[:8]
    
    # Create root directory structure
    root_path = Path(root)
    processed_dir = root_path / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Create the processed file name exactly as GNNMILDataset would
    processed_file_name = f'data_{split}_{config_hash}.pt'
    processed_file_path = processed_dir / processed_file_name
    
    # Check if file already exists and we don't want to reload
    if processed_file_path.exists() and not force_reload:
        logger.info(f"Processed file already exists: {processed_file_path}")
        return str(processed_file_path)
    
    # Create a mapping from slide names to datasets
    dataset_slide_mapping: Dict[str, Tuple[str, Any]] = {}
    
    for dataset in pyg_datasets:
        logger.info(f"Mapping slides from {dataset.split} dataset with {len(dataset)} samples")

        for data_obj in dataset:
            if hasattr(data_obj, 'slide_name'):
                slide_name = data_obj.slide_name
                dataset_slide_mapping[slide_name] = (dataset.split, data_obj)

    logger.info(f"Total slides mapped: {len(dataset_slide_mapping)}")
    
    # Process the data and create the dataset
    logger.info(f"Processing reused dataset for {split} split with label '{label}'")
    
    data_list: List[Data] = []
    successful_matches = 0
    missing_slides: List[str] = []
    
    for _, row in split_data.iterrows():
        slide_name = ""
        try:
            # Extract slide name
            file_path = Path(str(row["FULL_PATH"]))
            slide_name = extract_slide_name(file_path)
            
            # Get new label
            new_label = int(row[label])
            
            # Find corresponding data in existing datasets
            if slide_name in dataset_slide_mapping:
                source_split, source_data = dataset_slide_mapping[slide_name]
                
                # Create new data object with same features but new label
                new_data = Data(
                    x=source_data.x.clone() if source_data.x is not None else torch.zeros(1, 1),  # Reuse node features
                    y=torch.tensor([new_label], dtype=torch.long),  # New label
                    num_nodes=source_data.num_nodes if hasattr(source_data, 'num_nodes') else (source_data.x.shape[0] if source_data.x is not None else 1),
                )
                
                # Copy optional attributes if they exist
                if hasattr(source_data, 'pos') and source_data.pos is not None:
                    new_data.pos = source_data.pos.clone()
                if hasattr(source_data, 'edge_index') and source_data.edge_index is not None:
                    new_data.edge_index = source_data.edge_index.clone()
                if hasattr(source_data, 'edge_attr') and source_data.edge_attr is not None:
                    new_data.edge_attr = source_data.edge_attr.clone()
                if hasattr(source_data, 'cell_ids'):
                    new_data.cell_ids = source_data.cell_ids.clone()
                if hasattr(source_data, 'metadata'):
                    new_data.metadata = source_data.metadata
                
                # Store slide name for reference
                new_data.slide_name = slide_name
                
                data_list.append(new_data)
                successful_matches += 1
                
                logger.debug(f"Matched slide {slide_name} from {source_split} dataset with new label {new_label}")
            else:
                missing_slides.append(slide_name)
                logger.warning(f"Slide {slide_name} not found in any existing dataset")
                
        except Exception as e:
            logger.error(f"Error processing row for slide {slide_name}: {e}")
            continue
    
    logger.info(f"Successfully matched {successful_matches} slides for {split} split")
    if missing_slides:
        logger.warning(f"Missing {len(missing_slides)} slides: {missing_slides[:10]}{'...' if len(missing_slides) > 10 else ''}")
    
    if not data_list:
        raise ValueError(f"No valid data found for {split} split")
    
    # Save the processed data in the exact same format as InMemoryDataset.save()
    torch.save(data_list, processed_file_path)
    logger.info(f"Saved processed dataset to {processed_file_path}")
    logger.info(f"Created dataset for {split} split with {len(data_list)} samples using label '{label}'")
    logger.info(f"Configuration hash: {config_hash}")
    
    return str(processed_file_path)
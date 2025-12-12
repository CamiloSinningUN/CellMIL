import logging
import argparse
import sys
import traceback
from pathlib import Path
from cellmil.interfaces import DatasetCreatorConfig
from cellmil.dataset import DatasetCreator

# Setup logging with enhanced format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

def create_dataset(args: argparse.Namespace) -> None:
    """Create a dataset for MIL training based on the provided configuration."""
    try:
        # Create configuration for patch extraction
        config = DatasetCreatorConfig(
            excel_path=args.excel_path,
            output_path=args.output_path,
            gpu=args.gpu,
            segmentation_models=args.segmentation_models,
            extractors=args.extractors,
            graph_methods=args.graph_methods
        )
        
        dataset_creator = DatasetCreator(config)
        dataset_creator.create()

        logger.info("Finished creating dataset.")
    except Exception as e:
        # Get traceback information
        tb = traceback.format_exc()
        logger.error(f"Error during dataset creation: {e}\n{tb}")
        sys.exit(1)

def setup_parser() -> argparse.ArgumentParser:
    """Set up command line argument parser"""
    parser = argparse.ArgumentParser(
        description="CLI tool for creating a dataset for MIL training"
    )

    # Create dataset arguments
    parser.add_argument(
        "--excel_path",
        type=Path,
        required=True,
        help="Path to the Excel file containing metadata information",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        required=True,
        help="Path where the dataset will be saved",
    )
    parser.add_argument(
        "--gpu", type=int, default=0, help="GPU index to use for processing (default: 0)"
    )
    parser.add_argument(
        "--segmentation_models",
        nargs='+',
        required=False,
        help="List of segmentation models to use for cell segmentation",
    )
    parser.add_argument(
        "--extractors",
        nargs='+',
        required=True,
        help="List of feature extractors to use for feature extraction",
    )
    
    parser.add_argument(
        "--graph_methods",
        nargs='+',
        required=False,
        help="List of graph creation methods to use",
    )

    return parser

def main() -> None:
    """Main function to run the CLI tool"""
    parser = setup_parser()
    args = parser.parse_args()

    create_dataset(args)
    
if __name__ == "__main__":
    main()
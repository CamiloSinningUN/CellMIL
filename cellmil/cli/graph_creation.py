import logging
import argparse
import sys
import traceback
from pathlib import Path
from cellmil.interfaces import GraphCreatorConfig
from cellmil.graph import GraphCreator
from cellmil.interfaces.GraphCreatorConfig import GraphCreatorType
from cellmil.interfaces.CellSegmenterConfig import ModelType

# Setup logging with enhanced format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

def graph_creation(args: argparse.Namespace) -> None:
    """Create graph from whole slide images with specified configuration."""
    try:
        # Create configuration
        config = GraphCreatorConfig(
            method=args.method,
            patched_slide_path=args.patched_slide_path,
            segmentation_model=args.segmentation_model,
            gpu=args.gpu,
            plot=True,
            debug=args.debug
        )

        # Process slide
        slide_processor = GraphCreator(config)
        slide_processor.create_graph()

        if not args.debug:
            logger.info("Finished Graph Creation.")
        else:
            logger.info("Debug visualizer closed.")
    except Exception as e:
        # Get traceback information
        tb = traceback.format_exc()
        logger.error(f"Error during graph creation: {e}\n{tb}")
        sys.exit(1)


def setup_parser() -> argparse.ArgumentParser:
    """Set up command line argument parser"""
    parser = argparse.ArgumentParser(
        description="CLI tool for preparing data from whole slide images"
    )

    # Graph creation arguments
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=GraphCreatorType.values(),
        help="Graph creation method to use, options are: " + ", ".join(GraphCreatorType.values()),
    )
    
    parser.add_argument(
        "--patched_slide_path",
        type=Path,
        required=True,
        help="Path to the patched slide image folder",
    )
    
    parser.add_argument(
        "--segmentation_model",
        type=str,
        required=True,
        choices=ModelType.values(),
        help="Segmentation model used to extract cells",
    )

    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU device ID to use for processing",
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with interactive visualization for hyperparameter tuning",
    )

    return parser


def main():
    """Entry point for the CLI tool."""
    parser = setup_parser()
    args = parser.parse_args()

    # Call graph creation directly
    graph_creation(args)


if __name__ == "__main__":
    main()
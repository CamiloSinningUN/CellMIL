import logging
import argparse
import sys
import traceback
from pathlib import Path
from cellmil.interfaces import FeatureExtractorConfig
from cellmil.features import FeatureExtractor
from cellmil.interfaces.FeatureExtractorConfig import ExtractorType
from cellmil.interfaces.CellSegmenterConfig import ModelType

# Setup logging with enhanced format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

def feature_extraction(args: argparse.Namespace) -> None:
    """Extract patches from whole slide images with specified configuration."""
    try:
        # Create configuration
        config = FeatureExtractorConfig(
            extractor=args.extractor,
            wsi_path=args.wsi_path,
            patched_slide_path=args.patched_slide_path,
            segmentation_model=args.segmentation_model,
            graph_method=args.graph_method,
        )

        # Process slide
        slide_processor = FeatureExtractor(config)
        slide_processor.get_features()

        logger.info("Finished Preprocessing.")
    except Exception as e:
        # Get traceback information
        tb = traceback.format_exc()
        logger.error(f"Error during preprocessing: {e}\n{tb}")
        sys.exit(1)


def setup_parser() -> argparse.ArgumentParser:
    """Set up command line argument parser"""
    parser = argparse.ArgumentParser(
        description="CLI tool for preparing data from whole slide images"
    )

    # Patch extraction arguments
    parser.add_argument(
        "--extractor",
        type=str,
        required=True,
        choices=ExtractorType.values(),
        help="Feature extractor to use, options are: " + ", ".join(ExtractorType.values()),
    )
    
    parser.add_argument(
        "--wsi_path", 
        type=Path, 
        required=False, 
        default=None,
        help="Path to the whole slide image"
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
        required=False,
        choices=ModelType.values(),
        default=None,
        help="Segmentation model used to extract cells",
    )

    parser.add_argument(
        "--graph_method",
        type=str,
        required=False,
        default=None,
        help="Graph method to use for feature extraction",
    )

    return parser


def main():
    """Entry point for the CLI tool."""
    parser = setup_parser()
    args = parser.parse_args()

    # Call patch extraction directly
    feature_extraction(args)


if __name__ == "__main__":
    main()
import logging
import argparse
import sys
import traceback
from pathlib import Path
from cellmil.data import PatchExtractor
from cellmil.interfaces import PatchExtractorConfig

# Setup logging with enhanced format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def patch_extraction(args: argparse.Namespace) -> None:
    """Extract patches from whole slide images with specified configuration."""
    try:
        # Create configuration
        config = PatchExtractorConfig(
            output_path=args.output_path,
            wsi_path=args.wsi_path,
            patch_size=args.patch_size,
            patch_overlap=args.patch_overlap,
            target_mag=args.target_mag,
        )

        # Process slide
        slide_processor = PatchExtractor(config)
        slide_processor.get_patches()

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
        "--output_path",
        type=Path,
        required=True,
        help="Path where output files will be saved",
    )
    parser.add_argument(
        "--wsi_path", type=Path, required=True, help="Path to whole slide image file"
    )
    parser.add_argument(
        "--patch_size", type=int, required=True, help="Patch size in pixels"
    )
    parser.add_argument(
        "--patch_overlap",
        type=float,
        required=True,
        help="Overlap between adjacent patches (0.0-100.0)",
    )
    parser.add_argument(
        "--target_mag", type=float, required=True, help="Target magnification"
    )

    return parser


def main():
    """Entry point for the CLI tool."""
    parser = setup_parser()
    args = parser.parse_args()

    # Call patch extraction directly
    patch_extraction(args)


if __name__ == "__main__":
    main()

import logging
import argparse
import sys
import traceback
from pathlib import Path
from cellmil.interfaces import FeatureVisualizerConfig
from cellmil.visualization import FeatureVisualizer

# Setup logging with enhanced format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)
def visualize_features(args: argparse.Namespace) -> None:
    """Visualize features from a patched slide."""
    try:
        # Create configuration
        config = FeatureVisualizerConfig(
            dataset=args.dataset,
        )

        # Initialize feature visualizer
        feature_visualizer = FeatureVisualizer(config)

        # Perform visualization
        feature_visualizer.visualize()

        logger.info("Finished feature visualization.")
    except Exception as e:
        # Get traceback information
        tb = traceback.format_exc()
        logger.error(f"Error during feature visualization: {e}\n{tb}")
        sys.exit(1)


def setup_parser() -> argparse.ArgumentParser:
    """Set up command line argument parser"""
    parser = argparse.ArgumentParser(
        description="CLI tool for visualizing features from a dataset folder with extracted features"
    )
    
    # Path to patched slide
    parser.add_argument(
        "--dataset", type=Path, required=True, help="Path to the folder with "
    )

    return parser

def main():
    """Entry point for the CLI tool."""
    parser = setup_parser()
    args = parser.parse_args()

    # Run MIL prediction
    visualize_features(args)
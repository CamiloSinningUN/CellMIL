import logging
import argparse
import sys
import traceback
from cellmil.statistics import StatsPrinter
from cellmil.interfaces.StatsPrinterConfig import StatsPrinterConfig

# Setup logging with enhanced format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def stats_print(args: argparse.Namespace) -> None:
    """Create statistics report."""
    try:
        config = StatsPrinterConfig(
            team=args.team,
            projects=args.projects
        )

        stats_printer = StatsPrinter(config)
        stats_printer.create(args.metric)

        logger.info(f"Statistics report created successfully for team {args.team} and projects {args.projects}.")
    except Exception as e:
        # Get traceback information
        tb = traceback.format_exc()
        logger.error(f"Error during statistics report creation: {e}\n{tb}")
        sys.exit(1)


def setup_parser() -> argparse.ArgumentParser:
    """Set up command line argument parser"""
    parser = argparse.ArgumentParser(
        description="Generate statistics report from wandb data"
    )
    
    parser.add_argument(
        "--metric",
        type=str,
        required=True,
        help="Metric to generate statistics for",
    )
    
    parser.add_argument(
        "--team",
        type=str,
        required=True,
        help="Team name where the project belongs in wandb",
    )
    
    parser.add_argument(
        "--projects",
        type=str,
        nargs='+',
        required=True,
        help="Project/s name where data is allocated in wandb",
    )

    return parser


def main():
    """Entry point for the cell segmentation CLI tool."""
    parser = setup_parser()
    args = parser.parse_args()

    stats_print(args)


if __name__ == "__main__":
    main()

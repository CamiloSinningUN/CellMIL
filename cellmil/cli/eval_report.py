import logging
import argparse
import sys
import traceback
from pathlib import Path
from cellmil.evaluation import EvaluationReporter
from cellmil.interfaces.EvaluationReporterConfig import EvaluationReporterConfig

# Setup logging with enhanced format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def eval_report(args: argparse.Namespace) -> None:
    """Create statistics report."""
    try:
        config = EvaluationReporterConfig(
            metrics=args.metrics,
            team=args.team,
            projects=args.projects,
            output_dir=args.output_dir
        )

        reporter = EvaluationReporter(config)
        reporter.create()

        logger.info(f"Evaluation report created successfully for team {args.team} and projects {args.projects}.")
    except Exception as e:
        # Get traceback information
        tb = traceback.format_exc()
        logger.error(f"Error during evaluation report creation: {e}\n{tb}")
        sys.exit(1)


def setup_parser() -> argparse.ArgumentParser:
    """Set up command line argument parser"""
    parser = argparse.ArgumentParser(
        description="Generate statistics report from wandb data"
    )
    
    parser.add_argument(
        "--metrics",
        nargs='+',
        required=True,
        help="Metrics to generate evaluation reports for",
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
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default="./evaluation_reports",
        help="Directory to save evaluation reports",
    )

    return parser


def main():
    """Entry point for the cell segmentation CLI tool."""
    parser = setup_parser()
    args = parser.parse_args()

    eval_report(args)


if __name__ == "__main__":
    main()

import logging
import argparse
import sys
import traceback
from pathlib import Path
from cellmil.evaluation import ExternalValidator
from cellmil.interfaces.EvaluationExternalValidatorConfig import (
    EvaluationExternalValidatorConfig,
    FinalModel,
    AggregationMethod,
)

# Setup logging with enhanced format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

def eval_external(args: argparse.Namespace) -> None:
    """Perform external validation using a pre-trained model."""
    try:
        config = EvaluationExternalValidatorConfig(
            metrics=args.metrics,
            output_dir=args.output_dir,
            models_dir=args.models_dir,
            final_model=args.final_model,
            aggregation_method=args.aggregation_method,
            dataset_dir=args.dataset_dir,
            root_dir=args.root_dir,
            dp_metadata_file=args.dp_metadata_file,
        )

        validator = ExternalValidator(config)
        validator.validate()

        logger.info(f"External validation completed successfully using models at {args.models_dir}.")
    except Exception as e:
        # Get traceback information
        tb = traceback.format_exc()
        logger.error(f"Error during external validation: {e}\n{tb}")
        sys.exit(1)

def setup_parser() -> argparse.ArgumentParser:
    """Set up command line argument parser"""
    parser = argparse.ArgumentParser(
        description="Perform external validation using a pre-trained model"
    )
    
    parser.add_argument(
        "--metrics",
        nargs='+',
        required=True,
        help="Metrics to generate evaluation reports for",
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        default=Path("./external_validation_reports"),
        help="Directory to save evaluation reports",
    )
    
    parser.add_argument(
        "--models-dir",
        type=Path,
        required=True,
        help="Directory containing pre-trained models for external validation",
    )
    
    parser.add_argument(
        "--final-model",
        type=str,
        choices=FinalModel.values(),
        default=FinalModel.final,
        help="Whether to use the final model or an ensemble for validation",
    )
    
    parser.add_argument(
        "--aggregation-method",
        type=str,
        choices=AggregationMethod.values(),
        default=AggregationMethod.mean,
        help="Method to aggregate predictions from multiple models in ensemble",
    )
    
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Directory containing datasets for external validation",
    )
    
    parser.add_argument(
        "--root-dir",
        type=Path,
        required=True,
        help="Root directory for caching datasets",
    )
    
    parser.add_argument(
        "--dp-metadata-file",
        type=Path,
        required=True,
        help="Path to the metadata Excel file for datasets",
    )
    
    return parser

def main():
    parser = setup_parser()
    args = parser.parse_args()

    eval_external(args)

if __name__ == "__main__":
    main()
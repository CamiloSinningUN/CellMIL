import ujson as json
import pandas as pd
import wandb
from tqdm import tqdm
from pathlib import Path
import openslide
from cellmil.interfaces import DatasetCreatorConfig
from cellmil.utils import logger
from cellmil.data import PatchExtractor
from cellmil.interfaces import PatchExtractorConfig
from cellmil.segmentation import CellSegmenter
from cellmil.interfaces import CellSegmenterConfig
from cellmil.interfaces.CellSegmenterConfig import ModelType
from cellmil.interfaces import FeatureExtractorConfig
from cellmil.features import FeatureExtractor
from cellmil.graph import GraphCreator
from cellmil.interfaces.FeatureExtractorConfig import (
    ExtractorType,
    FeatureExtractionType,
)
from cellmil.interfaces import GraphCreatorConfig
from cellmil.interfaces.GraphCreatorConfig import GraphCreatorType, GraphCreatorCategory

# TODO: ADAPT FOR NEW GRAPH CREATION


class DatasetCreator:
    """Class to create a dataset for MIL training based on the provided configuration."""

    def __init__(self, config: DatasetCreatorConfig):
        self.config = config

        # TODO: Make this configurable
        self.patch_size = 256
        self.patch_overlap = 6.25
        self.wsl = False
        # TODO ---
        self.test = False

        self._check_config()
        self._setup_wandb()
        self._read_excel()

    def _check_config(self):
        """Check the configuration for any issues."""
        has_morphology = False
        has_topology = False

        for extractor in self.config.extractors:
            if extractor in FeatureExtractionType.Morphological:
                has_morphology = True
            elif extractor in FeatureExtractionType.Topological:
                has_topology = True

        if has_morphology or has_topology:
            if not self.config.segmentation_models:
                raise ValueError(
                    "Morphological or topological feature extraction requires at least one segmentation model."
                )

        if has_topology:
            if not self.config.graph_methods:
                raise ValueError(
                    "Topological feature extraction requires at least one graph creation method."
                )

    def _setup_wandb(self):
        # Initialize wandb
        config: dict[str, str | int | list[str] | float] = {
            "output_path": str(self.config.output_path),
            "excel_path": str(self.config.excel_path),
            "test": self.test,
            "extractors": [str(extractor) for extractor in self.config.extractors],
            "patch_size": self.patch_size,
            "patch_overlap": self.patch_overlap,
            "gpu": self.config.gpu,
        }

        if self.config.graph_methods:
            config["graph_methods"] = [
                str(method) for method in self.config.graph_methods
            ]

        if self.config.segmentation_models:
            config["segmentation_models"] = [
                str(model) for model in self.config.segmentation_models
            ]

        wandb.login()
        wandb.init(
            project="cellmil-dataset", name="dataset_creation_run", config=config
        )
        logger.info("Wandb initialized for tracking")

    def _read_excel(self):
        """Read the Excel file specified in the configuration."""
        try:
            self.slides = pd.read_excel(self.config.excel_path)  # type: ignore
            self._preprocess_excel()

            logger.info(f"Metadata loaded from {self.config.excel_path}")

            if self.test:
                self.slides = self.slides.head(5)
                logger.info("Running in test mode, limiting to 5 slides.")
        except Exception as e:
            raise ValueError(f"Failed to read Excel file: {e}")

    def _preprocess_excel(self):
        """Preprocess the Excel data to ensure paths are correctly formatted."""
        if self.wsl:
            self.slides["FULL_PATH"] = self.slides["FULL_PATH"].apply(  # type: ignore
                lambda path: path.replace("\\", "/").replace("D:", "/mnt/d")  # type: ignore
            )

        if "MAGNIFICATION" not in self.slides.columns:
            logger.info(
                "Magnification column not found. Extracting magnification from slide metadata..."
            )
            magnifications: list[float | int | None] = []
            for _, row in tqdm(
                self.slides.iterrows(),
                total=len(self.slides),
                desc="Extracting magnifications",
            ):  # type: ignore
                wsi_path = Path(row["FULL_PATH"])  # type: ignore
                mag = self._get_magnification_openslide(str(wsi_path))
                magnifications.append(mag)
            self.slides["MAGNIFICATION"] = magnifications
            logger.info("Magnification extraction completed.")

    def _get_magnification_openslide(self, path: str) -> float | None:
        """Extract magnification from slide metadata using openslide."""
        try:
            slide = openslide.OpenSlide(path)
            mag = slide.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER)
            if mag is not None:
                slide.close()
                return float(mag)
            mpp_x = slide.properties.get(openslide.PROPERTY_NAME_MPP_X)
            if mpp_x is not None:
                mag = round(10 / float(mpp_x), 1)
                slide.close()
                return mag
            for prop_name in slide.properties:
                if (
                    "MAGNIFICATION" in prop_name.upper()
                    or "OBJECTIVE" in prop_name.upper()
                ):
                    try:
                        slide.close()
                        return float(slide.properties[prop_name])
                    except (ValueError, TypeError):
                        continue

            logger.warning(f"Could not extract magnification for {path}")
            slide.close()
            return None

        except Exception as e:
            logger.error(f"Error extracting magnification from {path}: {e}")
            return None

    def _load_progress(self) -> dict[str, list[str]]:
        """Load previous progress from log.json if it exists."""
        log_path = self.config.output_path / "log.json"
        processed: dict[str, list[str]] = {
            "processed_patch_extraction": [],
            "processed_cell_segmentation": [],
            "processed_graph_creation": [],
            "processed_feature_extraction": [],
        }

        if log_path.exists():
            try:
                with open(log_path, "r") as f:
                    loaded_data = json.load(f)
                    processed.update(loaded_data)
                logger.info(f"Loaded previous progress from {log_path}")
                logger.info(
                    f"Found {len(processed['processed_patch_extraction'])} extracted patches, "
                    f"{len(processed['processed_cell_segmentation'])} segmented slides, "
                    f"{len(processed['processed_graph_creation'])} slides with created graphs, "
                    f"{len(processed['processed_feature_extraction'])} slides with extracted features."
                )

            except Exception as e:
                logger.warning(
                    f"Failed to load progress from log.json: {e}. Starting from scratch."
                )
        else:
            # If log.json doesn't exist but output folders do, try to infer progress
            logger.info(
                "No previous log.json found. Checking for existing output folders..."
            )
            try:
                for item in self.config.output_path.iterdir():
                    if item.is_dir() and item.name not in [
                        "__pycache__",
                        ".ipynb_checkpoints",
                    ]:
                        wsi_name = item.name
                        processed["processed_patch_extraction"].append(wsi_name)

                        # Check for cell segmentation
                        if self.config.segmentation_models:
                            for model in self.config.segmentation_models:
                                seg_folder = (
                                    self.config.output_path
                                    / wsi_name
                                    / "cell_detection"
                                    / model
                                )
                                if seg_folder.exists():
                                    processed["processed_cell_segmentation"].append(
                                        f"{wsi_name}_{model}"
                                    )
                                    logger.info(
                                        f"Found existing segmentation folder for {wsi_name} ({model}), assuming cells were segmented"
                                    )

                        # Check for graph creation
                        if (
                            self.config.graph_methods
                            and self.config.segmentation_models
                        ):
                            for graph_method in self.config.graph_methods:
                                for model in self.config.segmentation_models:
                                    graph_folder = (
                                        self.config.output_path
                                        / wsi_name
                                        / "graphs"
                                        / str(graph_method)
                                        / str(model)
                                    )
                                    if graph_folder.exists():
                                        slide_graph_key = (
                                            f"{wsi_name}_{model}_{graph_method}"
                                        )
                                        processed["processed_graph_creation"].append(
                                            slide_graph_key
                                        )
                                        logger.info(
                                            f"Found existing graph folder for {wsi_name} ({graph_method}), assuming graphs were created"
                                        )

                        # Check for feature extraction

                        # Embedding features
                        for extractor in self.config.extractors:
                            feature_folder = (
                                self.config.output_path
                                / wsi_name
                                / "feature_extraction"
                                / str(extractor)
                                / "features.pt"
                            )
                            if feature_folder.exists():
                                slide_model_extractor_key = f"{wsi_name}_{extractor}"
                                processed["processed_feature_extraction"].append(
                                    slide_model_extractor_key
                                )
                                logger.info(
                                    f"Found existing feature folder for {wsi_name} ({extractor}), assuming features were extracted"
                                )

                        # Morphological features
                        if self.config.segmentation_models:
                            for extractor in self.config.extractors:
                                for model in self.config.segmentation_models:
                                    feature_folder = (
                                        self.config.output_path
                                        / wsi_name
                                        / "feature_extraction"
                                        / str(extractor)
                                        / str(model)
                                        / "features.pt"
                                    )
                                    if feature_folder.exists():
                                        slide_model_extractor_key = (
                                            f"{wsi_name}_{model}_{extractor}"
                                        )
                                        processed[
                                            "processed_feature_extraction"
                                        ].append(slide_model_extractor_key)
                                        logger.info(
                                            f"Found existing feature folder for {wsi_name} ({model}, {extractor}), assuming features were extracted"
                                        )

                        # Topological features
                        if (
                            self.config.graph_methods
                            and self.config.segmentation_models
                        ):
                            for extractor in self.config.extractors:
                                for method in self.config.graph_methods:
                                    for model in self.config.segmentation_models:
                                        feature_folder = (
                                            self.config.output_path
                                            / wsi_name
                                            / "feature_extraction"
                                            / str(extractor)
                                            / str(method)
                                            / str(model)
                                            / "features.pt"
                                        )
                                        if feature_folder.exists():
                                            slide_model_extractor_key = f"{wsi_name}_{model}_{method}_{extractor}"
                                            processed[
                                                "processed_feature_extraction"
                                            ].append(slide_model_extractor_key)
                                            logger.info(
                                                f"Found existing feature folder for {wsi_name} ({model}, {extractor}, {method}), assuming features were extracted"
                                            )

                        logger.info(
                            f"Found existing folder for {wsi_name}, assuming patches were extracted"
                        )

                # Save the inferred progress to log.json
                with open(self.config.output_path / "log.json", "w") as f:
                    json.dump(processed, f, indent=4)
                logger.info(
                    "Created log.json with inferred progress from existing folders"
                )
            except Exception as e:
                logger.warning(f"Error checking output folders: {e}")

        return processed

    def _extract_patches(self, wsi_path: Path, magnification: int):
        config = PatchExtractorConfig(
            output_path=self.config.output_path,
            wsi_path=wsi_path,
            patch_size=self.patch_size,
            patch_overlap=self.patch_overlap,
            target_mag=magnification,
        )
        # Process slide
        slide_processor = PatchExtractor(config)
        slide_processor.get_patches()

    def _setup_segment_cells(
        self, wsi_path: Path, patched_slide_path: Path, model: ModelType
    ):
        config = CellSegmenterConfig(
            model=model,
            gpu=self.config.gpu,
            wsi_path=wsi_path,
            patched_slide_path=patched_slide_path,
        )

        cell_segmenter = CellSegmenter(config)
        return cell_segmenter

    def _use_segment_cells(
        self, cell_segmenter: CellSegmenter, wsi_path: Path, patched_slide_path: Path
    ):
        cell_segmenter.set_wsi(wsi_path, patched_slide_path)
        cell_segmenter.process()

    def _create_graph(
        self,
        patched_slide_path: Path,
        graph_method: GraphCreatorType,
        segmentation_model: ModelType,
    ):
        config = GraphCreatorConfig(
            method=graph_method,
            gpu=self.config.gpu,
            patched_slide_path=patched_slide_path,
            segmentation_model=segmentation_model,
            plot=False,
        )

        graph_creator = GraphCreator(config)
        graph_creator.create_graph()

    def _extract_features(
        self,
        patched_slide_path: Path,
        extractor: ExtractorType,
        wsi_path: Path | None = None,
        graph_method: GraphCreatorType | None = None,
        segmentation_model: ModelType | None = None,
    ):
        """Extract features from the segmented cells."""
        # Create configuration
        config = FeatureExtractorConfig(
            extractor=extractor,
            wsi_path=wsi_path,
            patched_slide_path=patched_slide_path,
            graph_method=graph_method,
            segmentation_model=segmentation_model,
        )

        # Process slide
        slide_processor = FeatureExtractor(config)
        slide_processor.get_features()

    def create(self):
        """Create the dataset based on the configuration."""

        logger.info("Starting dataset creation process...")

        # Check for existing progress
        processed = self._load_progress()

        # Make sure output directory exists
        self.config.output_path.mkdir(parents=True, exist_ok=True)

        # --- Extract patches ---
        logger.info("1/4 Starting patch extraction...")
        wandb.log({"status": "Patch extraction", "progress": 0})

        total_slides = len(self.slides)
        for idx, (_, row) in enumerate(  # type: ignore
            tqdm(self.slides.iterrows(), total=total_slides, desc="Patch extraction")  # type: ignore
        ):
            try:
                wsi_path = Path(row["FULL_PATH"])  # type: ignore
                wsi_name = wsi_path.stem

                # Skip if already processed
                if wsi_name in processed["processed_patch_extraction"]:
                    logger.info(
                        f"Slide {wsi_name} already has patches extracted. Skipping."
                    )
                    continue

                if not wsi_path.exists():
                    warning_msg = (
                        f"Slide {wsi_name} does not exist at {wsi_path}. Skipping."
                    )
                    logger.warning(warning_msg)
                    wandb.log({"warning": warning_msg})
                    continue

                magnification = row["MAGNIFICATION"]  # type: ignore

                self._extract_patches(wsi_path, magnification)  # type: ignore
                processed["processed_patch_extraction"].append(wsi_name)
                logger.info(f"Extracted patches for slide {wsi_name}")

                # Save progress after each slide to prevent data loss
                with open(self.config.output_path / "log.json", "w") as f:
                    json.dump(processed, f, indent=4)

                progress = (idx + 1) / total_slides
                wandb.log(
                    {
                        "patch_extraction_progress": progress * 100,
                        "patches_extracted": len(
                            processed["processed_patch_extraction"]
                        ),
                        "current_slide": wsi_name,
                    }
                )
            except Exception as e:
                error_msg = f"Error processing slide {row['FULL_PATH']}: {e}"
                logger.error(error_msg)
                wandb.log({"error": error_msg})
                continue
        # --- End of patch extraction ---

        # --- Cell segmentation ---
        if self.config.segmentation_models:
            logger.info("2/4 Starting cell segmentation...")
            wandb.log({"status": "Cell segmentation", "progress": 25})

            for model_idx, model in enumerate(self.config.segmentation_models):
                logger.info(f"Starting cell segmentation with model: {model}")
                wandb.log({"current_segmentation_model": str(model)})

                for idx, (_, row) in enumerate(  # type: ignore
                    tqdm(  # type: ignore
                        self.slides.iterrows(),  # type: ignore
                        total=total_slides,  # type: ignore
                        desc=f"Cell segmentation with {model}",  # type: ignore
                    )
                ):  # type: ignore
                    try:
                        wsi_path = Path(row["FULL_PATH"])  # type: ignore
                        wsi_name = wsi_path.stem
                        model_name = str(model)
                        # Create a unique key for this slide+model combination
                        slide_model_key = f"{wsi_name}_{model_name}"

                        # Skip if already processed with this model
                        if slide_model_key in processed["processed_cell_segmentation"]:
                            logger.info(
                                f"Slide {wsi_name} already segmented with model {model_name}. Skipping."
                            )
                            continue

                        if wsi_name not in processed["processed_patch_extraction"]:
                            skip_msg = f"Slide {wsi_name} has not been processed by patch extraction yet. Skipping segmentation."
                            logger.info(skip_msg)
                            wandb.log({"info": skip_msg})
                            continue

                        patched_slide_path = self.config.output_path / wsi_name

                        # Create and use a new segmenter for each slide
                        cell_segmenter = self._setup_segment_cells(
                            wsi_path, patched_slide_path, model
                        )
                        self._use_segment_cells(
                            cell_segmenter, wsi_path, patched_slide_path
                        )

                        processed["processed_cell_segmentation"].append(slide_model_key)
                        logger.info(
                            f"Segmented cells for slide {wsi_name} with model {model_name}"
                        )

                        # Save progress after each slide to prevent data loss
                        with open(self.config.output_path / "log.json", "w") as f:
                            json.dump(processed, f, indent=4)

                        model_progress = (idx + 1) / total_slides
                        overall_progress = 25 + (
                            25
                            * (
                                (model_idx + model_progress)
                                / len(self.config.segmentation_models)
                            )
                        )
                        wandb.log(
                            {
                                "cell_segmentation_progress": model_progress * 100,
                                "overall_progress": overall_progress,
                                "cells_segmented": len(
                                    processed["processed_cell_segmentation"]
                                ),
                                "current_slide": wsi_name,
                            }
                        )
                    except Exception as e:
                        error_msg = (
                            f"Error segmenting cells for slide {row['FULL_PATH']}: {e}"
                        )
                        logger.error(error_msg)
                        wandb.log({"error": error_msg})
                        continue
            else:
                logger.info(
                    "Cell segmentation already processed for some slides. Skipping cell segmentation."
                )
        # --- End of cell segmentation ---

        # --- Morphological Feature extraction (needed for feature-dependent graphs) ---
        morphological_extractors = [
            extractor
            for extractor in self.config.extractors
            if extractor in FeatureExtractionType.Morphological
        ]

        if morphological_extractors:
            logger.info("3/7 Starting morphological feature extraction...")
            wandb.log({"status": "Morphological feature extraction", "progress": 35})

            for extractor in morphological_extractors:
                if (
                    self.config.segmentation_models is None
                    or not self.config.segmentation_models
                ):
                    logger.info(
                        f"No segmentation models available for extractor: {extractor}. Skipping."
                    )
                    continue

                for model in self.config.segmentation_models:
                    logger.info(
                        f"Using extractor: {extractor} with segmentation model: {model}"
                    )
                    wandb.log(
                        {
                            "current_extractor": str(extractor),
                            "current_model_for_extraction": str(model),
                        }
                    )

                    for idx, (_, row) in enumerate(  # type: ignore
                        tqdm(  # type: ignore
                            self.slides.iterrows(),  # type: ignore
                            total=total_slides,  # type: ignore
                            desc=f"Morphological feature extraction with {extractor}",  # type: ignore
                        )
                    ):  # type: ignore
                        try:
                            wsi_path = Path(row["FULL_PATH"])  # type: ignore
                            wsi_name = wsi_path.stem
                            model_name = str(model)
                            extractor_name = str(extractor)

                            # Create a unique key for this slide+model+extractor combination
                            slide_model_extractor_key = (
                                f"{wsi_name}_{model_name}_{extractor_name}"
                            )

                            # Skip if already processed with this model and extractor
                            if (
                                slide_model_extractor_key
                                in processed["processed_feature_extraction"]
                            ):
                                logger.info(
                                    f"Features already extracted for slide {wsi_name} using model {model_name} and extractor {extractor_name}. Skipping."
                                )
                                continue

                            # Check for cell segmentation
                            slide_model_key = f"{wsi_name}_{model_name}"
                            if (
                                slide_model_key
                                not in processed["processed_cell_segmentation"]
                            ):
                                skip_msg = f"Slide {wsi_name} has not been processed by cell segmentation with model {model_name} yet. Skipping feature extraction."
                                logger.info(skip_msg)
                                wandb.log({"info": skip_msg})
                                continue

                            patched_slide_path = self.config.output_path / wsi_name
                            self._extract_features(
                                patched_slide_path,
                                extractor,
                                wsi_path=wsi_path,
                                segmentation_model=model,
                            )
                            processed["processed_feature_extraction"].append(
                                slide_model_extractor_key
                            )
                            logger.info(
                                f"Extracted morphological features for slide {wsi_name} using model {model_name} and extractor {extractor_name}"
                            )

                            # Save progress after each extraction to prevent data loss
                            with open(self.config.output_path / "log.json", "w") as f:
                                json.dump(processed, f, indent=4)

                            slide_progress = (idx + 1) / total_slides
                            wandb.log(
                                {
                                    "morphological_extraction_progress": slide_progress
                                    * 100,
                                    "features_extracted": len(
                                        processed["processed_feature_extraction"]
                                    ),
                                    "current_slide": wsi_name,
                                }
                            )
                        except Exception as e:
                            error_msg = f"Error processing slide {row['FULL_PATH']} for morphological feature extraction: {e}"
                            logger.error(error_msg)
                            wandb.log({"error": error_msg})
                            continue
        # --- End of morphological feature extraction ---

        # --- Positional Graph creation (graphs that don't require features) ---
        positional_graph_methods = (
            [
                method
                for method in self.config.graph_methods
                if method in GraphCreatorCategory.Positional
            ]
            if self.config.graph_methods
            else []
        )

        if positional_graph_methods and self.config.segmentation_models:
            logger.info("4/7 Starting positional graph creation...")
            wandb.log({"status": "Positional graph creation", "progress": 50})

            total_methods = len(positional_graph_methods)
            total_models = len(self.config.segmentation_models)

            for method_idx, method in enumerate(positional_graph_methods):
                for model_idx, model in enumerate(self.config.segmentation_models):
                    logger.info(
                        f"Using positional graph method: {method} with segmentation model: {model}"
                    )
                    wandb.log(
                        {
                            "current_graph_method": str(method),
                            "current_model_for_graph": str(model),
                        }
                    )

                    for idx, (_, row) in enumerate(  # type: ignore
                        tqdm(  # type: ignore
                            self.slides.iterrows(),  # type: ignore
                            total=total_slides,  # type: ignore
                            desc=f"Positional graph creation with {method}",  # type: ignore
                        )
                    ):  # type: ignore
                        try:
                            wsi_path = Path(row["FULL_PATH"])  # type: ignore
                            wsi_name = wsi_path.stem
                            model_name = str(model)
                            method_name = str(method)

                            # Create a unique key for this slide+model+method combination
                            slide_model_method_key = (
                                f"{wsi_name}_{model_name}_{method_name}"
                            )

                            # Skip if already processed with this model and method
                            if (
                                slide_model_method_key
                                in processed["processed_graph_creation"]
                            ):
                                logger.info(
                                    f"Positional graph already created for slide {wsi_name} using model {model_name} and method {method_name}. Skipping."
                                )
                                continue

                            # Check for cell segmentation
                            slide_model_key = f"{wsi_name}_{model_name}"
                            if (
                                slide_model_key
                                not in processed["processed_cell_segmentation"]
                            ):
                                skip_msg = f"Slide {wsi_name} has not been processed by cell segmentation with model {model_name} yet. Skipping graph creation."
                                logger.info(skip_msg)
                                wandb.log({"info": skip_msg})
                                continue

                            patched_slide_path = self.config.output_path / wsi_name
                            self._create_graph(patched_slide_path, method, model)

                            processed["processed_graph_creation"].append(
                                slide_model_method_key
                            )
                            logger.info(
                                f"Created positional graph for slide {wsi_name} with model {model_name} and method {method_name}"
                            )

                            # Save progress after each slide to prevent data loss
                            with open(self.config.output_path / "log.json", "w") as f:
                                json.dump(processed, f, indent=4)

                            slide_progress = (idx + 1) / total_slides
                            combo_progress = (
                                (method_idx * total_models) + model_idx + slide_progress
                            ) / (total_methods * total_models)

                            wandb.log(
                                {
                                    "positional_graph_creation_progress": slide_progress
                                    * 100,
                                    "graphs_created": len(
                                        processed["processed_graph_creation"]
                                    ),
                                    "current_slide": wsi_name,
                                }
                            )
                        except Exception as e:
                            error_msg = f"Error creating positional graph for slide {row['FULL_PATH']}: {e}"
                            logger.error(error_msg)
                            wandb.log({"error": error_msg})
                            continue
        # --- End of positional graph creation ---

        # --- Feature-dependent Graph creation (graphs that require morphological features) ---
        feature_dependent_graph_methods = (
            [
                method
                for method in self.config.graph_methods
                if method in GraphCreatorCategory.FeatureDependent
            ]
            if self.config.graph_methods
            else []
        )

        if feature_dependent_graph_methods and self.config.segmentation_models:
            logger.info("5/7 Starting feature-dependent graph creation...")
            wandb.log({"status": "Feature-dependent graph creation", "progress": 60})

            total_methods = len(feature_dependent_graph_methods)
            total_models = len(self.config.segmentation_models)

            for method_idx, method in enumerate(feature_dependent_graph_methods):
                for model_idx, model in enumerate(self.config.segmentation_models):
                    logger.info(
                        f"Using feature-dependent graph method: {method} with segmentation model: {model}"
                    )
                    wandb.log(
                        {
                            "current_graph_method": str(method),
                            "current_model_for_graph": str(model),
                        }
                    )

                    for idx, (_, row) in enumerate(  # type: ignore
                        tqdm(  # type: ignore
                            self.slides.iterrows(),  # type: ignore
                            total=total_slides,  # type: ignore
                            desc=f"Feature-dependent graph creation with {method}",  # type: ignore
                        )
                    ):  # type: ignore
                        try:
                            wsi_path = Path(row["FULL_PATH"])  # type: ignore
                            wsi_name = wsi_path.stem
                            model_name = str(model)
                            method_name = str(method)

                            # Create a unique key for this slide+model+method combination
                            slide_model_method_key = (
                                f"{wsi_name}_{model_name}_{method_name}"
                            )

                            # Skip if already processed with this model and method
                            if (
                                slide_model_method_key
                                in processed["processed_graph_creation"]
                            ):
                                logger.info(
                                    f"Feature-dependent graph already created for slide {wsi_name} using model {model_name} and method {method_name}. Skipping."
                                )
                                continue

                            # Check for cell segmentation
                            slide_model_key = f"{wsi_name}_{model_name}"
                            if (
                                slide_model_key
                                not in processed["processed_cell_segmentation"]
                            ):
                                skip_msg = f"Slide {wsi_name} has not been processed by cell segmentation with model {model_name} yet. Skipping graph creation."
                                logger.info(skip_msg)
                                wandb.log({"info": skip_msg})
                                continue

                            # Check for morphological features (required for feature-dependent graphs)
                            if morphological_extractors:
                                has_required_features = False
                                for morph_extractor in morphological_extractors:
                                    morph_key = f"{wsi_name}_{model_name}_{str(morph_extractor)}"
                                    if (
                                        morph_key
                                        in processed["processed_feature_extraction"]
                                    ):
                                        has_required_features = True
                                        break

                                if not has_required_features:
                                    skip_msg = f"Slide {wsi_name} does not have required morphological features for model {model_name}. Skipping feature-dependent graph creation."
                                    logger.info(skip_msg)
                                    wandb.log({"info": skip_msg})
                                    continue

                            patched_slide_path = self.config.output_path / wsi_name
                            self._create_graph(patched_slide_path, method, model)

                            processed["processed_graph_creation"].append(
                                slide_model_method_key
                            )
                            logger.info(
                                f"Created feature-dependent graph for slide {wsi_name} with model {model_name} and method {method_name}"
                            )

                            # Save progress after each slide to prevent data loss
                            with open(self.config.output_path / "log.json", "w") as f:
                                json.dump(processed, f, indent=4)

                            slide_progress = (idx + 1) / total_slides
                            combo_progress = (
                                (method_idx * total_models) + model_idx + slide_progress
                            ) / (total_methods * total_models)
                            overall_progress = 75 + (25 * combo_progress)

                            wandb.log(
                                {
                                    "graph_creation_progress": slide_progress * 100,
                                    "overall_progress": overall_progress,
                                    "graphs_created": len(
                                        processed["processed_graph_creation"]
                                    ),
                                    "current_slide": wsi_name,
                                }
                            )
                        except Exception as e:
                            error_msg = f"Error creating feature-dependent graph for slide {row['FULL_PATH']}: {e}"
                            logger.error(error_msg)
                            wandb.log({"error": error_msg})
                            continue
        # --- End of feature-dependent graph creation ---

        # --- Embedding Feature extraction ---
        embedding_extractors = [
            extractor
            for extractor in self.config.extractors
            if extractor in FeatureExtractionType.Embedding
        ]

        if embedding_extractors:
            logger.info("6/7 Starting embedding feature extraction...")
            wandb.log({"status": "Embedding feature extraction", "progress": 75})

            # Embedding Features
            for extractor in embedding_extractors:
                logger.info(f"Using extractor: {extractor}")
                wandb.log(
                    {
                        "current_extractor": str(extractor),
                    }
                )

                for idx, (_, row) in enumerate(  # type: ignore
                    tqdm(  # type: ignore
                        self.slides.iterrows(),  # type: ignore
                        total=total_slides,  # type: ignore
                        desc=f"Embedding feature extraction with {extractor}",  # type: ignore
                    )
                ):  # type: ignore
                    try:
                        wsi_path = Path(row["FULL_PATH"])  # type: ignore
                        wsi_name = wsi_path.stem
                        extractor_name = str(extractor)

                        # Create a unique key for this slide+model+extractor combination
                        slide_model_extractor_key = f"{wsi_name}_{extractor_name}"

                        # Skip if already processed with this model and extractor
                        if (
                            slide_model_extractor_key
                            in processed["processed_feature_extraction"]
                        ):
                            logger.info(
                                f"Embedding features already extracted for slide {wsi_name} using extractor {extractor_name}. Skipping."
                            )
                            continue

                        patched_slide_path = self.config.output_path / wsi_name
                        self._extract_features(patched_slide_path, extractor, wsi_path)
                        processed["processed_feature_extraction"].append(
                            slide_model_extractor_key
                        )
                        logger.info(
                            f"Extracted embedding features for slide {wsi_name} using extractor {extractor_name}"
                        )

                        # Save progress after each extraction to prevent data loss
                        with open(self.config.output_path / "log.json", "w") as f:
                            json.dump(processed, f, indent=4)

                        slide_progress = (idx + 1) / total_slides
                        wandb.log(
                            {
                                "embedding_extraction_progress": slide_progress * 100,
                                "features_extracted": len(
                                    processed["processed_feature_extraction"]
                                ),
                                "current_slide": wsi_name,
                            }
                        )
                    except Exception as e:
                        error_msg = f"Error processing slide {row['FULL_PATH']} for embedding feature extraction: {e}"
                        logger.error(error_msg)
                        wandb.log({"error": error_msg})
                        continue
        # --- End of embedding feature extraction ---

        # --- Topological Feature extraction (must be done after all graph creations) ---
        topological_extractors = [
            extractor
            for extractor in self.config.extractors
            if extractor in FeatureExtractionType.Topological
        ]

        if topological_extractors:
            logger.info("7/7 Starting topological feature extraction...")
            wandb.log({"status": "Topological feature extraction", "progress": 85})

            # Topological Features
            for extractor in topological_extractors:
                if (
                    self.config.segmentation_models is None
                    or not self.config.segmentation_models
                ):
                    logger.info(
                        f"No segmentation models available for extractor: {extractor}. Skipping."
                    )
                    continue

                for model in self.config.segmentation_models:
                    if (
                        self.config.graph_methods is None
                        or not self.config.graph_methods
                    ):
                        logger.info(
                            f"No graph methods available for extractor: {extractor}. Skipping."
                        )
                        continue

                    for graph_method in self.config.graph_methods:
                        logger.info(
                            f"Using extractor: {extractor} with segmentation model: {model} and graph method: {graph_method}"
                        )
                        wandb.log(
                            {
                                "current_extractor": str(extractor),
                                "current_model_for_extraction": str(model),
                                "current_graph_method": str(graph_method),
                            }
                        )

                        for idx, (_, row) in enumerate(  # type: ignore
                            tqdm(  # type: ignore
                                self.slides.iterrows(),  # type: ignore
                                total=total_slides,  # type: ignore
                                desc=f"Topological feature extraction with {extractor}",  # type: ignore
                            )
                        ):  # type: ignore
                            try:
                                wsi_path = Path(row["FULL_PATH"])  # type: ignore
                                wsi_name = wsi_path.stem
                                model_name = str(model)
                                extractor_name = str(extractor)

                                # Create a unique key for this slide+model+extractor combination
                                slide_model_extractor_key = f"{wsi_name}_{model_name}_{graph_method}_{extractor_name}"

                                # Skip if already processed with this model and extractor
                                if (
                                    slide_model_extractor_key
                                    in processed["processed_feature_extraction"]
                                ):
                                    logger.info(
                                        f"Topological features already extracted for slide {wsi_name} using model {model_name}, extractor {extractor_name}, graph method {graph_method}. Skipping."
                                    )
                                    continue

                                # Check for cell segmentation
                                slide_model_key = f"{wsi_name}_{model_name}"
                                if (
                                    slide_model_key
                                    not in processed["processed_cell_segmentation"]
                                ):
                                    skip_msg = f"Slide {wsi_name} has not been processed by cell segmentation with model {model_name} yet. Skipping topological feature extraction."
                                    logger.info(skip_msg)
                                    wandb.log({"info": skip_msg})
                                    continue

                                # Check for graph creation
                                slide_model_graph_key = (
                                    f"{wsi_name}_{model_name}_{graph_method}"
                                )
                                if (
                                    slide_model_graph_key
                                    not in processed["processed_graph_creation"]
                                ):
                                    skip_msg = f"Slide {wsi_name} has not been processed by graph creation with model {model_name} and method {graph_method} yet. Skipping topological feature extraction."
                                    logger.info(skip_msg)
                                    wandb.log({"info": skip_msg})
                                    continue

                                patched_slide_path = self.config.output_path / wsi_name
                                self._extract_features(
                                    patched_slide_path,
                                    extractor,
                                    wsi_path=wsi_path,
                                    segmentation_model=model,
                                    graph_method=graph_method,
                                )
                                processed["processed_feature_extraction"].append(
                                    slide_model_extractor_key
                                )
                                logger.info(
                                    f"Extracted topological features for slide {wsi_name} using model {model_name}, extractor {extractor_name}, and graph method {graph_method}"
                                )

                                # Save progress after each extraction to prevent data loss
                                with open(
                                    self.config.output_path / "log.json", "w"
                                ) as f:
                                    json.dump(processed, f, indent=4)

                                slide_progress = (idx + 1) / total_slides
                                wandb.log(
                                    {
                                        "topological_extraction_progress": slide_progress
                                        * 100,
                                        "features_extracted": len(
                                            processed["processed_feature_extraction"]
                                        ),
                                        "current_slide": wsi_name,
                                    }
                                )
                            except Exception as e:
                                error_msg = f"Error processing slide {row['FULL_PATH']} for topological feature extraction: {e}"
                                logger.error(error_msg)
                                wandb.log({"error": error_msg})
                                continue
        # --- End of topological feature extraction ---

        # --- End of all feature extraction ---

        logger.info("Dataset creation completed successfully.")

        # Save processed processed data
        with open(self.config.output_path / "log.json", "w") as f:
            json.dump(processed, f, indent=4)

        logger.info(
            f"Processed data log saved to {self.config.output_path / 'log.json'}"
        )

        wandb.log(
            {
                "status": "Completed",
                "progress": 100.0,
                "patches_extracted": len(processed["processed_patch_extraction"]),
                "cells_segmented": len(processed["processed_cell_segmentation"]),
                "features_extracted": len(processed["processed_feature_extraction"]),
            }
        )
        # Upload log.json as an artifact
        log_artifact = wandb.Artifact("dataset_log", type="log")
        log_artifact.add_file(str(self.config.output_path / "log.json"))
        wandb.log_artifact(log_artifact)
        wandb.finish()

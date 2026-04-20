import logging
from pathlib import Path
from typing import List, Union, Optional, Dict

import npec_cv._tf_setup  # noqa: F401

import cv2
import pandas as pd
from tqdm.auto import tqdm

from npec_cv.configs.inference import ROOT_MODEL_CONFIG, ANCHOR_MODEL_CONFIG
from npec_cv.configs.segmentation import SEGMENTATION_CONFIG
from npec_cv.configs.lengths import LENGTH_CALCULATOR_CONFIG
from npec_cv.configs.thresholds import ADAPTIVE_THRESHOLD_CONFIG
from npec_cv.core.plant_master import PlantMaster
from npec_cv.helpers import extract_sort_keys
from npec_cv.models import PlantMeasurement, ImageResult
from npec_cv.exceptions import InitializationError

# Configure logging
logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Keep pipeline logger at DEBUG


class CVPipeline:
    """Computer vision pipeline for plant root detection and measurement.

    This pipeline processes plant images to detect roots, segment individual
    plants, and calculate root lengths using a unified PlantMaster component.
    """

    def __init__(
        self,
        inference_config: Optional[tuple] = None,
        segmentation_config=None,
        calculator_config=None,
        thresholds_config=None,
    ):
        """Initialize the pipeline with PlantMaster component.

        Args:
            inference_config: Tuple of (root_model_config, anchor_model_config).
                Defaults to module-level configs if not provided.
            segmentation_config: Configuration for plant segmentation.
            calculator_config: Configuration for length calculation.
            thresholds_config: Configuration for adaptive thresholding.

        Raises:
            InitializationError: If PlantMaster fails to initialize.
        """
        self._img_files: List[str] = []
        self.measurements: List[PlantMeasurement] = []
        self.image_results: Dict[str, ImageResult] = {}

        # Use defaults if not provided
        inference_config = inference_config or (ROOT_MODEL_CONFIG, ANCHOR_MODEL_CONFIG)
        segmentation_config = segmentation_config or SEGMENTATION_CONFIG
        calculator_config = calculator_config or LENGTH_CALCULATOR_CONFIG
        thresholds_config = thresholds_config or ADAPTIVE_THRESHOLD_CONFIG

        self.plant_master = self._init_plant_master(
            inference_config,
            segmentation_config,
            calculator_config,
            thresholds_config,
        )

    def _init_plant_master(
        self,
        inference_config: tuple,
        segmentation_config,
        calculator_config,
        thresholds_config,
    ) -> PlantMaster:
        """Initialize the PlantMaster component.

        Args:
            inference_config: Tuple of model configurations.
            segmentation_config: Segmentation parameters.
            calculator_config: Length calculation parameters.
            thresholds_config: Adaptive threshold parameters.

        Returns:
            Initialized PlantMaster instance.

        Raises:
            InitializationError: If PlantMaster fails to initialize.
        """
        try:
            plant_master = PlantMaster(
                inference_config=inference_config,
                segmentation_config=segmentation_config,
                calculator_config=calculator_config,
                thresholds_config=thresholds_config,
            )
            logger.info(f"PlantMaster initialized for {plant_master.num_plants} plants")
            return plant_master
        except Exception as e:
            logger.exception("Failed to initialize PlantMaster")
            raise InitializationError(f"PlantMaster initialization failed: {e}") from e

    def _load_images(self, input_dir: Union[Path, str]) -> None:
        """Load list of image files from the specified directory.

        Args:
            input_dir: Path to directory containing image files.
        """
        input_path = Path(input_dir)
        valid_extensions = {".png", ".jpg", ".jpeg"}

        self._img_files = sorted(
            [
                f.name
                for f in input_path.iterdir()
                if f.suffix.lower() in valid_extensions
            ]
        )
        logger.info(f"Found {len(self._img_files)} image files")

    def _collect_results(self, result: ImageResult, img_name: str) -> None:
        """Collect plant measurements and store full results."""
        # Store full result for visualization
        self.image_results[img_name] = result

        # Collect measurements for CSV
        for plant_id, plant_result in result.plant_results.items():
            if plant_result.detected:
                logger.debug(
                    f"  Plant {plant_id}: detected, length={plant_result.length}px"
                )
            else:
                logger.debug(f"  Plant {plant_id}: not detected")

            self.measurements.append(
                PlantMeasurement(
                    image_name=img_name,
                    plant_id=plant_id,
                    length=plant_result.length if plant_result.detected else 0,
                )
            )

    def _save_results_csv(self, output_dir: Path) -> Path:
        """Export results to a sorted CSV file.

        Args:
            output_dir: Directory where the CSV file will be saved.

        Returns:
            Path to the created CSV file.
        """
        csv_rows = [
            {"Plant ID": m.plant_label, "Length (px)": m.length}
            for m in self.measurements
        ]

        df = pd.DataFrame(csv_rows)
        df["sort_key"] = df["Plant ID"].apply(extract_sort_keys)
        df = (
            df.sort_values("sort_key").drop(columns=["sort_key"]).reset_index(drop=True)
        )

        csv_path = output_dir / "plant_lengths.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Results saved to {csv_path}")

        return csv_path

    def process_directory(
        self,
        input_dir: Union[Path, str],
        max_images: Optional[int] = None,
    ) -> Path:
        """Process all images in a directory through the full pipeline.

        Args:
            input_dir: Path to folder containing plant images.
            max_images: Optional limit on number of images to process.

        Returns:
            Path to the generated CSV results file.

        Raises:
            ValueError: If directory doesn't exist or contains no images.
        """
        input_path = Path(input_dir)

        if not input_path.exists():
            raise ValueError(f"Input directory does not exist: {input_dir}")

        if not self._img_files:
            self._load_images(input_path)

        if not self._img_files:
            raise ValueError(f"No valid images found in {input_dir}")

        # Apply image limit if specified
        img_files = self._img_files
        if max_images:
            img_files = img_files[:max_images]

        logger.info(f"Processing {len(img_files)} images")

        # Disable tqdm when in debug mode to avoid cluttering log output
        disable_tqdm = logger.getEffectiveLevel() <= logging.DEBUG

        for img_filename in tqdm(
            img_files, desc="Processing images", unit="img", disable=disable_tqdm
        ):
            img_path = input_path / img_filename
            img_name = img_path.stem

            # Load image
            img = cv2.imread(str(img_path))
            if img is None:
                logger.warning(f"Failed to read image: {img_path}")
                continue

            logger.debug(f"Processing: {img_name}")

            # Process with PlantMaster
            try:
                result = self.plant_master.process_image(img, img_name)
                logger.debug(
                    f"  {img_name}: {result.num_detected}/{result.total_plants} detected"
                )
                self._collect_results(result, img_name)
            except Exception as e:
                logger.error(f"Failed to process {img_name}: {e}")
                continue

        # Save results
        output_dir = input_path.parent
        return self._save_results_csv(output_dir)

    def process_single_image(self, img_path: Union[Path, str]):
        """Process a single image and return the result.

        Args:
            img_path: Path to the image file.

        Returns:
            Processing result from PlantMaster.

        Raises:
            ValueError: If image cannot be loaded.
        """
        img_path = Path(img_path)
        img = cv2.imread(str(img_path))

        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")

        return self.plant_master.process_image(img, img_path.stem)


def cv_pipeline(
    input_dir: Union[Path, str],
    max_images: Optional[int] = None,
) -> Path:
    """Convenience function to run the full CV pipeline.

    Args:
        input_dir: Path to folder containing plant images.
        max_images: Optional limit on number of images to process.

    Returns:
        Path to the generated CSV results file.

    Raises:
        InitializationError: If pipeline components fail to initialize.
        ValueError: If the directory is invalid or contains no images.
    """
    pipeline = CVPipeline()
    return pipeline.process_directory(input_dir, max_images=max_images)


if __name__ == "__main__":
    # Configure logging for CLI usage
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Run pipeline from current directory as a sensible CLI default
    result_path = cv_pipeline(Path.cwd())
    print(f"Results saved to: {result_path}")

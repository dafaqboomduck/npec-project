"""NPEC CV command-line interface.

Three commands:
  predict  — run inference on a plant image or directory of images
  info     — print model metadata and pipeline configuration
  version  — print the installed package version
"""

from __future__ import annotations

import json
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as pkg_version
from pathlib import Path
from typing import Optional

import typer
from typing_extensions import Annotated

# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

try:
    import numpy as np

    class _NumpyEncoder(json.JSONEncoder):
        """Serialise numpy scalars and arrays to native Python types."""

        def default(self, o):
            if isinstance(o, np.integer):
                return int(o)
            if isinstance(o, np.floating):
                return float(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            return super().default(o)

    def _dumps(obj: object) -> str:
        return json.dumps(obj, indent=2, cls=_NumpyEncoder)

except ImportError:

    def _dumps(obj: object) -> str:  # type: ignore[misc]
        return json.dumps(obj, indent=2)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_FILE_SIZE_BYTES: int = 50 * 1024 * 1024  # 50 MB
MIN_RESOLUTION: int = 512
MAX_RESOLUTION: int = 5000
ACCEPTED_CONTENT_TYPES: frozenset[str] = frozenset(["image/png", "image/tiff", "image/jpg", "image/jpeg"])
ACCEPTED_EXTENSIONS: frozenset[str] = frozenset([".png", ".tiff", ".tif", ".jpg", ".jpeg"])

# Keep old name as alias so existing internal references don't break
SUPPORTED_EXTENSIONS = ACCEPTED_EXTENSIONS

# ---------------------------------------------------------------------------
# Typer app
# ---------------------------------------------------------------------------

app = typer.Typer(
    name="npec-cv",
    help=(
        "NPEC plant root analysis CLI.\n"
        "\n"
        "Analyse petri-dish plant images from the terminal — detect roots,\n"
        "measure their lengths, and export results as JSON without writing\n"
        "any Python code.\n"
        "\n"
        "Quick start:\n"
        "\n"
        "  pip install ./npec_cv\n"
        "  npec-cv predict plant.jpg\n"
        "  npec-cv predict images/\n"
        "\n"
        "Run 'npec-cv <command> --help' for full options on any command."
    ),
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode="markdown",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_version() -> str:
    """Return the installed npec_cv package version."""
    try:
        return pkg_version("npec_cv")
    except PackageNotFoundError:
        return "unknown"


def _image_result_to_dict(result) -> dict:
    """Convert an ImageResult dataclass into a JSON-serialisable dict.

    Output schema
    -------------
    {
      "image": str,
      "num_detected": int,
      "total_plants": int,
      "plants": {
        "<plant_id>": {
          "detected": bool,
          "length_px": int | null,
          "tip_location": [row, col] | null,
          "confidence_score": float | null,
          "pixel_count": int | null,
          "mask_available": bool
        }
      }
    }
    """
    plants: dict = {}
    for plant_id, pr in result.plant_results.items():
        plants[str(plant_id)] = {
            "detected": pr.detected,
            "length_px": pr.length,
            "tip_location": list(pr.end_point) if pr.end_point is not None else None,
            "confidence_score": pr.threshold_used,
            "pixel_count": pr.pixel_count,
            "mask_available": pr.mask is not None,
        }
    return {
        "image": result.image_name,
        "num_detected": result.num_detected,
        "total_plants": result.total_plants,
        "plants": plants,
    }


def _build_pipeline(confidence: float):
    """Construct a CVPipeline with a custom root-detection threshold.

    Heavy imports are deferred here so that `npec-cv version` and
    `npec-cv info` start fast without loading TensorFlow.
    """
    from npec_cv import CVPipeline  # noqa: PLC0415
    from npec_cv.configs.inference import (  # noqa: PLC0415
        ANCHOR_MODEL_CONFIG,
        ROOT_MODEL_CONFIG,
    )
    from npec_cv.models.inference import ModelConfig  # noqa: PLC0415

    root_cfg = ModelConfig(
        checkpoint=ROOT_MODEL_CONFIG.checkpoint,
        threshold=confidence,
        patch_size=ROOT_MODEL_CONFIG.patch_size,
        batch_size=ROOT_MODEL_CONFIG.batch_size,
        custom_objects=ROOT_MODEL_CONFIG.custom_objects,
    )
    return CVPipeline(inference_config=(root_cfg, ANCHOR_MODEL_CONFIG))


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_image(path: Path) -> None:
    """Run all image constraints against a single file.

    Raises typer.Exit(1) with a descriptive error message if any check fails.

    Checks (in order):
      1. Extension is in ACCEPTED_EXTENSIONS
      2. File size is within MAX_FILE_SIZE_BYTES
      3. Image can be opened and resolution is within [MIN_RESOLUTION, MAX_RESOLUTION]
    """
    # --- extension ---------------------------------------------------------
    if path.suffix.lower() not in ACCEPTED_EXTENSIONS:
        typer.echo(
            f"Error: unsupported file format '{path.suffix}'. "
            f"Accepted formats: {', '.join(sorted(ACCEPTED_EXTENSIONS))}",
            err=True,
        )
        raise typer.Exit(code=1)

    # --- file size ---------------------------------------------------------
    size = path.stat().st_size
    if size > MAX_FILE_SIZE_BYTES:
        mb = size / (1024 * 1024)
        limit_mb = MAX_FILE_SIZE_BYTES / (1024 * 1024)
        typer.echo(
            f"Error: '{path.name}' is {mb:.1f} MB which exceeds the "
            f"{limit_mb:.0f} MB limit. Please reduce the file size or split the image.",
            err=True,
        )
        raise typer.Exit(code=1)

    # --- resolution --------------------------------------------------------
    try:
        from PIL import Image as _PILImage  # noqa: PLC0415

        with _PILImage.open(path) as img:
            w, h = img.size
    except Exception as exc:
        typer.echo(
            f"Error: could not read image '{path.name}': {exc}",
            err=True,
        )
        raise typer.Exit(code=1)

    if w < MIN_RESOLUTION or h < MIN_RESOLUTION:
        typer.echo(
            f"Error: '{path.name}' is {w}x{h} px — both dimensions must be "
            f"at least {MIN_RESOLUTION} px. "
            "The model requires a minimum patch size of 512 px.",
            err=True,
        )
        raise typer.Exit(code=1)

    if w > MAX_RESOLUTION or h > MAX_RESOLUTION:
        typer.echo(
            f"Error: '{path.name}' is {w}x{h} px — dimensions must not exceed "
            f"{MAX_RESOLUTION} px. Please downscale the image before running inference.",
            err=True,
        )
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.command()
def predict(
    path: Annotated[
        Path,
        typer.Argument(
            help=(
                "Image file or directory of images. "
                f"Accepted formats: {', '.join(sorted(ACCEPTED_EXTENSIONS))}. "
                f"Max file size: {MAX_FILE_SIZE_BYTES // (1024 * 1024)} MB. "
                f"Resolution: {MIN_RESOLUTION}–{MAX_RESOLUTION} px per side. "
                "Directories are processed as a batch and output becomes a JSON array."
            ),
        ),
    ],
    confidence: Annotated[
        float,
        typer.Option(
            "--confidence",
            "-c",
            help=(
                "Root-detection probability threshold applied to the model output (0–1). "
                "The pipeline starts at this value and lowers it adaptively until every "
                "plant zone is detected. "
                "Default **0.01** works for most images; raise it (e.g. 0.3) to suppress "
                "false positives, lower it for faint or young roots."
            ),
            min=0.0,
            max=1.0,
        ),
    ] = 0.01,
    output: Annotated[
        Optional[Path],
        typer.Option(
            "--output",
            "-o",
            help=(
                "Write the JSON results to this file instead of printing to stdout. "
                "Useful for saving results without shell redirection."
            ),
        ),
    ] = None,
) -> None:
    """Run plant-root inference on a single image or a directory of images.

    **Output format**

    Single image → JSON object. Directory → JSON array (one object per image).

    Each object contains:

    - `image` — filename stem
    - `num_detected` / `total_plants` — detection summary
    - `plants` — per-plant dict with:
        - `detected` — whether a root was found
        - `length_px` — geodesic root length in pixels
        - `tip_location` — [row, col] pixel coordinate of the root tip
        - `confidence_score` — adaptive threshold at which the root was detected
        - `pixel_count` — total pixels in the root mask
        - `mask_available` — whether a binary mask was produced

    Examples:
      # Single image → prints JSON to stdout
      npec-cv predict plant.jpg

      # Batch process a directory
      npec-cv predict images/

      # Lower threshold for faint roots, save to file
      npec-cv predict images/ --confidence 0.005 --output results.json

      # Raise threshold to reduce false positives
      npec-cv predict plant.jpg -c 0.3
    """
    if not path.exists():
        typer.echo(f"Error: path does not exist: {path}", err=True)
        raise typer.Exit(code=1)

    if path.is_file():
        _validate_image(path)
        pipeline = _build_pipeline(confidence)
        result = pipeline.process_single_image(path)
        payload = _image_result_to_dict(result)

    else:
        images = sorted(p for p in path.iterdir() if p.suffix.lower() in ACCEPTED_EXTENSIONS)
        if not images:
            typer.echo(
                f"Error: no supported images ({', '.join(sorted(ACCEPTED_EXTENSIONS))}) " f"found in {path}",
                err=True,
            )
            raise typer.Exit(code=1)

        pipeline = _build_pipeline(confidence)
        payload = []
        for img_path in images:
            _validate_image(img_path)
            result = pipeline.process_single_image(img_path)
            payload.append(_image_result_to_dict(result))

    json_str = _dumps(payload)

    if output:
        output.write_text(json_str, encoding="utf-8")
        typer.echo(f"Results written to {output}")
    else:
        typer.echo(json_str)


@app.command()
def info() -> None:
    """Print model metadata and pipeline configuration as JSON.

    Useful for verifying which model checkpoints are loaded, confirming
    patch and batch sizes, and checking the default detection thresholds
    before running inference.

    **Output fields**

    - `package` / `version` — installed package identity
    - `models.root_detection` — checkpoint path, patch size, batch size, threshold
    - `models.anchor_detection` — same fields for the anchor model
    - `supported_formats` — file extensions accepted by `predict`
    - `description` — human-readable pipeline summary

    Example:
      npec-cv info
      npec-cv info | python -m json.tool   # pretty-print with indent
    """
    from npec_cv.configs.inference import (  # noqa: PLC0415
        ANCHOR_MODEL_CONFIG,
        ROOT_MODEL_CONFIG,
    )

    metadata = {
        "package": "npec_cv",
        "version": _get_version(),
        "models": {
            "root_detection": {
                "checkpoint": str(ROOT_MODEL_CONFIG.checkpoint),
                "patch_size": ROOT_MODEL_CONFIG.patch_size,
                "batch_size": ROOT_MODEL_CONFIG.batch_size,
                "default_threshold": ROOT_MODEL_CONFIG.threshold,
            },
            "anchor_detection": {
                "checkpoint": str(ANCHOR_MODEL_CONFIG.checkpoint),
                "patch_size": ANCHOR_MODEL_CONFIG.patch_size,
                "batch_size": ANCHOR_MODEL_CONFIG.batch_size,
                "default_threshold": ANCHOR_MODEL_CONFIG.threshold,
            },
        },
        "accepted_extensions": sorted(ACCEPTED_EXTENSIONS),
        "accepted_content_types": sorted(ACCEPTED_CONTENT_TYPES),
        "constraints": {
            "max_file_size_mb": MAX_FILE_SIZE_BYTES // (1024 * 1024),
            "min_resolution_px": MIN_RESOLUTION,
            "max_resolution_px": MAX_RESOLUTION,
        },
        "description": (
            "NPEC computer vision pipeline for plant root detection and measurement. "
            "Processes petri-dish images to segment roots, detect anchor points, "
            "and calculate geodesic root lengths."
        ),
    }
    typer.echo(_dumps(metadata))


@app.command()
def version() -> None:
    """Print the installed npec_cv package version.

    Prints the version string from the installed package metadata
    (set in pyproject.toml). Prints 'unknown' if the package is not
    installed in the current environment.

    Example:
      npec-cv version
      # 0.2.0
    """
    typer.echo(_get_version())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()

"""Image validation utilities for the inference endpoint.

All public functions raise fastapi.HTTPException with the correct
status code and api.schemas.common.ErrorResponse-shaped bodies so
that the router can call them without any extra error handling.
"""

import io
import logging

from fastapi import HTTPException, status
from PIL import Image, UnidentifiedImageError

from src.api.schemas.common import ErrorCode, ErrorResponse

logger = logging.getLogger(__name__)

# Constants

MAX_FILE_SIZE_BYTES: int = 50 * 1024 * 1024  # 50 MB
MIN_RESOLUTION: int = 512
MAX_RESOLUTION: int = 5000
ACCEPTED_CONTENT_TYPES: frozenset[str] = frozenset(["image/png", "image/tiff", "image/jpg", "image/jpeg"])
ACCEPTED_EXTENSIONS: frozenset[str] = frozenset([".png", ".tiff", ".tif", ".jpg", ".jpeg"])


def _error(status_code: int, error_code: ErrorCode, message: str) -> HTTPException:
    """Build an `HTTPException` with an `ErrorResponse` body."""
    return HTTPException(
        status_code=status_code,
        detail=ErrorResponse(error_code=error_code, message=message).model_dump(),
    )


def validate_file_size(data: bytes) -> None:
    """Raise HTTP 413 if `data` exceeds the 50 MB limit.

    Args:
        data: Raw bytes of the uploaded file.

    Raises:
        HTTPException: 413 with error code `FILE_TOO_LARGE`.
    """
    if len(data) > MAX_FILE_SIZE_BYTES:
        raise _error(
            status.HTTP_413_CONTENT_TOO_LARGE,
            ErrorCode.FILE_TOO_LARGE,
            f"File size {len(data):,} bytes exceeds the 50 MB limit.",
        )


def validate_content_type(content_type: str, filename: str) -> None:
    """Raise HTTP 415 if the MIME type or file extension is not accepted.

    Checks both the MINE type reported by the client and the file extension
    so that mislabelled uploads are caught as well.

    Args:
        content_type: MIME type from the `Content-Type` header.
        filename: Original filename used to extract the extension.

    Raises:
         HTTPException: 415 with error code `UNSUPPORTED_FORMAT`.
    """
    mime_ok = any(content_type.startswith(ct) for ct in ACCEPTED_CONTENT_TYPES)
    ext_ok = any(filename.lower().endswith(ext) for ext in ACCEPTED_EXTENSIONS)

    if not mime_ok or not ext_ok:
        raise _error(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            ErrorCode.UNSUPPORTED_FORMAT,
            f"Unsupported file fromat '{filename}'. " f"Accepted formats: {', '.join(sorted(ACCEPTED_EXTENSIONS))}.",
        )


def validate_image(data: bytes, filename: str) -> Image.Image:
    """Decode and validate the image, returning the opened `PIL.Image.Image`.

    Validates:
    - The bytes are a readablel image (catches corrupt files).
    - Resolution is withing [512, 5000] on both axes.
    - TIFF files are single-frame only.

    Args:
        data: Raw image bytes.
        filename: Original filename, used in error messages and TIFF detection.

    Returns:
        The opened `PIL.Image.Image` instance.

    Raises:
        HTTPException: 422 with appropriate error codes for each violation.
    """
    try:
        img = Image.open(io.BytesIO(data))
        img.verify()  # Check for corruption without fully decoding
        # Re-open after verify()
        img = Image.open(io.BytesIO(data))
    except UnidentifiedImageError:
        raise _error(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            ErrorCode.CORRUPT_FILE,
            "The uploaded file could not be decoded as an image.",
        )
    except Exception as exc:
        logger.warning("Unexpected error opening image '%s': '%s", filename, exc)
        raise _error(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            ErrorCode.CORRUPT_FILE,
            "The uploaded file could not be opened.",
        )

    # Resolution check
    width, height = img.size
    if width < MIN_RESOLUTION or height < MIN_RESOLUTION or width > MAX_RESOLUTION or height > MAX_RESOLUTION:
        raise _error(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            ErrorCode.INVALID_RESOLUTION,
            f"Image resolution {width}x{height} is out of the accepted range "
            f"{MIN_RESOLUTION}-{MAX_RESOLUTION} px on each axis.",
        )

    # Single-frame TIFF check
    if filename.lower().endswith((".tiff", ".tif")):
        try:
            img.seek(1)
            raise _error(
                status.HTTP_422_UNPROCESSABLE_CONTENT,
                ErrorCode.MULTI_PAGE_TIFF,
                "Only single-frame TIFF files are accepted.",
            )
        except EOFError:
            pass  # Exactly one frame
        img.seek(0)

    return img

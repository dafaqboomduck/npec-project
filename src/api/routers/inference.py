"""Inference router.

Exposes the four inference-related endpoints defined in api_spec.yaml:


- ``POST   /predict``                       — submit an image for inference
- ``GET    /predictions/{prediction_id}``   — retrieve a stored result
- ``DELETE /predictions/{prediction_id}``   — discard an unsaved result
- ``POST   /predictions/{prediction_id}/confirm`` — confirm and save a result
"""

import logging

from fastapi import APIRouter, Depends, File, Form, UploadFile, status

from api.dependencies import get_inference_service, get_prediction_repo
from api.repositories.base import AbstractPredictionRepository
from api.schemas.common import ErrorResponse
from api.schemas.predictions import PredictionResponse
from api.services.base import AbstractInferenceService
from api.services.image_validation import (
    validate_content_type,
    validate_file_size,
    validate_image,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Inference"])

# Endpoints


@router.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Submit a root scan image for inference",
    responses={
        413: {"model": ErrorResponse, "description": "File exceeds 50 MB limit"},
        415: {"model": ErrorResponse, "description": "Unsupported file foormat"},
        422: {"model": ErrorResponse, "description": "Validation error (resolution, corrupt file, ...)"},
    },
)
async def predict(
    file: UploadFile = File(..., description="Root scan image (PNG, TIFF, JPEG). Max 50 MB."),
    confidence_threshold: float = Form(
        default=0.50,
        ge=0.50,
        le=0.99,
        description="Minimum confidence for a detection to be included.",
    ),
    repo: AbstractPredictionRepository = Depends(get_prediction_repo),
    inference_service: AbstractInferenceService = Depends(get_inference_service),
) -> PredictionResponse:
    """Accept a root scan image and run U-Net inference.

    Validates file sizie, MIME type, image integrity, and resolution before
    dispatching to the inference service. The result is persisted in the
    rerpository and returned to the caller.

    Args:
        file: The uploaded image file.
        confidence_threshold: Predictions below this score are excluded from the results.
        repo: Injected prediction repository.
        inference_service: Injected inference service.

    Returns:
        :class:`~api.schemas.predictions.PredictionResponse` with a
        `prediction_id` UUID that can be used for later retrieval.

    Raises:
        HTTPException: 413 if the file exceeds 50 mb.
        HTTPException: 415 if the file format is not accepted.
        HTTPException: 422 if the image is corrupt or has an invalid resolution.
    """
    image_bytes = await file.read()

    # Validation pipeline
    validate_file_size(image_bytes)
    validate_content_type(file.content_type or "", file.filename or "")
    validate_image(image_bytes, file.filename or "")

    logger.info(
        "Running inference on '%s' (%d bytes, threshold=%.2f)",
        file.filename,
        len(image_bytes),
        confidence_threshold,
    )

    prediction = await inference_service.run(
        image_bytes=image_bytes,
        filename=file.filename or "",
        confidence_threshold=confidence_threshold,
    )

    await repo.save(prediction)
    return prediction

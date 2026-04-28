"""Inference router.

Exposes the four inference-related endpoints defined in api_spec.yaml:


- ``POST   /predict``                       — submit an image for inference
- ``GET    /predictions/{prediction_id}``   — retrieve a stored result
- ``DELETE /predictions/{prediction_id}``   — discard an unsaved result
- ``POST   /predictions/{prediction_id}/confirm`` — confirm and save a result
"""

import logging
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from db.session import get_session
from src.api.core.security import get_current_user
from src.api.dependencies import get_inference_service, get_prediction_repo
from src.api.repositories.base import AbstractPredictionRepository
from src.api.repositories.database import DatabasePredictionRepository
from src.api.schemas.auth import TokenData
from src.api.schemas.common import ErrorCode, ErrorResponse, MessageResponse
from src.api.schemas.predictions import (
    ConfirmPredictionRequest,
    ConfirmPredictionResponse,
    PredictionResponse,
)
from src.api.services.base import AbstractInferenceService
from src.api.services.image_validation import (
    validate_content_type,
    validate_file_size,
    validate_image,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Inference"])

# Helpers

_NOT_FOUND = ErrorResponse(
    error_code=ErrorCode.NOT_FOUND,
    message="Prediction not found.",
)


def _404() -> HTTPException:
    """404 HTTPException helper."""
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=_NOT_FOUND.model_dump(),
    )


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
    user: Annotated[TokenData, Depends(get_current_user)] = None,
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
        user: The authenticated user extracted from the Bearer token.
        repo: Injected prediction repository.
        inference_service: Injected inference service.

    Returns:
        `PredictionResponse` with a `prediction_id` UUID that can be used for later retrieval.

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


@router.get(
    "/predictions/{prediction_id}",
    response_model=PredictionResponse,
    summary="Retrieve a prediction result",
    responses={404: {"model": ErrorResponse, "description": "Prediction not found"}},
)
async def get_prediction(
    prediction_id: UUID,
    user: Annotated[TokenData, Depends(get_current_user)],
    repo: AbstractPredictionRepository = Depends(get_prediction_repo),
) -> PredictionResponse:
    """Return a previously stored prediction by its UUID.

    Args:
        prediction_id: UUID of the prediction to retrieve.
        user: The authenticated user extracted from the Bearer token.
        repo: Injected prediction repository.

    Returns:
        The stored `PredictionResponse`.

    Raises:
        HTTPException: 404 if no prediction with `prediction_id` exists.
    """
    prediction = await repo.get(prediction_id)
    if prediction is None:
        raise _404()
    return prediction


@router.delete(
    "/predictions/{prediction_id}",
    response_model=MessageResponse,
    summary="Discard an unsaved prediction",
    responses={404: {"model": ErrorResponse, "description": "Prediction not found"}},
)
async def delete_prediction(
    prediction_id: UUID,
    user: Annotated[TokenData, Depends(get_current_user)] = None,
    repo: AbstractPredictionRepository = Depends(get_prediction_repo),
) -> MessageResponse:
    """Remove a prediction from the store.

    Args:
        prediction_id: UUID of the prediction to discard.
        user: The authenticated user extracted from the Bearer token.
        repo: Injected prediction repository.

    Returns:
        MessageResponse confirming deletion.

    Raises:
        HTTPException: 404 if no prediction with `prediction_id` exists.
    """
    deleted = await repo.delete(prediction_id)
    if not deleted:
        raise _404()
    logger.info("Discarded prediction %s by user %s", prediction_id, user.sub)
    return MessageResponse(message=f"Prediction {prediction_id} discarded.")


@router.post(
    "/predictions/{prediction_id}/confirm",
    response_model=ConfirmPredictionResponse,
    summary="Confirm and save a prediction result",
    responses={
        401: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        409: {
            "model": ErrorResponse,
            "description": "Prediction already confirmed",
        },
    },
)
async def confirm_prediction(
    prediction_id: UUID,
    body: ConfirmPredictionRequest | None = None,
    user: Annotated[TokenData, Depends(get_current_user)] = None,
    db: AsyncSession = Depends(get_session),
) -> ConfirmPredictionResponse:
    """Confirm a prediction and create a linked annotation.

    Permanently saves the prediction, links it to a scan, and creates an
    annotation record pre-populated with the provided biological metadata.

    The annotation becomes queryable via GET /annotations after this call.

    Args:
        prediction_id: UUID of the prediction to confirm.
        body: Optional body with `species`, `age_days`, and `treatment`.
        user: Authenticated user from Bearer token.
        db: Injected database session.

    Returns:
        `ConfirmPredictionResponse`

    Raises:
        HTTPException: 404 if the prediction does not exist.
        HTTPException: 409 if the prediction is already confirmed.
    """
    repo = DatabasePredictionRepository(db=db, user_id=UUID(user.sub))

    try:
        _, already_confirmed = await repo.confirm(
            prediction_id=prediction_id,
            species=body.species if body else None,
            age_days=body.age_days if body else None,
            treatment=body.treatment if body else None,
        )
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=ErrorResponse(
                error_code=ErrorCode.NOT_FOUND,
                message="Prediction not found.",
            ).model_dump(),
        )

    if already_confirmed:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=ErrorResponse(
                error_code=ErrorCode.CONFLICT,
                message="Prediction has already been confirmed.",
            ).model_dump(),
        )

    logger.info("Confirmed prediction %s by user %s", prediction_id, user.sub)
    return ConfirmPredictionResponse(
        message="Prediction confirmed and annotation created.",
        prediction_id=prediction_id,
    )

"""Inference router.

Exposes the four inference-related endpoints defined in api_spec.yaml:


- ``POST   /predict``                       — submit an image for inference
- ``GET    /predictions/{prediction_id}``   — retrieve a stored result
- ``DELETE /predictions/{prediction_id}``   — discard an unsaved result
- ``POST   /predictions/{prediction_id}/confirm`` — confirm and save a result
"""

from __future__ import annotations

import asyncio
import logging
from typing import Annotated
from uuid import UUID

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    Query,
    UploadFile,
    status,
)
from sqlalchemy.ext.asyncio import AsyncSession

from api.src.core.security import get_current_user
from api.src.db.session import get_session
from api.src.dependencies import get_inference_service, get_storage
from api.src.repositories.database import DatabasePredictionRepository
from api.src.schemas.auth import TokenData
from api.src.schemas.common import ErrorCode, ErrorResponse, MessageResponse
from api.src.schemas.predictions import (
    ConfirmPredictionRequest,
    ConfirmPredictionResponse,
    PredictionResponse,
)
from api.src.services.artifact_storage import (
    save_input_image,
    save_segmentation_mask,
    save_tip_photo,
)
from api.src.services.base import AbstractInferenceService
from api.src.services.image_validation import (
    validate_content_type,
    validate_file_size,
    validate_image,
)
from api.src.services.monitoring import MonitoringService
from api.src.services.storage import AbstractStorageBackend

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
        415: {"model": ErrorResponse, "description": "Unsupported file format"},
        422: {"model": ErrorResponse, "description": "Validation error (resolution, corrupt file, ...)"},
    },
)
async def predict(
    file: UploadFile = File(..., description="Root scan image (PNG, TIFF, JPEG). Max 50 MB."),
    user: Annotated[TokenData, Depends(get_current_user)] = None,
    db: AsyncSession = Depends(get_session),
    inference_service: AbstractInferenceService = Depends(get_inference_service),
    storage: AbstractStorageBackend = Depends(get_storage),
) -> PredictionResponse:
    """Accept a root scan image and run U-Net inference.

    Validates file size, MIME type, image integrity, and resolution before
    dispatching to the inference service. The result is persisted in the
    repository and returned to the caller.

    Args:
        file: The uploaded image file.
        user: The authenticated user extracted from the Bearer token.
        db: Injected database session.
        inference_service: Injected inference service.
        storage: Injected storage backend.

    Returns:
        `PredictionResponse` with a `prediction_id` UUID that can be used for later retrieval.

    Raises:
        HTTPException: 413 if the file exceeds 50 mb.
        HTTPException: 415 if the file format is not accepted.
        HTTPException: 422 if the image is corrupt or has an invalid resolution.
    """
    import hashlib

    repo = DatabasePredictionRepository(db=db, user_id=UUID(user.sub), storage=storage)

    image_bytes = await file.read()
    input_hash = hashlib.sha256(image_bytes).hexdigest()
    original_filename = file.filename or "upload"

    # Validation pipeline
    validate_file_size(image_bytes)
    validate_content_type(file.content_type or "", original_filename)
    validate_image(image_bytes, original_filename)

    logger.info(
        "Running inference on '%s' (%d bytes)",
        file.filename,
        len(image_bytes),
    )

    prediction, image_result = await inference_service.run(
        image_bytes=image_bytes,
        filename=original_filename,
        db=db,
    )

    prediction_id = prediction.prediction_id
    logger.info("Inference complete for '%s' - prediction_id=%s user=%s", original_filename, prediction_id, user.sub)

    # Save input image
    input_uri: str | None = save_input_image(
        prediction_id=prediction_id,
        image_bytes=image_bytes,
        original_filename=original_filename,
        backend=storage,
    )

    # Save mask and tip photo
    mask_uri: str | None = None
    tip_photo_uri: str | None = None

    if image_result is not None:
        mask_uri = save_segmentation_mask(
            prediction_id=prediction_id,
            image_result=image_result,
            backend=storage,
        )
        tip_photo_uri = save_tip_photo(
            prediction_id=prediction_id,
            image_result=image_result,
            original_image_bytes=image_bytes,
            backend=storage,
        )

    # Patch prediction with URLs
    prediction = prediction.model_copy(update={"segmentation_mask_url": mask_uri})

    # Persist to repository
    prediction = await repo.save(
        prediction,
        filename=original_filename,
        input_hash=input_hash,
        input_uri=input_uri,
        tip_photo_uri=tip_photo_uri,
    )

    # Write monitoring metrics
    try:
        monitoring = MonitoringService(db)
        await monitoring.record_prediction_metrics(
            prediction_id=prediction.prediction_id,
            processing_time_ms=prediction.metadata.processing_time_ms,
        )
    except Exception as exc:
        logger.warning(
            "Failed to record monitoring metrics for prediction %s: %s",
            prediction.prediction_id,
            exc,
        )

    return prediction


@router.get(
    "/predictions",
    response_model=list[PredictionResponse],
    summary="List all predictions",
    responses={401: {"model": ErrorResponse}},
)
async def list_predictions(
    page: int = Query(default=1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(default=20, ge=1, le=100, description="Results per page"),
    confirmed: bool | None = Query(
        default=None,
        description="Fitler by confirmation status. True=confirmed, False=unconfirmed, omit=all.",
    ),
    user: Annotated[TokenData, Depends(get_current_user)] = None,
    db: AsyncSession = Depends(get_session),
    storage: AbstractStorageBackend = Depends(get_storage),
) -> list[PredictionResponse]:
    """Return a paginated list of predictions.

    Pagination is performed at the database level via SQL OFFSET/LIMIT.

    Args:
        page: 1-indexed page number.
        page_size: Number of results per page (max 100).
        confirmed: Boolean determining the confirmation status.
        user: Authenticated user from Bearer token.
        db: Injected database session.
        storage: Injected storage backend.

    Returns:
        Paginated list of `AnnotationResponse`.
    """
    repo = DatabasePredictionRepository(db=db, user_id=UUID(user.sub), storage=storage)

    return await repo.list_by_user(
        page=page,
        page_size=page_size,
        confirmed=confirmed,
        role=user.role,
    )


@router.get(
    "/predictions/{prediction_id}",
    response_model=PredictionResponse,
    summary="Retrieve a prediction result",
    responses={404: {"model": ErrorResponse, "description": "Prediction not found"}},
)
async def get_prediction(
    prediction_id: UUID,
    user: Annotated[TokenData, Depends(get_current_user)],
    db: AsyncSession = Depends(get_session),
    storage: AbstractStorageBackend = Depends(get_storage),
) -> PredictionResponse:
    """Return a previously stored prediction by its UUID.

    Args:
        prediction_id: UUID of the prediction to retrieve.
        user: The authenticated user extracted from the Bearer token.
        db: Injected database session.
        storage: Injected storage backend.

    Returns:
        The stored `PredictionResponse`.

    Raises:
        HTTPException: 404 if no prediction with `prediction_id` exists.
    """
    repo = DatabasePredictionRepository(db=db, user_id=UUID(user.sub), storage=storage)

    prediction = await repo.get(prediction_id, role=user.role)
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
    db: AsyncSession = Depends(get_session),
    storage: AbstractStorageBackend = Depends(get_storage),
) -> MessageResponse:
    """Remove a prediction from the store.

    Args:
        prediction_id: UUID of the prediction to discard.
        user: The authenticated user extracted from the Bearer token.
        db: Injected database session.
        storage: Injected storage backend.

    Returns:
        MessageResponse confirming deletion.

    Raises:
        HTTPException: 404 if no prediction with `prediction_id` exists.
    """
    repo = DatabasePredictionRepository(db=db, user_id=UUID(user.sub), storage=storage)

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
            "description": "Prediction already confirmed or the image was uploaded before",
        },
    },
)
async def confirm_prediction(
    prediction_id: UUID,
    body: ConfirmPredictionRequest | None = None,
    user: Annotated[TokenData, Depends(get_current_user)] = None,
    db: AsyncSession = Depends(get_session),
    storage: AbstractStorageBackend = Depends(get_storage),
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
        storage: Injected storage backend.

    Returns:
        `ConfirmPredictionResponse`

    Raises:
        HTTPException: 404 if the prediction does not exist.
        HTTPException: 404 if dataset_id is provided but the dataset does not exist.
        HTTPException: 409 if the prediction is already confirmed.
    """
    repo = DatabasePredictionRepository(db=db, user_id=UUID(user.sub), storage=storage)

    try:
        _, already_confirmed, annotation_id = await repo.confirm(
            prediction_id=prediction_id,
            species=body.species if body else None,
            age_days=body.age_days if body else None,
            treatment=body.treatment if body else None,
            dataset_id=body.dataset_id if body else None,
        )

        if body and body.feedback_signal:
            from api.src.repositories.feedback import FeedbackRepository

            await FeedbackRepository(db).create(
                user_id=UUID(user.sub),
                prediction_id=prediction_id,
                signal=body.feedback_signal,
                comment=None,
            )
    except ValueError as exc:
        if "identical image" in str(exc):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=ErrorResponse(
                    error_code=ErrorCode.CONFLICT,
                    message=str(exc),
                ).model_dump(),
            )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=ErrorResponse(
                error_code=ErrorCode.NOT_FOUND,
                message=str(exc),
            ).model_dump(),
        )

    if already_confirmed:
        return ConfirmPredictionResponse(
            message="Prediction already confirmed.",
            prediction_id=prediction_id,
            annotation_id=annotation_id,
        )

    asyncio.create_task(
        _trigger_retraining_check(),
        name="retraining-check-feedback",
    )

    logger.info("Confirmed prediction %s by user %s", prediction_id, user.sub)
    return ConfirmPredictionResponse(
        message="Prediction confirmed and annotation created.",
        prediction_id=prediction_id,
        annotation_id=annotation_id,
    )


async def _trigger_retraining_check() -> None:
    """Fire-and-forget wrapper for the retraining threshold check.

    Uses a fresh DB session for the retraining check so it is not
    tied to the request session which may be closed before the task runs.
    """
    from api.src.db.session import get_sessionmaker
    from api.src.services.retraining import RetrainingService

    try:
        async with get_sessionmaker()() as fresh_db:
            await RetrainingService(fresh_db).check_and_trigger()
    except Exception as exc:
        logger.warning("Retraining check failed after confirm: %s", exc)

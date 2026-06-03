"""Feedback router.

Exposes POST /feedback. Multiple submissions from the same user
on the same prediction are allowed.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from api.src.core.security import get_current_user
from api.src.db.session import get_session
from api.src.dependencies import get_storage
from api.src.repositories.annotations import AnnotationRepository
from api.src.repositories.feedback import FeedbackRepository
from api.src.schemas.auth import TokenData
from api.src.schemas.common import ErrorCode, ErrorResponse, MessageResponse
from api.src.schemas.feedback import FeedbackRequest
from api.src.schemas.library import AnnotationStatus
from api.src.services.storage import AbstractStorageBackend

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Feedback"])


@router.post(
    "/feedback",
    response_model=MessageResponse,
    summary="Submit feedback on a prediction",
    responses={
        401: {"model": ErrorResponse},
        404: {"model": ErrorResponse, "description": "Prediction not found"},
    },
)
async def submit_feedback(
    body: FeedbackRequest,
    user: Annotated[TokenData, Depends(get_current_user)] = None,
    db: AsyncSession = Depends(get_session),
    storage: AbstractStorageBackend = Depends(get_storage),
) -> MessageResponse:
    """Submit a thumbs-up or thumbs-down signal on a prediction.

    Multiple submissions from the same user are allowed - each is recorded
    independently. For structured mask corrections use
    POST /annotations/{id}/confirm instead.

    When signal='incorrect', checks the retraining threshold and triggers
    a new training run if enough incorrect signal have accumulated.

    Args:
        body: Request body with `prediction_id`, `feedback_signal`, and optional
            `comment`.
        user: Authenticated user from Bearer token.
        db: Injected database session.
        storage: Injected backend storage.

    Returns:
        `MessageResponse` confirming the feedback was recorded.

    Raises:
        HTTPException: 401 if the Bearer token is missing or invalid.
        HTTPException: 404 if the referenced prediction does not exist.
    """
    repo = FeedbackRepository(db)
    annot_repo = AnnotationRepository(db, storage)

    if not await repo.prediction_exists(body.prediction_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=ErrorResponse(
                error_code=ErrorCode.NOT_FOUND,
                message="Prediction not found.",
            ).model_dump(),
        )

    await repo.create(
        user_id=UUID(user.sub),
        prediction_id=body.prediction_id,
        signal=body.feedback_signal,
        comment=body.comment,
    )

    if body.feedback_signal == "incorrect":
        await annot_repo.set_status(
            body.prediction_id,
            AnnotationStatus.REVIEW_REQUIRED,
            create_if_missing=True,
        )
        asyncio.create_task(
            _trigger_retraining_check(),
            name="retraining-check-feedback",
        )
    else:
        await annot_repo.set_status(
            body.prediction_id,
            AnnotationStatus.ANNOTATED,
        )

    return MessageResponse(message="Feedback recorded.")


async def _trigger_retraining_check() -> None:
    """Fire-and-forget wrapper for the retraining threshold check.

    Creates a fresh DB session for the retraining check so it is not
    tied to the request session which may be closed before the task runs.
    """
    from api.src.db.session import get_sessionmaker
    from api.src.services.retraining import RetrainingService

    try:
        async with get_sessionmaker()() as fresh_db:
            await RetrainingService(fresh_db).check_and_trigger()
    except Exception as exc:
        logger.warning("Retraining check failed after feedback: %s", exc)

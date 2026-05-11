"""Feedback router.

Exposes POST /feedback. Multiple submissions from the same user
on the same prediction are allowed.
"""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from db.session import get_session
from src.api.core.security import get_current_user
from src.api.repositories.feedback import FeedbackRepository
from src.api.schemas.auth import TokenData
from src.api.schemas.common import ErrorCode, ErrorResponse, MessageResponse
from src.api.schemas.feedback import FeedbackRequest

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
) -> MessageResponse:
    """Submit a thumbs-up or thumbs-down signal on a prediction.

    Multiple submissions from the same user are allowed - each is recorded
    independently. For structured mask corrections use
    POST /annotations/{id}/confirm instead.

    Args:
        body: Request body with `prediction_id`, `signal`, and optional
            `comment`.
        user: Authenticated user from Bearer token.
        db: Injected database session.

    Returns:
        `MessageResponse` confirming the feedback was recorded.

    Raises:
        HTTPException: 401 if the Bearer token is missing or invalid.
        HTTPException: 404 if the referenced prediction does not exist.
    """
    repo = FeedbackRepository(db)

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
        signal=body.signal,
        comment=body.comment,
    )

    return MessageResponse(message="Feedback recorded.")

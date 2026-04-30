"""Feedback schemas."""

from enum import Enum
from uuid import UUID

from pydantic import BaseModel, Field


class FeedbackSignal(str, Enum):
    """Feedback signal enum."""

    CORRECT = "correct"
    INCORRECT = "incorrect"


class FeedbackRequest(BaseModel):
    """Body for POST /feedback."""

    prediction_id: UUID = Field(
        ...,
        description="UUID of the prediction being rated.",
    )
    signal: FeedbackSignal = Field(
        ...,
        description="Thumbs up (correct) or thumbs down (incorrect).",
    )
    comment: str | None = Field(
        None,
        max_length=500,
        descripton="Optional free-text note from the user.",
    )

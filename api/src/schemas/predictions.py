"""Prediction-related Pydantic schemas.

Schemas here cover the full request/response lifecycle for the
POST /predict, GET /predictions/{id}, DELETE /predictions/{id}, and
POST /predictions/{id}/confirm endpoints.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field

# Enums


class PredictionStatus(str, Enum):
    """Prediction status."""

    OK = "ok"
    ERROR = "error"


# Sub-models


class PredictionMetadata(BaseModel):
    """Metadata captured during inference."""

    model_version: str
    inference_timestamp: datetime
    processing_time_ms: int
    input_resolution: list[int] = Field(..., min_length=2, max_length=2)
    input_channels: int
    normalisation_applied: bool
    scale_factor_mm_per_px: float


class Plant(BaseModel):
    """Per-plant detection result."""

    plant_id: int = Field(..., description="1-indexed, left-to-right across image")
    tip_x: int = Field(..., description="X coordinate of root tip in pixel space")
    tip_y: int = Field(..., description="Y coordinate of root tip in pixel space")
    root_length_px: float
    root_length_mm: float


# Request / Response models


class PredictionResponse(BaseModel):
    """Full prediction result returned by POST /predict and GET /predictions/{id}."""

    prediction_id: UUID
    status: PredictionStatus
    input_filename: str | None = Field(None, description="Original filename of the uploaded image.")
    owner_email: str | None = None
    metadata: PredictionMetadata
    segmentation_mask_url: str | None = Field(
        None,
        description="Presigned URL to the output mask PNG. None if inference failed.",
    )
    input_url: str | None = Field(
        None,
        description="Presigned URL of the original input image.",
    )
    tip_photo_url: str | None = Field(
        None,
        description="Presigned URL of the tip visualisation overlay.",
    )
    plants: list[Plant] = Field(default_factory=list)
    feedback_signal: str | None = None
    warnings: list[str] = Field(default_factory=list)


class ConfirmPredictionRequest(BaseModel):
    """Body for POST /predictions/{id}/confirm."""

    species: str | None = Field(
        None,
        description="Plant species observed in the scan.",
    )
    age_days: int | None = Field(
        None,
        ge=0,
        description="Age of the plant in days at time of scan.",
    )
    treatment: str | None = Field(
        None,
        description="Experimental treatment applied to the plant.",
    )
    dataset_id: UUID | None = Field(
        None,
        description=(
            "Optional UUID of the dataset to link this annotation to. "
            "When provided, the created annotation has dataset_id set and "
            "datasets.count is incremented by 1."
        ),
    )
    feedback_signal: Literal["correct", "incorrect"] | None = None


class ConfirmPredictionResponse(BaseModel):
    """Returned after a prediction is confirmed and saved."""

    message: str
    prediction_id: UUID
    annotation_id: UUID

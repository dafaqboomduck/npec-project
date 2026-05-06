"""Prediction-related Pydantic schemas.

Schemas here cover the full request/response lifecycle for the
POST /predict, GET /predictions/{id}, DELETE /predictions/{id}, and
POST /predictions/{id}/confirm endpoints.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
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
    input_resolution: int
    input_resolution: list[int] = Field(..., min_length=2, max_length=2)
    input_channels: int
    normalisation_applied: bool
    scale_factor_mm_per_px: float


class Plant(BaseModel):
    """Per-plant detection result."""

    plant_id: int = Field(..., description="1-indexed, left-to-right across image")
    tip_x: int = Field(..., description="X coordinate of root tip in pixel space")
    tip_y: int = Field(..., description="Y coordinate of root tip in pixel space")
    confidence: float = Field(..., ge=0.0, le=1.0)
    root_length_px: float
    root_length_mm: float


# Request / Resopnse models


class PredictionResponse(BaseModel):
    """Full prediction result returned by POST /predict and GET /predictions/{id}."""

    prediction_id: UUID
    status: PredictionStatus
    input_filename: str | None = Field(None, description="Original filename of the uploaded image.")
    metadata: PredictionMetadata
    segmentation_mask_url: str | None = Field(
        None,
        description="Presigned URL to the output mask PNG. None if inference failed.",
    )
    plants: list[Plant] = Field(default_factory=list)
    overall_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Mean of per-plant confidence scores. 0.0 if no plants detected.",
    )
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


class ConfirmPredictionResponse(BaseModel):
    """Returned after a prediction is confirmed and saved."""

    message: str
    prediction_id: UUID

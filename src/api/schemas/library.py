"""Library Pydantic schemas.

These are used by the GET-only library endpoints that
feed the frontend's Library page and Model Info page.
"""

from datetime import datetime
from enum import Enum
from uuid import UUID

from pydantic import BaseModel

# Enums


class GrowthStage(str, Enum):
    """Growth stage enum."""

    SEEDLING = "seedling"
    MATURE = "mature"


class AnnotationStatus(str, Enum):
    """Annotation status enum."""

    PENDING = "pending"
    ANNOTATED = "annotated"
    REVIEW_REQUIRED = "review_required"


class ModelStatus(str, Enum):
    """Model status enum."""

    DEPLOYED = "deployed"
    REGISTERED = "registered"
    ARCHIVED = "archived"


# Dataset schemas


class DatasetSummary(BaseModel):
    """Returned bz GET /datasets (list item)."""

    dataset_id: UUID
    name: str
    created_at: datetime
    modified_at: datetime
    count: int


class DatasetDetail(DatasetSummary):
    """Returned by GET /datasets/{dataset_id}."""

    description: str | None = None
    species: str | None = None


# Annotation schemas


class AnnotationResponse(BaseModel):
    """Returned by GET /annotations and GET /annotations/{id}."""

    annotation_id: UUID
    scan_id: UUID
    status: AnnotationStatus
    mask_url: str | None = None
    species: str | None = None
    age_days: int | None = None
    treatment: str | None = None
    created_at: datetime
    updated_at: datetime


# Model schemas


class ModelSummary(BaseModel):
    """Returned by GET /models (list item)."""

    model_id: UUID
    name: str
    status: ModelStatus
    created_at: datetime


class ModelDetail(ModelSummary):
    """Returned by GET /models/{model_id}.

    Admin tokens additionally receive `artifact_uri` and `deployed_at`
    which are filtered server-side in the router.
    """

    description: str | None = None
    f1_score: float | None = None
    precision: float | None = None
    recall: float | None = None
    inference_latency_ms: int | None = None
    # Admin-only fields - None for user-role tokens
    artifact_uri: str | None = None
    deployed_at: datetime | None = None

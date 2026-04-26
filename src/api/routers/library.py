"""Library router.

Exposes six read-only endpoints:


- GET /datasets                      — list datasets (filterable)
- GET /datasets/{dataset_id}         — single dataset detail
- GET /annotations                   — paginated annotation list (filterable)
- GET /annotations/{annotation_id}   — single annotation detail
- GET /models                        — list all model versions
- GET /models/{model_id}             — single model detail (admin gets extra fields)

All endpoints require a valid Bearer token. Repositoreis are initiated per-request
from the injected DB session.
"""

import logging
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from db.session import get_session
from src.api.core.security import get_current_user
from src.api.repositories.annotations import AnnotationRepository
from src.api.repositories.datasets import DatasetRepository
from src.api.repositories.models import ModelRepository
from src.api.schemas.auth import TokenData, UserRole
from src.api.schemas.common import ErrorCode, ErrorResponse
from src.api.schemas.library import (
    AnnotationResponse,
    DatasetDetail,
    DatasetSummary,
    GrowthStage,
    ModelDetail,
    ModelSummary,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Library"])

# Shared 404 helper

_NOT_FOUND = ErrorResponse(error_code=ErrorCode.NOT_FOUND, message="Resource not found.")


def _404() -> HTTPException:
    """404 HTTPException helper."""
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=_NOT_FOUND.model_dump(),
    )


# Datasets


@router.get(
    "/datasets",
    response_model=list[DatasetSummary],
    summary="List all datasets",
    responses={401: {"model": ErrorResponse}},
)
async def list_datasets(
    species: str | None = Query(None, description="Filter by species name"),
    growth_stage: GrowthStage | None = Query(None, description="Filter by growth stage"),
    user: Annotated[TokenData, Depends(get_current_user)] = None,
    db: AsyncSession = Depends(get_session),
) -> list[DatasetSummary]:
    """Return all datasets, optionally filtered by species and growth stage.

    Args:
        species: Optional species name filter.
        growth_stage: Optional growth stage filter (`seedling` or `mature`).
        user: Authenticated user from Bearer token.
        db: Injected database session.

    Returns:
        List of `DatasetSummary`.
    """
    repo = DatasetRepository(db)
    return await repo.list(
        species=species,
        growth_stage=growth_stage.value if growth_stage else None,
    )


@router.get(
    "/datasets/{dataset_id}",
    response_model=DatasetDetail,
    summary="Single dataset detail",
    responses={
        401: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
    },
)
async def get_dataset(
    dataset_id: UUID,
    user: Annotated[TokenData, Depends(get_current_user)] = None,
    db: AsyncSession = Depends(get_session),
) -> DatasetDetail:
    """Return the dataset with `dataset_id`.

    Args:
        dataset_id: UUID of the dataset.
        user: Authenticated user from Bearer token.
        db: Injected database session.

    Returns:
        `DatasetDetail`

    Raises:
        HTTPException: 404 if the dataset does not exist.
    """
    repo = DatasetRepository(db)
    dataset = await repo.get(dataset_id)
    if dataset is None:
        raise _404()
    return dataset


# Annotations


@router.get(
    "/annotations",
    response_model=list[AnnotationResponse],
    summary="List all annotations",
    responses={401: {"model": ErrorResponse}},
)
async def list_annotations(
    species: str | None = Query(None, description="Filter by species name"),
    growth_stage: GrowthStage | None = Query(None, description="Filter by growth stage"),
    page: int = Query(default=1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(default=20, ge=1, le=100, description="Results per page"),
    user: Annotated[TokenData, Depends(get_current_user)] = None,
    db: AsyncSession = Depends(get_session),
) -> list[AnnotationResponse]:
    """Return a paginated list of annotations.

    Pagination is performed at the database level via SQL OFFSET/LIMIT.

    Args:
        species: Optional species name filter.
        growth_stage: Optional growth stage filter.
        page: 1-indexed page number.
        page_size: Number of results per page (max 100).
        user: Authenticated user from Bearer token.
        db: Injected database session.

    Returns:
        Paginated list of `AnnotationResponse`.
    """
    repo = AnnotationRepository(db)
    return await repo.list(
        species=species,
        growth_stage=growth_stage.value if growth_stage else None,
        page=page,
        page_size=page_size,
    )


@router.get(
    "/annotations/{annotation_id}",
    response_model=AnnotationResponse,
    summary="Single annotation detail",
    responses={
        401: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
    },
)
async def get_annotation(
    annotation_id: UUID,
    user: Annotated[TokenData, Depends(get_current_user)] = None,
    db: AsyncSession = Depends(get_session),
) -> AnnotationResponse:
    """Return the annotation with `annotation_id`.

    Args:
        annotation_id: UUID of the annotation.
        user: Authenticated user from Bearer token.
        db: Injected database session.

    Returns:
        AnnotationResponse

    Raises:
        HTTPException: 404 if the annotation does not exist.
    """
    repo = AnnotationRepository(db)
    annotation = await repo.get(annotation_id)
    if annotation is None:
        raise _404()
    return annotation


# Models


@router.get(
    "/models",
    response_model=list[ModelSummary],
    summary="List all models",
    responses={401: {"model": ErrorResponse}},
)
async def list_models(
    user: Annotated[TokenData, Depends(get_current_user)] = None,
    db: AsyncSession = Depends(get_session),
) -> list[ModelSummary]:
    """Return all registered model versions.

    Args:
        user: Authenticated user from Bearer token.
        db: Injected database session.

    Returns:
        List of `ModelSummary`.
    """
    repo = ModelRepository(db)
    return await repo.list()


@router.get(
    "/models/{model_id}",
    response_model=ModelDetail,
    summary="Single model detail",
    description=(
        "Returns model metadata, parameters, and metrics. "
        "Admin tokens receive additional internal fields (artifact_uri, deployed_at) "
        "filtered server-side based on JWT role claim."
    ),
    responses={
        401: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
    },
)
async def get_model(
    model_id: UUID,
    user: Annotated[TokenData, Depends(get_current_user)] = None,
    db: AsyncSession = Depends(get_session),
) -> ModelDetail:
    """Return the model version with `model_id`.

    Admin tokens receive additional internal fields — ``artifact_uri`` and
    ``deployed_at`` — which are ``None`` for user-role tokens.

    Args:
        model_id: UUID of the model version.
        user: Authenticated user from Bearer token - role determines which
            fields are returned.
        db: Injected database session.

    Returns:
        ModelDetail

    Raises:
        HTTPException: 404 if the model does not exist.
    """
    is_admin = user.role == UserRole.ADMIN
    repo = ModelRepository(db)
    model = await repo.get(model_id, include_admin_fields=is_admin)
    if model is None:
        raise _404()
    return model

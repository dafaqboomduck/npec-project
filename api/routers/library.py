"""Library router.

Read and write endpoints for datasets, annotations, and model versions.

Read endpoints :
- GET  /datasets                       — list datasets
- GET  /datasets/{dataset_id}          — single dataset detail
- GET  /annotations                    — paginated annotation list
- GET  /annotations/{annotation_id}    — single annotation detail
- GET  /models                         — list all model versions
- GET  /models/{model_id}              — single model detail (admin gets extra fields)

Write endpoints:
- DELETE /annotations/{id}             — remove annotation
- POST   /annotations/{id}/confirm     — submit corrected mask
- DELETE /models/{id}                  — remove model version (admin only)
- POST   /models/{id}/deploy           — deploy model version (admin only)

All endpoints require a valid Bearer token. Repositoreis are initiated per-request
from the injected DB session.
"""

from __future__ import annotations

import logging
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from db.session import get_session

from ..core.security import get_current_user, require_admin
from ..repositories.annotations import AnnotationRepository
from ..repositories.datasets import DatasetRepository
from ..repositories.models import ModelRepository
from ..schemas.auth import TokenData, UserRole
from ..schemas.common import ErrorCode, ErrorResponse, MessageResponse
from ..schemas.library import (
    AnnotationResponse,
    DatasetCreate,
    DatasetDetail,
    DatasetSummary,
    GrowthStage,
    ModelDetail,
    ModelSummary,
)
from ..services.deployment import StubDeploymentService
from ..services.mask_storage import save_annotation_mask, validate_tiff_mask

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Library"])

# Shared 404 helper


def _404(message: str = "Resource not found.") -> HTTPException:
    """404 HTTPException helper."""
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=ErrorResponse(
            error_code=ErrorCode.NOT_FOUND,
            message=message,
        ).model_dump(),
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


@router.post(
    "/datasets",
    response_model=DatasetDetail,
    summary="Create a new dataset (admin only)",
    responses={
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
    },
)
async def create_dataset(
    data: DatasetCreate,
    user: Annotated[TokenData, Depends(require_admin)] = None,
    db: AsyncSession = Depends(get_session),
) -> DatasetDetail:
    """Create a new dataset.

    Args:
        data: `DatasetCreate` containing name, description, species, growth_stage.
        user: Admin-role authenticated user.
        db: Injected database session.

    Returns:
        The created `DatasetDetail`.

    Raises:
        HTTPException: 403 for non-admin tokens.
    """
    repo = DatasetRepository(db)
    dataset = await repo.create(
        name=data.name,
        description=data.description,
        species=data.species,
        growth_stage=data.growth_stage.value if data.growth_stage else None,
    )
    logger.info("Admin %s created dataset %s", user.sub, dataset.dataset_id)
    return dataset


@router.delete(
    "/datasets/{dataset_id}",
    response_model=MessageResponse,
    summary="Delete a dataset (admin only)",
    responses={
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
    },
)
async def delete_dataset(
    dataset_id: UUID,
    user: Annotated[TokenData, Depends(require_admin)] = None,
    db: AsyncSession = Depends(get_session),
) -> MessageResponse:
    """Remove the dataset with `dataset_id`.

    Admin only.

    Args:
        dataset_id: UUID of the dataset to delete.
        user: Admin-role authenticated user.
        db: Injected database session.

    Returns:
        `MessageResponse` confirming deletion.

    Raises:
        HTTPException: 403 for non-admin tokens.
        HTTPException: 404 if the dataset does not exist.
    """
    deleted = await DatasetRepository(db).delete(dataset_id)
    if not deleted:
        raise _404("Dataset not found.")
    logger.info("Admin %s deleted dataset %s", user.sub, dataset_id)
    return MessageResponse(message=f"Dataset {dataset_id} deleted.")


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


@router.delete(
    "/annotations/{annotation_id}",
    response_model=MessageResponse,
    summary="Delete an annotation",
    responses={
        401: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
    },
)
async def delete_annotation(
    annotation_id: UUID,
    user: Annotated[TokenData, Depends(get_current_user)] = None,
    db: AsyncSession = Depends(get_session),
) -> MessageResponse:
    """Remove the annotation with `annotation_id`.

    Args:
        annotation_id: UUID of the annotation to remove.
        user: Authenticated user from Bearer token.
        db: Injected database session.

    Returns:
        `MessageResponse` confirming deletion.

    Raises:
        HTTPException: 404 if the annotation does not exist.
    """
    deleted = await AnnotationRepository(db).delete(annotation_id)
    if not deleted:
        raise _404("Annotation not found.")
    logger.info("Deleted annotation %s by use %s", annotation_id, user.sub)
    return MessageResponse(message=f"Annotation {annotation_id} deleted.")


@router.post(
    "/annotations/{annotation_id}/confirm",
    response_model=AnnotationResponse,
    summary="Submit a corrected annotation mask",
    responses={
        401: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
    },
)
async def confirm_annotation(
    annotation_id: UUID,
    mask: UploadFile = File(..., description="Corrected segmentation mask - .tiff, single-frame only."),
    user: Annotated[TokenData, Depends(get_current_user)] = None,
    db: AsyncSession = Depends(get_session),
) -> AnnotationResponse:
    """Upload a corrected .tiff mask and mark the annotation as annotated.

    The submitted mask becomes ground truth for model retraining.

    Args:
        annotation_id: UUID of the annotation to confirm.
        mask: The corrected segmentation mask (.tiff, single-frame only).
        user: Authenticated user from Bearer token.
        db: Injected database session.

    Returns:
        The updated `AnnotationResponse`.

    Raises:
        HTTPException: 404 if the annotation does not exist.
        HTTPException: 422 if the uploaded file is not a valid single-frame TIFF.
    """
    mask_bytes = await mask.read()

    try:
        validate_tiff_mask(mask_bytes, mask.filename or "")
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=ErrorResponse(
                error_code=ErrorCode.INVALID_REQUEST,
                message=str(exc),
            ).model_dump(),
        )

    mask_url = save_annotation_mask(annotation_id, mask_bytes)
    updated = await AnnotationRepository(db).confirm(annotation_id, mask_url=mask_url)
    if updated is None:
        raise _404("Annotation not found.")

    logger.info("Confirmed annotation %s by user %s", annotation_id, user.sub)
    return updated


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


@router.delete(
    "/models/{model_id}",
    response_model=MessageResponse,
    summary="Delete a registered model (admin only)",
    responses={
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        409: {
            "model": ErrorResponse,
            "description": "Model has associated predictions",
        },
    },
)
async def delete_model(
    model_id: UUID,
    user: Annotated[TokenData, Depends(require_admin)] = None,
    db: AsyncSession = Depends(get_session),
) -> MessageResponse:
    """Remove the model version with `model_id`.

    Admin only. Returns 409 if predictions reference this model - archive
    the model instead if it needs to be removed from active use.

    Args:
        model_id: UUID of the model to delete.
        user: Admin-role authenticated user.
        db: Injected database session.

    Returns:
        `MessageResponse` confirming deletion.

    Raises:
        HTTPException: 403 for non-admin tokens.
        HTTPException: 404 if the model does not exist.
        HTTPException: 409 if predictions reference this model.
    """
    from sqlalchemy.exc import IntegrityError

    try:
        deleted = await ModelRepository(db).delete(model_id)
    except IntegrityError:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=ErrorResponse(
                error_code=ErrorCode.CONFLICT,
                message="Cannot delete - predictions reference this model. Archive instead",
            ).model_dump(),
        )

    if not deleted:
        raise _404("Model not found.")

    logger.info("Admin %s deleted model %s", user.sub, model_id)
    return MessageResponse(message=f"Model {model_id} deleted.")


@router.post(
    "/models/{model_id}/deploy",
    response_model=MessageResponse,
    summary="Deploy a registered model",
    responses={
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
    },
)
async def deploy_model(
    model_id: UUID,
    user: Annotated[TokenData, Depends(require_admin)] = None,
    db: AsyncSession = Depends(get_session),
) -> MessageResponse:
    """Trigge deployment for the model with `model_id`.

    Admin only.

    Args:
        model_id: UUID of the model to deploy.
        user: Admin-role authenticated user.
        db: Injected database session.

    Returns:
        `MessageResponse` confirming the deployment.

    Raises:
        HTTPException: 403 for non-admin tokens.
        HTTPException: 404 if the model does not exist.
    """
    if not await ModelRepository(db).exists(model_id):
        raise _404("Model not found.")

    try:
        await StubDeploymentService(db).deploy(model_id)
    except ValueError as exc:
        raise _404(str(exc))

    logger.info("Admin %s deployed model %s", user.sub, model_id)
    return MessageResponse(message=f"Model {model_id} deployment triggered.")

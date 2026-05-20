"""Library router.

Read and write endpoints for datasets, annotations, and model versions.

Read endpoints :
- GET  /datasets                            — list datasets
- GET  /datasets/{dataset_id}               — single dataset detail
- POST  /datasets/{dataset_id}/upload        - add a pending image to a dataset
- POST /datasets/{dataset_id}/run-inference - run inference on all pending images
- GET  /annotations                         — paginated annotation list
- GET  /annotations/{annotation_id}         — single annotation detail
- GET  /models                              — list all model versions
- GET  /models/{model_id}                   — single model detail (admin gets extra fields)

Write endpoints:
- DELETE /annotations/{id}                  — remove annotation
- POST   /annotations/{id}/confirm          — submit corrected mask
- DELETE /models/{id}                       — remove model version (admin only)
- POST   /models/{id}/deploy                — deploy model version (admin only)

All endpoints require a valid Bearer token. Repositories are initiated per-request
from the injected DB session.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, Query, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from api.src.core.security import get_current_user, require_admin
from api.src.db.orm import Annotation
from api.src.db.session import get_session
from api.src.dependencies import get_deployment_service, get_inference_service, get_storage
from api.src.repositories.annotations import AnnotationRepository
from api.src.repositories.datasets import DatasetRepository
from api.src.repositories.models import ModelRepository
from api.src.schemas.auth import TokenData, UserRole
from api.src.schemas.common import ErrorCode, ErrorResponse, MessageResponse
from api.src.schemas.library import (
    AnnotationResponse,
    AnnotationStatus,
    DatasetCreate,
    DatasetDetail,
    DatasetSummary,
    GrowthStage,
    ModelCreate,
    ModelDetail,
    ModelSummary,
)
from api.src.services.base import AbstractInferenceService
from api.src.services.deployment import AbstractDeploymentService
from api.src.services.mask_storage import save_annotation_mask, validate_tiff_mask
from api.src.services.storage import AbstractStorageBackend

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Library"])

# Helpers


def _404(message: str = "Resource not found.") -> HTTPException:
    """404 HTTPException helper."""
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=ErrorResponse(
            error_code=ErrorCode.NOT_FOUND,
            message=message,
        ).model_dump(),
    )


def _ext(filename: str) -> str:
    """Extract the file extension including the dot, e.g. '.png'."""
    import os

    return os.path.splitext(filename)[1].lower()


async def _run_inference_background(
    dataset_id: uuid.UUID,
    annotations: list[dict],
    inference_service: AbstractInferenceService,
    storage: AbstractStorageBackend,
    db_session_factory,
) -> None:
    """Background task: run inference on all pending annotations in a dataset.

    Iterates over pending annotations, fetches each image from storage, runs
    inference, saves the mask, and updates the annotation row.

    Partial failures are isolated - a single bad image sets that annotation
    to 'review_required' and logs the error, then processing continues with
    the remaining images.

    Args:
        dataset_id: UUID of the dataset being processed (for logging).
        annotations: List of pending Annotation rows to process as a dictionary.
        inference_service: The active inference service singleton.
        storage: The active storage backend.
        db_session_factory: Callable that returns a new AsyncSession.
    """
    succeeded = 0
    failed = 0

    logger.info(
        "run-inference background task started for dataset %s - %d image(s)",
        dataset_id,
        len(annotations),
    )

    for annotation in annotations:
        async with db_session_factory() as db:
            try:
                # Fetch image bytes from storage
                image_bytes = storage.download(annotation["image_uri"])
                filename = annotation["image_uri"].split("/")[-1]

                # Run inference
                prediction_response, mask_bytes = await inference_service.run(
                    image_bytes=image_bytes,
                    filename=filename,
                )

                # Save mask to storage if the service produced one
                mask_uri: str | None = None
                if mask_bytes is not None:
                    mask_key = f"masks/{annotation['id']}_mask.png"
                    mask_uri = storage.upload(mask_key, mask_bytes, content_type="image/png")

                # Update annotation in DB
                row = await db.get(Annotation, annotation["id"])
                if row is not None:
                    row.mask_uri = mask_uri
                    row.status = AnnotationStatus.ANNOTATED
                    row.updated_at = datetime.now(timezone.utc)
                    await db.commit()

                succeeded += 1
                logger.info(
                    "run-inference: processed annotation %s successfully",
                    annotation["id"],
                )
            except Exception as exc:
                failed += 1
                logger.error(
                    "run-inference: failed for annotation %s - %s: %s",
                    annotation["id"],
                    type(exc).__name__,
                    exc,
                )
                # Mark as review_required so it's visible in the UI
                try:
                    async with db_session_factory() as err_db:
                        row = await err_db.get(Annotation, annotation["id"])
                        if row is not None:
                            row.status = AnnotationStatus.REVIEW_REQUIRED
                            row.updated_at = datetime.now(timezone.utc)
                            await err_db.commit()
                except Exception as db_exc:
                    logger.error(
                        "run-inference: could not mark annotation %s as review_required - %s",
                        annotation["id"],
                        db_exc,
                    )

    logger.info(
        "run-inference background task completed for dataset %s - %d succeeded, %d failed",
        dataset_id,
        succeeded,
        failed,
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


@router.post(
    "/datasets/{dataset_id}/upload",
    response_model=AnnotationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload an image to a dataset",
    responses={
        401: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        412: {"model": ErrorResponse},
        415: {"model": ErrorResponse},
    },
)
async def upload_dataset_image(
    dataset_id: UUID,
    file: UploadFile = File(..., description="Root scan image. Accepted: .png, .tiff, .tif, .jpg, .jpeg. Max 50 MB."),
    user: Annotated[TokenData, Depends(get_current_user)] = None,
    storage: AbstractStorageBackend = Depends(get_storage),
    db: AsyncSession = Depends(get_session),
) -> AnnotationResponse:
    """Upload a raw image to a dataset and create a pending annotation.

    The image is saved to the storage backend and a pending annotation
    is created and linked to the dataset. The dataset count is incremented
    atomically in the same transaction.

    Args:
        dataset_id: UUID of the dataset to upload to.
        file: The image file to upload.
        user: Authenticated user from Bearer token.
        storage: Injected storage backend.
        db: Injected database session.

    Returns:
        The created AnnotationResponse with status='pending'.

    Raises:
        HTTPException: 404 if the dataset does not exist.
        HTTPException: 413 if the file exceeds 50 MB.
        HTTPException: 415 if the file format is not accepted.
    """
    from api.src.services.image_validation import (
        validate_content_type,
        validate_file_size,
    )

    dataset_repo = DatasetRepository(db)
    annotation_repo = AnnotationRepository(db)

    # Verify dataset exists
    dataset = await dataset_repo.get(dataset_id)
    if dataset is None:
        raise _404("Dataset not found.")

    # Read and validate the file
    image_bytes = await file.read()
    validate_file_size(image_bytes)
    validate_content_type(file.content_type or "", file.filename or "")

    # Save to storage
    storage_key = f"datasets/{dataset_id}/{uuid.uuid4()}{_ext(file.filename or '')}"
    try:
        image_uri = storage.upload(storage_key, image_bytes)
    except Exception as exc:
        logger.error("Storage save failed for dataset %s: %s", dataset_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ErrorResponse(
                error_code=ErrorCode.INTERNAL_ERROR,
                message="Failed to save image to storage.",
            ).model_dump(),
        )

    # Create annotation + increment count in a single transaction
    annotation = await annotation_repo.create_pending(
        dataset_id=dataset_id,
        image_uri=image_uri,
    )
    await dataset_repo.increment_count(dataset_id)
    await db.commit()

    logger.info(
        "User %s uploaded image to dataset %s - annotation %s created",
        user.sub,
        dataset_id,
        annotation.annotation_id,
    )
    return annotation


@router.post(
    "/datasets/{dataset_id}/run-inference",
    response_model=MessageResponse,
    summary="Run inference on all pending images in a dataset",
    responses={
        401: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
    },
)
async def run_dataset_inference(
    dataset_id: UUID,
    background_tasks: BackgroundTasks,
    user: Annotated[TokenData, Depends(get_current_user)] = None,
    inference_service: AbstractInferenceService = Depends(get_inference_service),
    storage: AbstractStorageBackend = Depends(get_storage),
    db: AsyncSession = Depends(get_session),
) -> MessageResponse:
    """Trigger background inference on all pending annotations in a dataset.

    Returns immediately with a count of queued images. Inference runs in
    the background - poll GET /annotations?dataset_id={id} to watch
    individual annotations flip from 'pending' to 'annotated' or
    'review_required'.

    A single image failure never aborts the batch - failed annotations are
    marked 'review_required' and the rest continue processing.

    Args:
        dataset_id: UUID of the dataset to process.
        background_tasks: FastAPI background task manager.
        user: Authenticated user from Bearer token.
        inference_service: Injected inference service singleton.
        storage: Injected storage backend.
        db: Injected database session.

    Returns:
        MessageResponse with queued count.

    Raises:
        HTTPException: 404 if the dataset does not exist.
    """
    from api.src.db.session import get_session as session_factory

    dataset = await DatasetRepository(db).get(dataset_id)
    if dataset is None:
        raise _404("Dataset not found.")

    pending = await AnnotationRepository(db).list_pending_for_dataset(dataset_id)

    if not pending:
        return MessageResponse(message="No pending annotations found for this dataset.")

    pending_data = [{"id": ann.id, "image_uri": ann.image_uri} for ann in pending]

    background_tasks.add_task(
        _run_inference_background,
        dataset_id=dataset_id,
        annotations=pending_data,
        inference_service=inference_service,
        storage=storage,
        db_session_factory=session_factory,
    )

    logger.info(
        "User %s queued inference for %d image(s) in dataset %s",
        user.sub,
        len(pending),
        dataset_id,
    )

    return MessageResponse(
        message=f"Inference started for {len(pending)} image(s). "
        f"Poll GET /annotations?dataset_id={dataset_id} for progress."
    )


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
    dataset_id: UUID | None = Query(None, description="Filter by dataset UUID"),
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
        dataset_id: Optional dataset UUID filter.
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
        dataset_id=dataset_id,
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
        409: {
            "model": ErrorResponse,
            "description": "Annotation already confirmed",
        },
        422: {
            "model": ErrorResponse,
            "description": "Uploaded file is not a valid single-frame TIFF",
        },
    },
)
async def confirm_annotation(
    annotation_id: UUID,
    mask: UploadFile = File(..., description="Corrected segmentation mask - .tiff, single-frame only."),
    user: Annotated[TokenData, Depends(get_current_user)] = None,
    storage: AbstractStorageBackend = Depends(get_storage),
    db: AsyncSession = Depends(get_session),
) -> AnnotationResponse:
    """Upload a corrected .tiff mask and mark the annotation as annotated.

    The submitted mask becomes ground truth for model retraining.

    Args:
        annotation_id: UUID of the annotation to confirm.
        mask: The corrected segmentation mask (.tiff, single-frame only).
        user: Authenticated user from Bearer token.
        storage: Injected backend storage.
        db: Injected database session.

    Returns:
        The updated `AnnotationResponse`.

    Raises:
        HTTPException: 404 if the annotation does not exist.
        HTTPException: 409 if the annotation has already been confirmed.
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

    try:
        mask_url = save_annotation_mask(
            annotation_id,
            mask_bytes,
            backend=storage,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ErrorResponse(
                error_code=ErrorCode.INTERNAL_ERROR,
                message=f"Failed to save annotation mask: {exc}",
            ).model_dump(),
        )

    try:
        updated = await AnnotationRepository(db).confirm(annotation_id, mask_url=mask_url)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=ErrorResponse(
                error_code=ErrorCode.CONFLICT,
                message="Annotation has already been confirmed.",
            ).model_dump(),
        )

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


@router.post(
    "/models",
    response_model=ModelDetail,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new model version (admin only)",
    responses={
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        409: {
            "model": ErrorResponse,
            "description": "A model with this version string already exists.",
        },
    },
)
async def register_model(
    data: ModelCreate,
    user: Annotated[TokenData, Depends(require_admin)] = None,
    db: AsyncSession = Depends(get_session),
) -> ModelDetail:
    """Register a new model version with status = 'registered'.

    The model is not deployed after registration - use
    POST /models/{id}/deploy to make it active.

    Args:
        data: Registration payload with version, name, artifact_uri,
            and optional metric fields.
        user: Admin-role authenticated user.
        db: Injected database session.

    Returns:
        The created ModelDetail with admin fields (artifact_uri, deployed_at).

    Raises:
        HTTPException: 403 for non-admin tokens.
        HTTPException: 409 if a model with the same version already exists.
    """
    try:
        model = await ModelRepository(db).create(
            version=data.version,
            name=data.name,
            artifact_uri=data.artifact_uri,
            description=data.description,
            f1_score=data.f1_score,
            precision=data.precision,
            recall=data.recall,
            inference_latency_ms=data.inference_latency_ms,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=ErrorResponse(
                error_code=ErrorCode.CONFLICT,
                message=str(exc),
            ).model_dump(),
        )

    logger.info("Admin %s registered model version '%s'", user.sub, data.version)
    return model


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
    deployment_service: AbstractDeploymentService = Depends(get_deployment_service),
) -> MessageResponse:
    """Trigge deployment for the model with `model_id`.

    Admin only.

    Args:
        model_id: UUID of the model to deploy.
        user: Admin-role authenticated user.
        db: Injected database session.
        deployment_service: Injected deployment service.

    Returns:
        `MessageResponse` confirming the deployment.

    Raises:
        HTTPException: 403 for non-admin tokens.
        HTTPException: 404 if the model does not exist.
    """
    try:
        await deployment_service.deploy(model_id)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=ErrorResponse(
                error_code=ErrorCode.INVALID_REQUEST,
                message=str(exc),
            ).model_dump(),
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=ErrorResponse(
                error_code=ErrorCode.NOT_FOUND,
                message=str(exc),
            ).model_dump(),
        )

    logger.info("Admin %s deployed model %s", user.sub, model_id)
    return MessageResponse(message=f"Model {model_id} deployed successfully.")

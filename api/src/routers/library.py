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

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from api.src.core.security import get_current_user, require_admin
from api.src.db.orm import Annotation
from api.src.db.session import get_session
from api.src.dependencies import get_deployment_service, get_storage
from api.src.repositories.annotations import AnnotationRepository
from api.src.repositories.datasets import DatasetRepository
from api.src.repositories.models import ModelRepository
from api.src.schemas.auth import TokenData, UserRole
from api.src.schemas.common import ErrorCode, ErrorResponse, MessageResponse
from api.src.schemas.library import (
    AnnotationResponse,
    AnnotationStatus,
    AnnotationUpdate,
    DatasetCreate,
    DatasetDetail,
    DatasetImportResponse,
    DatasetSummary,
    GrowthStage,
    ModelCreate,
    ModelDetail,
    ModelSummary,
)
from api.src.services.deployment import AbstractDeploymentService
from api.src.services.image_validation import validate_image
from api.src.services.mask_storage import save_annotation_mask, validate_mask
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
    return await DatasetRepository(db).list(
        species=species,
        growth_stage=growth_stage.value if growth_stage else None,
        role=user.role,
        requesting_user_id=UUID(user.sub),
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
    dataset = await repo.get(
        dataset_id,
        role=user.role,
        requesting_user_id=UUID(user.sub),
    )
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
    dataset = await DatasetRepository(db).create(
        name=data.name,
        description=data.description,
        species=data.species,
        growth_stage=data.growth_stage.value if data.growth_stage else None,
        created_by=UUID(user.sub),
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
    "/datasets/import",
    response_model=DatasetImportResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Bulk-import a dataset from a ZIP archive (admin only)",
    responses={
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        404: {"model": ErrorResponse, "description": "Target dataset not found"},
        415: {"model": ErrorResponse, "description": "Uploaded file is not a ZIP"},
    },
)
async def import_dataset(
    file: UploadFile = File(..., description="ZIP archive of root-scan images."),
    target_mode: str = Form("new", description="'new' or 'existing'"),
    dataset_id: UUID | None = Form(None, description="Required when target_mode='existing'"),
    name: str | None = Form(None),
    description: str | None = Form(None),
    species: str | None = Form(None),
    growth_stage: str | None = Form(None),
    user: Annotated[TokenData, Depends(require_admin)] = None,
    storage: AbstractStorageBackend = Depends(get_storage),
    db: AsyncSession = Depends(get_session),
) -> DatasetImportResponse:
    """Extract images from a ZIP and create one pending annotation per image.

    Reuses the single-upload pipeline (validate -> store -> dedup -> count). A bad
    entry is reported in `warnings`/`errors`. It never aborts the whole import.

    Args:
        file: ZIP archive of root-scan images.
        target_mode: 'new' or 'existing'.
        dataset_id: If target_mode='existing', UUID of the dataset.
        name: Optional name.
        description: Optional description.
        species: Optional species.
        growth_stage: Optional growth stage.
        user: Injected authenticated user.
        storage: Injected backend storage.
        db: Injected database session.

    Returns:
        DatasetImportResponse with the created/adjusted dataset.

    Raises:
        HTTPException: 403 for non-admin users.
        HTTPException: 404 if the dataset does not exist.
        HTTPException: 415 if the file is not a ZIP file.
    """
    import hashlib
    import tempfile
    import zipfile

    from sqlalchemy import select

    from api.src.services.image_validation import validate_file_size

    _IMAGE_EXT = {".png", ".tif", ".tiff", ".jpg", ".jpeg"}
    _CHUNK_SIZE = 1024 * 1024  # 1 MB streaming chunks

    dataset_repo = DatasetRepository(db)
    annotation_repo = AnnotationRepository(db, storage)

    # 1. Resolve / create the target dataset
    if target_mode == "existing":
        if dataset_id is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=ErrorResponse(
                    error_code=ErrorCode.NOT_FOUND,
                    message="dataset_id is required when target_mode='existing'.",
                ).model_dump(),
            )
        dataset = await dataset_repo.get(dataset_id)
        if dataset is None:
            raise _404("Dataset not found.")
    else:
        dataset = await dataset_repo.create(
            name=name or "Imported dataset",
            description=description,
            species=species,
            growth_stage=growth_stage,
            created_by=UUID(user.sub),
        )
        await db.commit()
        dataset_id = dataset.dataset_id

    # 2. Stream the ZIP to a temporary file instead of loading it all into RAM.
    #    This keeps memory usage constant regardless of archive size.
    try:
        tmp = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
        tmp_path = tmp.name
        while True:
            chunk = await file.read(_CHUNK_SIZE)
            if not chunk:
                break
            tmp.write(chunk)
        tmp.close()

        try:
            archive = zipfile.ZipFile(tmp_path)
        except zipfile.BadZipFile:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=ErrorResponse(
                    error_code=ErrorCode.NOT_FOUND,
                    message="Uploaded file is not a valid ZIP.",
                ).model_dump(),
            )
    except HTTPException:
        # Re-raise HTTP exceptions (like BadZipFile above)
        import os

        os.unlink(tmp_path)
        raise
    except Exception:
        import os

        os.unlink(tmp_path)
        raise

    imported = 0
    warnings: list[str] = []
    errors: list[str] = []

    # 3. One pending annotation per image, deduped within the dataset
    try:
        for entry in archive.infolist():
            if entry.is_dir():
                continue
            ext = ("." + entry.filename.rsplit(".", 1)[-1].lower()) if "." in entry.filename else ""
            if ext not in _IMAGE_EXT:
                warnings.append(f"Skipped non-image entry: {entry.filename}")
                continue
            try:
                data = archive.read(entry)
                validate_file_size(data)
                validate_image(data, entry.filename)

                image_hash = hashlib.sha256(data).hexdigest()
                existing = await db.execute(
                    select(Annotation).where(
                        Annotation.dataset_id == dataset_id,
                        Annotation.image_hash == image_hash,
                    )
                )
                if existing.scalar_one_or_none() is not None:
                    warnings.append(f"Duplicate skipped: {entry.filename}")
                    continue

                key = f"datasets/{dataset_id}/{uuid.uuid4()}/{ext}"
                image_uri = await asyncio.to_thread(storage.upload, key, data)
                await annotation_repo.create_pending(
                    dataset_id=dataset_id,
                    image_uri=image_uri,
                    image_hash=image_hash,
                    created_by=UUID(user.sub),
                )
                await dataset_repo.increment_count(dataset_id)
                imported += 1
            except Exception as exc:
                errors.append(f"{entry.filename}: {type(exc).__name__}: {exc}")
    finally:
        # Always clean up the temporary ZIP file
        archive.close()
        import os

        os.unlink(tmp_path)

    await db.commit()
    logger.info("User %s import %d image(s) into dataset %s", user.sub, imported, dataset_id)
    return DatasetImportResponse(
        dataset_id=dataset_id,
        dataset_name=dataset.name,
        scans_imported=imported,
        warnings=warnings,
        errors=errors,
    )


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
    import hashlib

    from sqlalchemy import select

    from api.src.services.image_validation import (
        validate_content_type,
        validate_file_size,
    )

    dataset_repo = DatasetRepository(db)
    annotation_repo = AnnotationRepository(db, storage)

    # Verify dataset exists
    dataset = await dataset_repo.get(dataset_id)
    if dataset is None:
        raise _404("Dataset not found.")

    # Read and validate the file
    image_bytes = await file.read()
    validate_file_size(image_bytes)
    validate_content_type(file.content_type or "", file.filename or "")
    validate_image(image_bytes, file.filename or "")

    # Save to storage
    storage_key = f"datasets/{dataset_id}/{uuid.uuid4()}{_ext(file.filename or '')}"
    try:
        image_uri = await asyncio.to_thread(storage.upload, storage_key, image_bytes)
    except Exception as exc:
        logger.error("Storage save failed for dataset %s: %s", dataset_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ErrorResponse(
                error_code=ErrorCode.INTERNAL_ERROR,
                message="Failed to save image to storage.",
            ).model_dump(),
        )

    # Check for existing duplicate in this dataset
    image_hash = hashlib.sha256(image_bytes).hexdigest()
    existing = await db.execute(
        select(Annotation).where(
            Annotation.dataset_id == dataset_id,
            Annotation.image_hash == image_hash,
        )
    )
    if existing.scalar_one_or_none() is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=ErrorResponse(
                error_code=ErrorCode.CONFLICT,
                message="An identical image already exists in this dataset.",
            ).model_dump(),
        )

    # Create annotation + increment count in a single transaction
    annotation = await annotation_repo.create_pending(
        dataset_id=dataset_id,
        image_uri=image_uri,
        image_hash=image_hash,
        created_by=UUID(user.sub),
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
    "/datasets/{dataset_id}/annotations/{annotation_id}",
    response_model=AnnotationResponse,
    summary="Add an existing annotation to a dataset",
    responses={
        401: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        409: {"model": ErrorResponse, "description": "Annotation already belongs to a dataset"},
    },
)
async def add_annotation_to_dataset(
    dataset_id: UUID,
    annotation_id: UUID,
    user: Annotated[TokenData, Depends(get_current_user)] = None,
    db: AsyncSession = Depends(get_session),
    storage: AbstractStorageBackend = Depends(get_storage),
) -> AnnotationResponse:
    """Link an existing annotation to a dataset.

    The annotation must not already belong to a dataset.

    Args:
        dataset_id: UUID of the dataset to add the annotation to.
        annotation_id: UUID of the annotation to link.
        user: Authenticated user from Bearer token.
        db: Injected database session.
        storage: Injected storage backend.

    Returns:
        Updated `AnnotationResponse` with dataset_id set.

    Raises:
        HTTPException: 404 if dataset or annotation not found.
        HTTPException: 409 if annotation already belongs to a dataset.
    """
    dataset_repo = DatasetRepository(db)
    annotation_repo = AnnotationRepository(db, storage)

    dataset = await dataset_repo.get(dataset_id)
    if dataset is None:
        raise _404("Dataset not found.")

    row = await db.get(Annotation, annotation_id)
    if row is None:
        raise _404("Annotation not found.")

    if row.dataset_id is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=ErrorResponse(
                error_code=ErrorCode.CONFLICT,
                message="Annotation already belongs to a dataset.",
            ).model_dump(),
        )

    row.dataset_id = dataset_id
    row.updated_at = datetime.now(timezone.utc)
    await dataset_repo.increment_count(dataset_id)
    await db.commit()
    await db.refresh(row)

    logger.info(
        "User %s added annotation %s to dataset %s",
        user.sub,
        annotation_id,
        dataset_id,
    )

    return await annotation_repo.get(
        annotation_id,
        role=user.role,
        requesting_user_id=UUID(user.sub),
    )


@router.post(
    "/datasets/{dataset_id}/run-inference",
    response_model=MessageResponse,
    summary="Queue inference for all pending images in a dataset",
    responses={401: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
)
async def run_dataset_inference(
    dataset_id: UUID,
    user: Annotated[TokenData, Depends(get_current_user)] = None,
    storage: AbstractStorageBackend = Depends(get_storage),
    db: AsyncSession = Depends(get_session),
) -> MessageResponse:
    """Durably queue inference for every pending annotation in a dataset.

    Cross-worker safe: a transaction-scoped advisory lock guards the
    check-then-enqueue, so two concurrent requests (or two replicas) cannot
    double-queue the same dataset. The consumer loop processes the queue.

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
    from api.src.services.inference_queue import enqueue_batch
    from api.src.services.locks import advisory_xact_lock, dataset_lock_key

    if await DatasetRepository(db).get(dataset_id) is None:
        raise _404("Dataset not found.")

    async with advisory_xact_lock(db, dataset_lock_key(dataset_id)) as got_lock:
        if not got_lock:
            return MessageResponse(message="Inference is already being queued for this dataset.")
        queued = await enqueue_batch(db, dataset_id=dataset_id, requested_by=UUID(user.sub))

    if queued == 0:
        return MessageResponse(message="No pending annotations to queue.")
    return MessageResponse(message=f"Queued {queued} image(s). Poll GET /inference/queue for progress.")


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
    status: AnnotationStatus | None = Query(None, description="Filter by annotation status"),
    dataset_id: UUID | None = Query(None, description="Filter by dataset UUID"),
    page: int = Query(default=1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(default=20, ge=1, le=100, description="Results per page"),
    order: str = Query(
        default="desc",
        pattern="^(asc|desc)$",
        description="Sort order for created_at: 'asc' or 'desc'",
    ),
    user: Annotated[TokenData, Depends(get_current_user)] = None,
    db: AsyncSession = Depends(get_session),
    storage: AbstractStorageBackend = Depends(get_storage),
) -> list[AnnotationResponse]:
    """Return a paginated list of annotations.

    Pagination is performed at the database level via SQL OFFSET/LIMIT.

    Args:
        species: Optional species name filter.
        growth_stage: Optional growth stage filter.
        status: Optional annotatino status filter.
        dataset_id: Optional dataset UUID filter.
        page: 1-indexed page number.
        page_size: Number of results per page (max 100).
        order: Sort direction for the list.
        user: Authenticated user from Bearer token.
        db: Injected database session.
        storage: Injected backend storage.

    Returns:
        Paginated list of `AnnotationResponse`.
    """
    return await AnnotationRepository(db, storage).list(
        species=species,
        growth_stage=growth_stage.value if growth_stage else None,
        status=status,
        dataset_id=dataset_id,
        page=page,
        page_size=page_size,
        role=user.role,
        requesting_user_id=UUID(user.sub),
        order=order,
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
    storage: AbstractStorageBackend = Depends(get_storage),
) -> AnnotationResponse:
    """Return the annotation with `annotation_id`.

    Args:
        annotation_id: UUID of the annotation.
        user: Authenticated user from Bearer token.
        db: Injected database session.
        storage: Injected backend storage.

    Returns:
        AnnotationResponse

    Raises:
        HTTPException: 404 if the annotation does not exist.
    """
    annotation = await AnnotationRepository(db, storage).get(
        annotation_id,
        role=user.role,
        requesting_user_id=UUID(user.sub),
    )
    if annotation is None:
        raise _404()
    return annotation


@router.post(
    "/annotations/{annotation_id}/update",
    response_model=AnnotationResponse,
    summary="Update annotation metadata",
    responses={
        401: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
    },
)
async def update_annotation(
    annotation_id: UUID,
    body: AnnotationUpdate,
    user: Annotated[TokenData, Depends(get_current_user)] = None,
    db: AsyncSession = Depends(get_session),
    storage: AbstractStorageBackend = Depends(get_storage),
) -> AnnotationResponse:
    """Update species, age_days, treatment, or growth_stage on an annotation.

    Args:
        annotation_id: UUID of the annotation to flag.
        body: Body including fields to update.
        user: Authenticated user from Bearer token.
        db: Injected database session.
        storage: Injected backend storage.

    Returns:
        AnnotationResponse

    Raises:
        HTTPException: 404 if the annotation does not exist.
    """
    updated = await AnnotationRepository(db, storage).update_metadata(
        annotation_id=annotation_id,
        species=body.species,
        age_days=body.age_days,
        treatment=body.treatment,
        growth_stage=body.growth_stage.value if body.growth_stage else None,
    )
    if updated is None:
        raise _404("Annotation not found.")
    return updated


@router.post(
    "/annotations/{annotation_id}/run-inference",
    response_model=MessageResponse,
    summary="Run inference on a single annotation",
    responses={
        401: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
    },
)
async def run_annotation_inference(
    annotation_id: UUID,
    user: Annotated[TokenData, Depends(get_current_user)] = None,
    db: AsyncSession = Depends(get_session),
) -> MessageResponse:
    """Run inference on a single annotation durably via the DB queue.

    Queues an inference job. Poll GET /annotations/{annotation_id} to watch the
    status transition from 'pending' to 'annotated'.
    """
    from api.src.db.orm import InferenceJob

    row = await db.get(Annotation, annotation_id)
    if row is None:
        raise _404("Annotation not found.")

    if row.image_uri is None:
        raise _404("Annotation has no source image.")

    # Idempotence check: is there already an active job for this annotation?
    from sqlalchemy import select

    existing = (
        await db.execute(
            select(InferenceJob).where(
                InferenceJob.annotation_id == annotation_id,
                InferenceJob.status.in_(["queued", "running"]),
            )
        )
    ).scalar_one_or_none()

    if existing is not None:
        return MessageResponse(message="Inference is already queued or running for this annotation.")

    db.add(
        InferenceJob(
            kind="single",
            annotation_id=annotation_id,
            image_uri=row.image_uri,
            status="queued",
            requested_by=UUID(user.sub),
        )
    )
    await db.commit()

    from api.src.services.broadcaster import inference_broadcaster

    await inference_broadcaster.broadcast("queue_updated")

    logger.info(
        "User %s queued inference for annotation %s",
        user.sub,
        annotation_id,
    )

    return MessageResponse(
        message=f"Inference queued for annotation {annotation_id}. Poll GET /inference/queue for progress."
    )


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
    storage: AbstractStorageBackend = Depends(get_storage),
) -> MessageResponse:
    """Remove the annotation with `annotation_id`.

    Args:
        annotation_id: UUID of the annotation to remove.
        user: Authenticated user from Bearer token.
        db: Injected database session.
        storage: Injected backend storage.

    Returns:
        `MessageResponse` confirming deletion.

    Raises:
        HTTPException: 404 if the annotation does not exist.
    """
    deleted = await AnnotationRepository(db, storage).delete(annotation_id)
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
        422: {
            "model": ErrorResponse,
            "description": "Uploaded file is not a valid single-frame TIFF or PNG",
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
    """Upload a corrected mask and mark the annotation as annotated.

    Accepts .tiff or .png. PNG inputs are converted to single-frame
    grayscale TIFF before storage so the retraining pipeline always
    receives a consistent format. Re-submitting an already-confirmed
    annotation overwrites the existing mask.

    Args:
        annotation_id: UUID of the annotation to confirm.
        mask: The corrected segmentation mask (.tiff or .png, single-frame).
        user: Authenticated user from Bearer token.
        storage: Injected backend storage.
        db: Injected database session.

    Returns:
        The updated `AnnotationResponse`.

    Raises:
        HTTPException: 404 if the annotation does not exist.
        HTTPException: 422 if the uploaded file is not a valid single-frame TIFF or PNG.
    """
    mask_bytes = await mask.read()

    try:
        validate_mask(mask_bytes, mask.filename or "")
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=ErrorResponse(
                error_code=ErrorCode.INVALID_REQUEST,
                message=str(exc),
            ).model_dump(),
        )

    try:
        mask_key = save_annotation_mask(
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
    updated = await AnnotationRepository(db, storage).confirm(annotation_id, mask_url=mask_key)
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
    deployment_service: AbstractDeploymentService = Depends(get_deployment_service),
) -> MessageResponse:
    """Trigge deployment for the model with `model_id`.

    Admin only.

    Args:
        model_id: UUID of the model to deploy.
        user: Admin-role authenticated user.
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

"""Training router.

Four admin-only endpoints for managing training runs:

- POST   /training/start          - trigger a new training run
- GET    /training/runs           - list all runs
- GET    /training/runs/{run_id}  - single run detail
- DELETE /training/runs/{run_id}  - stop a running run
"""

from __future__ import annotations

import logging
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status

from api.src.core.security import require_admin
from api.src.dependencies import get_training_service
from api.src.schemas.auth import TokenData
from api.src.schemas.common import ErrorCode, ErrorResponse, MessageResponse
from api.src.schemas.training import TrainingRunDetail, TrainingRunSummary, TrainingStartResponse
from api.src.services.training import AbstractTrainingService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/training", tags=["Training"])

# Shared 501 response used by all stubs
_NOT_IMPLEMENTED = ErrorResponse(
    error_code=ErrorCode.INTERNAL_ERROR,
    message="Not implemented yet. This endpoint will be live after Task 2097.",
)


def _501() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail=_NOT_IMPLEMENTED.model_dump(),
    )


@router.post(
    "/start",
    response_model=TrainingStartResponse,
    status_code=status.HTTP_200_OK,
    summary="Trigger a training run (admin only)",
    responses={
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        501: {
            "model": ErrorResponse,
            "description": "Not yet implemented",
        },
    },
)
async def start_training(
    body: TrainingStartResponse | None = None,
    user: Annotated[TokenData, Depends(require_admin)] = None,
    service: AbstractTrainingService = Depends(get_training_service),
) -> TrainingStartResponse:
    """Trigger a new background training run.

    Inserts a training_runs row with status='running', launches
    npec_cv training script as a subprocess, and returns the run_id
    immediately without blocking.

    Args:
        body: Optional hyperparameter overrides. Uses pipeline defaults
            when omitted.
        user: Admin-role authenticated user.
        service: Injected training service.

    Returns:
        TrainingStartResponse with the new run_id.

    Raises:
        HTTPException: 403 for non-admin tokens.
        HTTPException: 501 until Task 2097 is implemented.
    """
    try:
        return await service.start(body)
    except NotImplementedError:
        raise _501()


@router.get(
    "/runs",
    response_model=list[TrainingRunSummary],
    summary="List all training runs (admin only)",
    responses={
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        501: {
            "model": ErrorResponse,
            "description": "Not yet implemented.",
        },
    },
)
async def list_training_runs(
    user: Annotated[TokenData, Depends(require_admin)] = None,
    service: AbstractTrainingService = Depends(get_training_service),
) -> list[TrainingRunSummary]:
    """Return all training runs ordered by started_at descending.

    Args:
        user: Admin-role authenticated user.
        service: Injected training service.

    Returns:
        List of TrainingRunsummary.

    Raises:
        HTTPException: 403 for non-admin tokens.
        HTTPException: 501 until Task 2097 is implemented.
    """
    try:
        return await service.get_runs()
    except NotImplementedError:
        raise _501()


@router.get(
    "/runs/{run_id}",
    response_model=TrainingRunDetail,
    summary="Single training run detail (admin only)",
    responses={
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        501: {
            "model": ErrorResponse,
            "description": "Not yet implemented",
        },
    },
)
async def get_training_run(
    run_id: UUID,
    user: Annotated[TokenData, Depends(require_admin)] = None,
    service: AbstractTrainingService = Depends(get_training_service),
) -> TrainingRunDetail:
    """Return detail for a single training run.

    Args:
        run_id: UUID of the training run.
        user: Admin-role authenticated user.
        service: Injected training service.

    Returns:
        TrainingRunDetail.

    Raises:
        HTTPException: 403 for non-admin tokens
        HTTPException: 404 if the run does not exist.
        HTTPException: 501 until Task 2097 is implemented.
    """
    try:
        run = await service.get_run(run_id)
    except NotImplementedError:
        raise _501()

    if run is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=ErrorResponse(
                error_code=ErrorCode.NOT_FOUND,
                message="Training run not found.",
            ).model_dump(),
        )
    return run


@router.delete(
    "/runs/{run_id}",
    response_model=MessageResponse,
    summary="Stop a running training job (admin only)",
    responses={
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        409: {
            "model": ErrorResponse,
            "description": "Run is not in running state",
        },
        501: {
            "model": ErrorResponse,
            "description": "Not yet implemented",
        },
    },
)
async def stop_training_run(
    run_id: UUID,
    user: Annotated[TokenData, Depends(require_admin)] = None,
    service: AbstractTrainingService = Depends(get_training_service),
) -> MessageResponse:
    """Terminate a running training job.

    Sets the run's status to 'stopped' and terminates the background
    subprocess. Returns 409 if the run is already completed, failed,
    or stopped.

    Args:
        run_id: UUID of the training run to stop.
        user: Admin-role authenticated user.
        service: Injected training service.

    Returns:
        MessageResponse confirming the run was stopped.

    Raises:
        HTTPException: 403 for non-admin tokens.
        HTTPException: 404 if the run does not exist.
        HTTPException: 409 if the run is not in running state.
        HTTPException: 501 until Task 2099 is implemented.
    """
    try:
        found = await service.stop(run_id)
    except NotImplementedError:
        raise _501()
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=ErrorResponse(
                error_code=ErrorCode.CONFLICT,
                message=str(exc),
            ).model_dump(),
        )

    if not found:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=ErrorResponse(
                error_code=ErrorCode.NOT_FOUND,
                message="Training run not found.",
            ).model_dump(),
        )

    logger.info("Admin %s stopped training run %s", user.sub, run_id)
    return MessageResponse(message=f"Training run {run_id} stopped.")

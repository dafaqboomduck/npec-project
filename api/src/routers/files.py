"""Local file proxy router.

Exposes GET /files/{object_name:path} which streams a locally stored
artifacts as an HTTP response. This endpoint exists only to bridge the gap
when STORAGE_BACKEND=local - the frontend cannot fetch file:// URIs
over HTTP, so this proxy converts them to real HTTP responses.

When STORAGE_BACKEND=minio this endpoint is unused.
"""

from __future__ import annotations

import logging
import mimetypes

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse

from api.src.core.security import get_current_user
from api.src.dependencies import DEPLOYMENT_ENV
from api.src.schemas.auth import TokenData
from api.src.schemas.common import ErrorCode, ErrorResponse
from api.src.services.storage import LOCAL_STORAGE_BASE, STORAGE_BACKEND

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Files"])


@router.get(
    "/files/{object_name:path}",
    summary="Stram a locally stored artifact",
    description=(
        "Proxy endpoint for STORAGE_BACKEND=local. "
        "Returns 404 when MinIO is active - use the presigned URL directly."
    ),
    responses={
        401: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
    },
)
async def get_file(
    object_name: str,
    user: TokenData = Depends(get_current_user),
) -> FileResponse:
    """Stream a locally stored artifact file.

    Args:
        object_name: Relative object path, e.g. `inputs/{id}_input.png`.
        user: Authenticated user.

    Returns:
        `FileResponse` streaming the file.

    Raises:
        HTTPException: 404 if MinIO is active or the file does not exist.
    """
    if DEPLOYMENT_ENV == "cloud" or STORAGE_BACKEND != "local":
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=ErrorResponse(
                error_code=ErrorCode.NOT_FOUND,
                message="File proxy is only available when STORAGE_BACKEND=local.",
            ).model_dump(),
        )

    # Prevent path traversal
    try:
        file_path = (LOCAL_STORAGE_BASE / object_name).resolve()
        file_path.relative_to(LOCAL_STORAGE_BASE.resolve())
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=ErrorResponse(
                error_code=ErrorCode.NOT_FOUND,
                message="File not found.",
            ).model_dump(),
        )

    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=ErrorResponse(
                error_code=ErrorCode.NOT_FOUND,
                message="File not found.",
            ).model_dump(),
        )

    media_type, _ = mimetypes.guess_type(str(file_path))
    return FileResponse(
        path=str(file_path),
        media_type=media_type or "application/octet-stream",
        filename=file_path.name,
    )

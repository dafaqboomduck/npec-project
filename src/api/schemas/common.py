"""Shared response schemas used across multiple routers."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel


class ErrorCode(str, Enum):
    """Enumeration of all error codes defined in the API spec."""

    INVALID_REQUEST = "INVALID_REQUEST"
    UNAUTHORIZED = "UNAUTHORIZED"
    FORBIDDEN = "FORBIDDEN"
    NOT_FOUND = "NOT_FOUND"
    CONFLICT = "CONFLICT"
    FILE_TOO_LARGE = "FILE_TOO_LARGE"
    UNSUPPORTED_FORMAT = "UNSUPPORTED_FORMAT"
    INVALID_RESOLUTION = "INVALID_RESOLUTION"
    CORRUPT_FILE = "CORRUPT_FILE"
    MULTI_PAGE_TIFF = "MULTI_PAGE_TIFF"
    EMPTY_IMAGE = "EMPTY_IMAGE"
    INVALID_CHANNELS = "INVALID_CHANNELS"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class ErrorResponse(BaseModel):
    """Returned for all error responses."""

    status: str = "error"
    error_code: ErrorCode
    message: str
    detail: dict[str, Any] | None = None


class MessageResponse(BaseModel):
    """Generic success message response."""

    message: str

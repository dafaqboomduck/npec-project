"""FastAPI application factory.

This module creates and configures the FastAPI instance. It is
the entry point for uvicorn:

    uvicorn src.api.main:app --reload

All routers are registered here.
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src.api.config import settings
from src.api.routers import auth, health, inference, library
from src.api.schemas.common import ErrorCode, ErrorResponse

# Logging

logging.basicConfig(level=logging.INFO, format="$(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)


# Lifespan


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application startup and shutdown.

    Startup tasks run before `yield`; shutdown tasks run after.
    Future uses include opening a DB connection pool and pre-loading
    model weights.
    """
    logger.info("ROOTR API starting up - docs at /docs")
    yield
    logger.info("ROOTR API shutting down")


# Application

app = FastAPI(
    title="ROOTR API",
    description=(
        "REST API for the NPEC plant root segmentation service. "
        "Accepts root scan images, runs U-Net interference, and returns "
        "segmentation masks, tip locations, and confidence scores."
    ),
    version=settings.api_version,
    docs_url="/docs",
    redoc_url="/redocs",
    lifespan=lifespan,
)


# Exception handlers
#  FastAPI wraps HTTPException.detail under a "detail" key by default.
#  We override this so that error bodies are flat ErrorResponse objects.


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Serialise HTTPException.detail directly as the response body.

    All routers raise `HTTPException(detail=ErrorResponse(...).model_dump())`.
    This handler ensures taht dict is returned as-is rather than being nested
    under a `"detail"` key.

    Args:
        request: The incoming request.
        exc: The raised `HTTPException`.

    Returns:
        `responses.JSONResponse` whose body is the
        `detail` dict.
    """
    if isinstance(exc.detail, dict):
        content = exc.detail
    else:
        content = ErrorResponse(
            error_code=ErrorCode.UNAUTHORIZED if exc.status_code == 401 else ErrorCode.INTERNAL_ERROR,
            message=str(exc.detail),
        ).model_dump()

    return JSONResponse(status_code=exc.status_code, content=content)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Convert Pydantic validation errors to our ErrorResponse shape.

    Args:
        request: The incoming request.
        exc: The validation error raised by Pydantic/FastAPI.

    Returns:
        HTTP 422 `responses.JSONResponse`.
    """
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        content=ErrorResponse(
            error_code=ErrorCode.INVALID_REQUEST,
            message="Request validation failed.",
            detail={"errors": exc.errors()},
        ).model_dump(),
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all handler that prevents raw tracebacks leaking to the client.

    Args:
        request: The incoming request.
        exc: The unhandled exception.

    Returns:
        HTTP 500 `responses.JSONResponse`.
    """
    logger.exception("Unhandled exception on %s %s", request.method, request.url)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error_code=ErrorCode.INTERNAL_ERROR,
            message="An unexpected error occurred. Please try again later.",
        ).model_dump(),
    )


# Routers - all mounted under /api/v1

API_PREFIX = "/api/v1"

app.include_router(health.router, prefix=API_PREFIX)
app.include_router(auth.router, prefix=API_PREFIX)
app.include_router(inference.router, prefix=API_PREFIX)
app.include_router(library.router, prefix=API_PREFIX)

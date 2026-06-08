"""FastAPI application factory.

This module creates and configures the FastAPI instance. It is
the entry point for uvicorn:

    uvicorn api.main:app --reload

All routers are registered here.
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.src.config import settings
from api.src.db.session import get_sessionmaker
from api.src.dependencies import shutdown_inference_service, warmup_inference_service
from api.src.routers import (
    auth,
    dashboard,
    feedback,
    files,
    health,
    inference,
    library,
    overview,
    training,
)
from api.src.schemas.common import ErrorCode, ErrorResponse
from api.src.services.inference_queue import consume_forever, reconcile_orphaned_jobs
from api.src.services.training import reconcile_orphaned_training_runs

# Logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

origins = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")

# Lifespan


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application startup and shutdown.

    Startup tasks run before `yield`; shutdown tasks run after.
    Future uses include opening a DB connection pool and pre-loading
    model weights.
    """
    logger.info("CV5 API starting up - docs at /docs")
    await warmup_inference_service()
    sm = get_sessionmaker()
    await reconcile_orphaned_jobs(sm)
    await reconcile_orphaned_training_runs()

    stop = asyncio.Event()
    asyncio.create_task(consume_forever(sm, stop), name="inference-consumer")

    yield
    await shutdown_inference_service()
    logger.info("CV5 API shutting down")


# Application

app = FastAPI(
    title="CV5 API",
    description=(
        "REST API for the NPEC plant root segmentation service. "
        "Accepts root scan images, runs U-Net inference, and returns "
        "segmentation masks, and tip locations"
    ),
    version=settings.api_version,
    docs_url="/docs",
    redoc_url="/redocs",
    lifespan=lifespan,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request body size limiter — reject oversized uploads before reading into memory.
# The per-route validation (validate_file_size) enforces the precise 50 MB limit;
# this middleware sets a generous ceiling (100 MB to allow for multipart overhead)
# so that truly abusive payloads never allocate memory.
MAX_REQUEST_BODY_BYTES = 100 * 1024 * 1024  # 100 MB


@app.middleware("http")
async def limit_request_body(request: Request, call_next):
    """Middleware to reject requests exceeding the maximum allowed body size."""
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_REQUEST_BODY_BYTES:
        limit_mb = MAX_REQUEST_BODY_BYTES // (1024 * 1024)
        return JSONResponse(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            content=ErrorResponse(
                error_code=ErrorCode.INVALID_REQUEST,
                message=f"Request body too large. Max size is {limit_mb} MB.",
            ).model_dump(),
        )
    return await call_next(request)


# Exception handlers
#  FastAPI wraps HTTPException.detail under a "detail" key by default.
#  We override this so that error bodies are flat ErrorResponse objects.


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Serialise HTTPException.detail directly as the response body.

    All routers raise `HTTPException(detail=ErrorResponse(...).model_dump())`.
    This handler ensures that dict is returned as-is rather than being nested
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
app.include_router(overview.router, prefix=API_PREFIX)
app.include_router(library.router, prefix=API_PREFIX)
app.include_router(inference.router, prefix=API_PREFIX)
app.include_router(feedback.router, prefix=API_PREFIX)
app.include_router(dashboard.router, prefix=API_PREFIX)
app.include_router(files.router, prefix=API_PREFIX)
app.include_router(training.router, prefix=API_PREFIX)

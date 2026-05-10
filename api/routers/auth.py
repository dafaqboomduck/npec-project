"""Authentication router.

Exposes three endpoints:

- POST /auth/token   — obtain access + refresh tokens (OAuth2 Password flow)
- POST /auth/refresh — rotate refresh token, get new access token
- POST /auth/logout  — invalidate current session
"""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession

from db.session import get_session

from ..core.security import (
    ACCESS_TOKEN_EXPIRE_SECONDS,
    create_access_token,
    get_current_user,
    verify_password,
)
from ..repositories.users import UserRepository
from ..schemas.auth import AuthResponse, TokenData, TokenRefreshRequest, UserRole
from ..schemas.common import ErrorCode, ErrorResponse, MessageResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["Auth"])

_INVALID_CREDENTIALS = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail=ErrorResponse(
        error_code=ErrorCode.UNAUTHORIZED,
        message="Invalid username or password.",
    ).model_dump(),
    headers={"WWW-Authenticate": "Bearer"},
)


@router.post(
    "/token",
    response_model=AuthResponse,
    summary="Obtain a JWT access token",
    responses={
        401: {"model": ErrorResponse, "description": "Invalid credentials"},
        422: {"model": ErrorResponse, "description": "Missing fields"},
    },
)
async def login(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
    db: AsyncSession = Depends(get_session),
) -> AuthResponse:
    """Authenticate with email and password, return access and refresh tokens.

    Args:
        form_data: OAuth2 form fields (username, password, grant_type).
        db: Injected database session.

    Returns:
        AuthResponse with both tokens.

    Raises:
        HTTPException: 401 for invalid credentials.
    """
    repo = UserRepository(db)
    user = await repo.get_by_email(form_data.username)

    if not user or not verify_password(form_data.password, user.password_hash):
        logger.warning("Failed login attempt for %s", form_data.username)
        raise _INVALID_CREDENTIALS

    access_token = create_access_token(
        subject=str(user.id),
        role=UserRole(user.role),
    )
    refresh_token = await repo.issue_refresh_token(user.id)

    logger.info("Issued tokens for '%s' (role=%s)", user.email, user.role)

    return AuthResponse(
        access_token=access_token,
        expires_in=ACCESS_TOKEN_EXPIRE_SECONDS,
        refresh_token=refresh_token,
        role=UserRole(user.role),
    )


@router.post(
    "/refresh",
    response_model=AuthResponse,
    summary="Refresh access token",
    responses={
        401: {"model": ErrorResponse, "description": "Refresh token invalid or expired"},
    },
)
async def refresh(
    body: TokenRefreshRequest,
    db: AsyncSession = Depends(get_session),
) -> AuthResponse:
    """Rotate the refresh token and return a new access + refresh token pair.

    The old refresh token is invalidated immediately.  The new access token
    carries the correct role fetched from the database.

    Args:
        body: Request body containing the current refresh token.
        db: Injected database session.

    Returns:
        A new `AuthResponse`.

    Raises:
        HTTPException: 401 if the refresh token is invalid or already used.
    """
    repo = UserRepository(db)
    result = await repo.rotate_refresh_token(body.refresh_token)

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ErrorResponse(
                error_code=ErrorCode.UNAUTHORIZED,
                message="Refresh token is invalid or has already been used.",
            ).model_dump(),
            headers={"WWW-Authenticate": "Bearer"},
        )

    new_raw_token, user_id = result
    user = await repo.get_by_id(user_id)

    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ErrorResponse(
                error_code=ErrorCode.UNAUTHORIZED,
                message="User account is inactive.",
            ).model_dump(),
        )

    access_token = create_access_token(
        subject=str(user.id),
        role=UserRole(user.role),
    )

    logger.info("Rotated refresh token for user '%s'", user.email)

    return AuthResponse(
        access_token=access_token,
        expires_in=ACCESS_TOKEN_EXPIRE_SECONDS,
        refresh_token=new_raw_token,
        role=UserRole(user.role),
    )


@router.post(
    "/logout",
    response_model=MessageResponse,
    summary="Invalidate the current session",
    responses={
        401: {"model": ErrorResponse, "description": "Missing or invalid token"},
    },
)
async def logout(
    user: Annotated[TokenData, Depends(get_current_user)],
    body: TokenRefreshRequest,
    db: AsyncSession = Depends(get_session),
) -> MessageResponse:
    """Revoke the refresh token and end the session.

    Requires a valid Bearer token and a refresh token in the
    body which is going to be revoked so it cannot be used again.

    Args:
        user: Authenticated user from the Bearer token.
        body: Body containing the refresh token to revoke.
        db: Injected database session.

    Returns:
        MessageResponse confirming logout.
    """
    repo = UserRepository(db)
    await repo.revoke_refresh_token(body.refresh_token)

    logger.info("User '%s' logged out", user.sub)
    return MessageResponse(message="Logged out successfully.")

"""JWT security utilities and injectable FastAPI dependencies.

Injectable dependencies:
- get_current_user  - validates Bearer token, returns TokenData
- require_admin     - extends get_current_user, enforces admin role
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.ext.asyncio import AsyncSession

from api.src.config import settings
from api.src.db.session import get_session
from api.src.schemas.auth import TokenData, UserRole
from api.src.schemas.common import ErrorCode, ErrorResponse

logger = logging.getLogger(__name__)

# Config
SECRET_KEY: str = settings.secret_key
ALGORITHM: str = settings.algorithm
ACCESS_TOKEN_EXPIRE_SECONDS: int = 3600  # 1h

# Password hashing

_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(plain: str) -> str:
    """Return a bcrypt hash of `plain`.

    Args:
        plain: The plaintext password.

    Returns:
        A bcrypt hash string safe to store in the database.
    """
    return _pwd_context.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    """Return `True` if `plain` matches `hashed`.

    Args:
        plain: The plaintext password to check.
        hashed: The stored bcrypt hash from the db.

    Returns:
        `True` if the password matches, `False` otherwise.
    """
    return _pwd_context.verify(plain, hashed)


# JWT access tokens


def create_access_token(subject: str, role: UserRole) -> str:
    """Create a signed JWT acess token.

    Args:
        subject: The user's UUID (as string) embedded as the `sub` claim.
        role: The user's role embedded as a custom claim.

    Returns:
        A signed JWT string.
    """
    now = datetime.now(timezone.utc)
    payload = {
        "sub": subject,
        "role": role.value,
        "iat": now,
        "exp": now + timedelta(seconds=ACCESS_TOKEN_EXPIRE_SECONDS),
        "jti": str(uuid.uuid4()),  # unique token ID
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_access_token(token: str) -> TokenData:
    """Decode and validate a JWT access token.

    Args:
        token: Raw JWT string from the Authorization header.

    Returns:
        TokenData with `sub` and `role`.

    Raises:
        HTTPException: 401 if the token is missing, invalid, or expired.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=ErrorResponse(
            error_code=ErrorCode.UNAUTHORIZED,
            message="Could not validate credentials.",
        ).model_dump(),
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        sub: str | None = payload.get("sub")
        role_str: str | None = payload.get("role")
        if sub is None or role_str is None:
            raise credentials_exception
        return TokenData(sub=sub, role=UserRole(role_str))
    except JWTError:
        raise credentials_exception


# FastAPI OAuth2 scheme + dependencies

_oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/token")


async def get_current_user(
    token: Annotated[str, Depends(_oauth2_scheme)],
    db: Annotated[AsyncSession, Depends(get_session)],
) -> TokenData:
    """Validate the Bearer token and return the current user's claims.

    Inject this on any protected endpoint:

        @router.get("/protected")
        async def protected(user: Annotated[TokenData, Depends(get_current_user)]):
            ...

    Args:
        token: JWT extracted from the Authorization header by FastAPI.
        db: AsyncSession for database queries.

    Returns:
        TokenData for the authenticated user.

    Raises:
        HTTPException: 401 if the token is missing, invalid, or expired.
    """
    from api.src.repositories.users import UserRepository

    token_data = decode_access_token(token)
    user = await UserRepository(db).get_by_id(uuid.UUID(token_data.sub))
    if user is None or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ErrorResponse(
                error_code=ErrorCode.UNAUTHORIZED,
                message="Could not validate credentials.",
            ).model_dump(),
            headers={"WWW-Authenticate": "Bearer"},
        )
    return token_data


def require_admin(
    user: Annotated[TokenData, Depends(get_current_user)],
) -> TokenData:
    """Require the `admin` role on top of a valid Bearer token.

    Inject this on admin-only endpoints:

        @router.delete("/models/{model_id}")
        async def delete_model(user: Annotated[TokenData, Depends(require_admin)]):
            ...

    Args:
        user: Authenticated user, provided by `get_current_user`.

    Returns:
        The same `TokenData` if the user is admin.

    Raises:
        HTTPException: 403 if the user's role is not `admin`.
    """
    if user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=ErrorResponse(
                error_code=ErrorCode.FORBIDDEN,
                message="Admin access required.",
            ).model_dump(),
        )
    return user

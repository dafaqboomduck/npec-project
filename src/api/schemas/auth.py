"""Authentication schemas."""

from enum import Enum

from pydantic import BaseModel, Field


class UserRole(str, Enum):
    """User role."""

    USER = "user"
    ADMIN = "admin"


class AuthRequest(BaseModel):
    """OAuth2 Password flow login request - submitted as form data."""

    username: str = Field(..., max_length=254, description="User email address")
    password: str = Field(..., max_length=128)
    grant_type: str = Field(..., pattern="^password$")


class AuthResponse(BaseModel):
    """Returned on successful login or token referesh."""

    access_token: str
    token_type: str = "bearer"
    expired_in: int = 3600
    refresh_token: str
    role: UserRole


class TokenRefreshRequest(BaseModel):
    """Body for POST /auth/refresh."""

    refresh_token: str = Field(..., min_length=1)


class TokenData(BaseModel):
    """Claims extracted from a decoded JWT access token."""

    sub: str
    role: UserRole

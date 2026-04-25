"""User and refresh token repository."""

import hashlib
import logging
import secrets
import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from db.orm import RefreshToken, User

logger = logging.getLogger(__name__)

REFRESH_TOKEN_EXPIRE_SECONDS: int = 86400  # 1d


def _hash_token(token: str) -> str:
    """Return the SHA-256 hex digest of `token`.

    Args:
        token: The opaque refresh token string.

    Returns:
        Hex-encoded SHA-256 digest.
    """
    return hashlib.sha256(token.encode()).hexdigest()


class UserRepository:
    """Database operations for `users` and `refresh_tokens`."""

    def __init__(self, db: AsyncSession) -> None:
        """Database operations for `users` and `refresh_tokens`."""
        self._db = db

    # Users

    async def get_by_email(self, email: str) -> User | None:
        """Return the active user with `email`, or `None`.

        Args:
            email: Email address to look up.

        Returns:
            A `db.orm.User` if found and active, otherwise `None`.
        """
        result = await self._db.execute(
            select(User).where(
                User.email == email,
                User.is_active.is_(True),
            )
        )
        return result.scalar_one_or_none()

    async def get_by_id(self, user_id: uuid.UUID) -> User | None:
        """Return the user with `user_id`, or `None`.

        Args:
            user_id: UUID to look up.

        Returns:
            A `db.orm.User` if found, otherwise `None`.
        """
        return await self._db.get(User, user_id)

    # Refresh tokens

    async def issue_refresh_token(self, user_id: uuid.UUID) -> str:
        """Generate a new refresh token and persist its hash.

        Args:
            user_id: UUID of the user this token belongs to.

        Returns:
            The raw (unhashed) token string to return to the client.
        """
        raw = secrets.token_urlsafe(32)
        record = RefreshToken(
            user_id=user_id,
            token_hash=_hash_token(raw),
            expires_at=datetime.now(timezone.utc) + timedelta(seconds=REFRESH_TOKEN_EXPIRE_SECONDS),
        )
        self._db.add(record)
        await self._db.commit()
        logger.debug("Issued refresh token for user %s", user_id)
        return raw

    async def rotate_refresh_token(self, raw_token: str) -> tuple[str, uuid.UUID] | None:
        """Invalidate `raw_token` and issue a replacement.

        Checks the token is valid (not revoked, not expired), then sets
        `revoked_at` and `replaced_by` on the old row before creating
        a new one.

        Args:
            raw_token: The opaque token string sent by the client.

        Returns:
            `(new_raw_token, user_id)` on success, or `None` if the
            token was not valid.
        """
        token_hash = _hash_token(raw_token)
        now = datetime.now(timezone.utc)

        result = await self._db.execute(
            select(RefreshToken).where(
                RefreshToken.token_hash == token_hash,
                RefreshToken.revoked_at.is_(None),
                RefreshToken.expires_at > now,
            )
        )
        old = result.scalar_one_or_none()
        if old is None:
            return None

        new_raw = secrets.token_urlsafe(32)
        new_record = RefreshToken(
            user_id=old.user_id,
            token_hash=_hash_token(new_raw),
            expires_at=now + timedelta(seconds=REFRESH_TOKEN_EXPIRE_SECONDS),
        )
        self._db.add(new_record)
        await self._db.flush()  # populate new_record.id before committing

        old.revoked_at = now
        old.replaced_by = new_record.id
        await self._db.commit()

        logger.debug("Rotated refresh token for use %s", old.user_id)
        return new_raw, old.user_id

    async def revoke_refresh_token(self, raw_token: str) -> None:
        """Mark `raw_token` as revoked (used on logout).

        Args:
            raw_token: The opaque token string send by the client.
        """
        await self._db.execute(
            update(RefreshToken)
            .where(
                RefreshToken.token_hash == _hash_token(raw_token),
                RefreshToken.revoked_at.is_(None),
            )
            .values(revoked_at=datetime.now(timezone.utc))
        )
        await self._db.commit()
        logger.debug("Revoked refresh token")

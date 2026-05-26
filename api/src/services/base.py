"""Abstract inference service interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from api.src.schemas.predictions import PredictionResponse


class AbstractInferenceService(ABC):
    """Interface for running inference on an uploaded image."""

    @abstractmethod
    async def run(
        self,
        image_bytes: bytes,
        filename: str,
        *,
        db: Optional[AsyncSession] = None,
    ) -> tuple[PredictionResponse, object | None]:
        """Run inference on image_bytes.

        Args:
            image_bytes: Raw bytes of the uploaded image.
            filename: Original filename.
            db: Optional async DB session. When provided, the
                implementation may write an audit row to
                ``inference_audit_log`` as part of the request's
                transaction. Pass None to disable audit logging
                (e.g. in tests).

        Returns:
            A fully populated `PredictionResponse`.

        Raises:
            ValueError: If the image cannot be decoded or is otherwise invalid.
        """

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

    async def warmup(self) -> None:
        """Pre-load any heavy resources before the first request.

        Optional lifecycle hook called once at server startup. The default is a
        no-op; implementations that load models (e.g. the U-Net pipeline) may
        override it to move cold-start cost off the first request.
        """
        return None

    async def aclose(self) -> None:
        """Release resources at server shutdown.

        Optional lifecycle hook called once at server shutdown. The default is a
        no-op; implementations that hold processes or GPU contexts may override
        it to tear them down cleanly.
        """
        return None

"""Monitoring service.

Writes per-prediction metrics to monitoring_metrics table after
each successful inference call.
"""

import logging
import uuid
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.orm import ModelVersion, MonitoringMetric
from src.api.schemas.library import ModelStatus

logger = logging.getLogger(__name__)

# Prometheus-style label applied to all prediction metrics
_PREDICT_LABELS = {"endpoint": "/predict"}


class MonitoringService:
    """Writes operational metrics to `monitoring_metrics`."""

    def __init__(self, db: AsyncSession) -> None:
        """Writes operational metrics to `monitoring_metrics`.

        Args:
            db: Injected database session.
        """
        self._db = db

    async def _get_deployed_model_version_id(self) -> uuid.UUID | None:
        """Return the UUID of the deployed model version, or `None`.

        Returns:
            UUID of the deployed `ModelVersion` row, or `None` if no
            deployed model exists.
        """
        result = await self._db.execute(
            select(ModelVersion.id).where(ModelVersion.status == ModelStatus.DEPLOYED.value)
        )
        return result.scalar_one_or_none()

    async def record_prediction_metrics(
        self,
        prediction_id: uuid.UUID,
        processing_time_ms: int,
        overall_confidence: float,
    ) -> None:
        """Write latency and confidence metrics for a completed prediction.

        Inserts two rows into `monitoring_metrics`:
        - `prediction_latency_ms`   - wall-clock inference time in milliseconds
        - `overall_confidence`      - mean per-plant confidence score (0-1)

        Both rows carry `{"endpoint": "/predict", "prediction_id": "<uuid>"}`
        labels so they can be filtered per-prediction in the dashboard.

        Args:
            prediction_id: UUID of the prediction.
            processing_time_ms: Inference time in milliseconds.
            overall_confidence: Mean per-plant confidence score (0.0-1.0).
        """
        model_version_id = await self._get_deployed_model_version_id()
        now = datetime.now(timezone.utc)
        labels = {**_PREDICT_LABELS, "prediction_id": str(prediction_id)}

        rows = [
            MonitoringMetric(
                metric_name="prediction_latency_ms",
                metric_value=float(processing_time_ms),
                model_version_id=model_version_id,
                labels=labels,
                recorded_at=now,
            ),
            MonitoringMetric(
                metric_name="overall_confidence",
                metric_value=overall_confidence,
                model_version_id=model_version_id,
                labels=labels,
                recorded_at=now,
            ),
        ]

        for row in rows:
            self._db.add(row)

        await self._db.commit()
        logger.debug(
            "Recorded metrics for prediction %s - latency=%dms confidence=%.4f",
            prediction_id,
            processing_time_ms,
            overall_confidence,
        )

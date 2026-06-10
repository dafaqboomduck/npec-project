"""Dashboard repository.

Tables read:
- `model_versions`     - deployed model ID, version string, f1_score
- `monitoring_metrics` - latest f1, throughput, and f1_validation history
- `annotations`        - species distribution via GROUP BY

Metric name conventions (must match what MonitoringService writes):
    `prediction_latency_ms`  - inference wall-clock time
    `throughput_img_per_s`   - images processed per second (written by training pipeline)
    `f1_validation`          - per-epoch validation F1 (written by training pipeline)
    `drift_score`            - data drift score (written by drift detection job)
"""

from __future__ import annotations

import logging
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from api.src.db.orm import Annotation, ModelVersion, MonitoringMetric
from api.src.schemas.dashboard import (
    DashboardMetricsResponse,
    DataDistributionResponse,
    F1HistoryPoint,
    F1HistoryResponse,
    SpeciesCount,
)
from api.src.schemas.library import AnnotationStatus, ModelStatus

logger = logging.getLogger(__name__)


class DashboardRepository:
    """Read-only queries for the three Dashboard enpdoints."""

    def __init__(self, db: AsyncSession) -> None:
        """Read-only queries for the three Dashboard enpdoints.

        Args:
            db: Injected database session.
        """
        self._db = db

    async def _get_deployed_model(self) -> tuple[object, str] | None:
        """Return `(model_id, version_string)` for the deployed model.

        Returns `None` if no deployed model exists.
        """
        result = await self._db.execute(
            select(ModelVersion.id, ModelVersion.version, ModelVersion.f1_score)
            .where(ModelVersion.status == ModelStatus.DEPLOYED.value)
            .limit(1)
        )
        return result.first()

    async def get_metrics(self, run_id: UUID | None = None) -> DashboardMetricsResponse:
        """Return top-level KPIs for the deployed model.

        Reads the latest `f1_validation` and `throughput_img_per_s`
        rows from `monitoring_metrics`. Falls back to model's
        registered `f1_score` column when no `f1_validation` metric
        exists yet.

        Returns sensible zero defaults when no deployed model or metrics exists.

        Args:
            run_id: Optional UUID of the run to filter on.

        Returns:
            `DashboardMetricsResponse`
        """
        deployed = await self._get_deployed_model()

        if deployed is None:
            logger.warning("get_metrics: no deployed model.")
            return DashboardMetricsResponse(f1_score=0.0, throughput_img_per_s=0.0)

        model_id, version, model_f1 = deployed

        # F1 score — always use the best epoch, not the last
        # When run_id given: best epoch of that specific run
        # When no run_id: best epoch ever recorded for the deployed model
        if run_id is not None:
            f1_query = (
                select(func.max(MonitoringMetric.metric_value))
                .where(MonitoringMetric.metric_name == "f1_validation")
                .where(MonitoringMetric.labels["run_id"].astext == str(run_id))
            )
        else:
            f1_query = (
                select(func.max(MonitoringMetric.metric_value))
                .where(MonitoringMetric.metric_name == "f1_validation")
                .where(MonitoringMetric.model_version_id == model_id)
            )

        f1_result = await self._db.execute(f1_query)
        f1_score = f1_result.scalar_one_or_none()
        if f1_score is None:
            f1_score = float(model_f1 or 0.0)

        # Throughput — calculate dynamically: round(1000.0 / avg(prediction_latency_ms), 3)
        tp_query = select(func.avg(MonitoringMetric.metric_value)).where(
            MonitoringMetric.metric_name == "prediction_latency_ms"
        )
        if run_id is not None:
            tp_query = tp_query.where(MonitoringMetric.labels["run_id"].astext == str(run_id))
        else:
            tp_query = tp_query.where(MonitoringMetric.model_version_id == model_id)

        throughput_result = await self._db.execute(tp_query)
        avg_latency = throughput_result.scalar_one_or_none()
        if avg_latency and float(avg_latency) > 0:
            throughput = round(1000.0 / float(avg_latency), 3)
        else:
            throughput = 0.0

        return DashboardMetricsResponse(
            f1_score=round(float(f1_score), 4),
            throughput_img_per_s=throughput,
        )

    async def get_f1_history(
        self,
        run_id: UUID | None = None,
    ) -> F1HistoryResponse:
        """Return the full f1_validation time-series for the deployed model.

        Reads all `f1_validation` rows for the deployed model from
        `monitoring_metrics`, ordered by epoch ascending so the frontend
        can plot them directly without any client-side sorting.

        The epoch is stored in the `labels` JSONB column as
        `{"epoch": N, ...}`. Rows without an epoch label are skipped.

        Returns empty defaults when no deployed model or history exists.

        Args:
            run_id: Optional UUID of the run to filter on.

        Returns:
            `F1HistoryResponse`
        """
        deployed = await self._get_deployed_model()

        if deployed is None:
            logger.warning("get_f1_history: no deployed model.")
            return F1HistoryResponse(model_version="unknown", data_points=[])

        model_id, version, _ = deployed

        query = select(
            MonitoringMetric.metric_value,
            MonitoringMetric.labels,
        ).where(
            MonitoringMetric.metric_name == "f1_validation",
        )

        if run_id is not None:
            query = query.where(MonitoringMetric.labels["run_id"].astext == str(run_id))
        else:
            query = query.where(
                MonitoringMetric.model_version_id == model_id,
            )

        query = query.order_by(MonitoringMetric.recorded_at.asc())

        result = await self._db.execute(query)
        rows = result.all()

        data_points: list[F1HistoryPoint] = []
        for metric_value, labels in rows:
            epoch = labels.get("epoch") if labels else None
            if epoch is None:
                continue
            data_points.append(
                F1HistoryPoint(
                    epoch=int(epoch),
                    f1_score=round(float(metric_value), 4),
                )
            )

        # Sort by epoch ascending in case rows arrived out of order
        data_points.sort(key=lambda p: p.epoch)

        return F1HistoryResponse(
            model_version=version,
            data_points=data_points,
        )

    async def get_data_distribution(self) -> DataDistributionResponse:
        """Return confirmed sample counts grouped by species.

        Only species with at least one confirmed sample are included.

        Returns:
            `DataDistributionResponse`
        """
        result = await self._db.execute(
            select(
                func.lower(Annotation.species).label("species_lower"),
                func.count(Annotation.id).label("count"),
            )
            .where(
                Annotation.species.is_not(None),
                Annotation.status == AnnotationStatus.ANNOTATED.value,
            )
            .group_by(func.lower(Annotation.species))
            .order_by(func.count(Annotation.id).desc())
        )
        rows = result.all()
        return DataDistributionResponse(
            species_counts=[
                SpeciesCount(species=species_lower.capitalize(), count=count) for species_lower, count in rows
            ]
        )

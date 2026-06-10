/**
 * pages/Dashboard.tsx
 *
 * Dedicated Dashboard Analytics workspace page.
 * Visualizes F1 history trend curves, pipeline throughput, and species counts.
 * Uses a zero-dependency high-fidelity Custom SVG + CSS Chart Engine.
 *
 * Handles empty database/offline states gracefully showing standard fallback indicators.
 * Route: /dashboard (protected)
 */

import { useEffect, useState } from 'react';
import NavBar from '../components/NavBar';
import {
  getDashboardMetrics,
  getF1History,
  getSpeciesDistribution,
} from '../api/dashboardApi';
import type {
  DashboardMetricsResponse,
  F1HistoryResponse,
  DataDistributionResponse,
  TrainingRunSummary,
  F1HistoryPoint,
} from '../types/api';
import './Dashboard.css';
import { listTrainingRuns } from '../api/trainingApi';

// Standard 600x240 responsive viewBox coordinates for F1 chart
const VIEW_W = 600;
const VIEW_H = 240;
const PAD_L = 50;
const PAD_R = 20;
const PAD_T = 20;
const PAD_B = 40;
const CHART_W = VIEW_W - PAD_L - PAD_R;
const CHART_H = VIEW_H - PAD_T - PAD_B;

export default function Dashboard() {
  // ── Data State ──
  const [metrics, setMetrics] = useState<DashboardMetricsResponse | null>(null);
  const [f1History, setF1History] = useState<F1HistoryResponse | null>(null);
  const [distribution, setDistribution] =
    useState<DataDistributionResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [runs, setRuns] = useState<TrainingRunSummary[]>([]);
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);

  // ── Interactive Hover State ──
  const [hoveredPoint, setHoveredPoint] = useState<
    (F1HistoryPoint & { x: number; y: number }) | null
  >(null);

  useEffect(() => {
    fetchDashboardData();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function fetchDashboardData() {
    return fetchDashboardDataForRun(selectedRunId);
  }

  async function fetchDashboardDataForRun(runId: string | null) {
    setLoading(true);
    setError(null);
    try {
      const [metricsRes, f1Res, distRes, runsRes] = await Promise.all([
        getDashboardMetrics(runId ?? undefined).catch(() => null),
        getF1History(runId ?? undefined).catch(() => null),
        getSpeciesDistribution().catch(() => null),
        listTrainingRuns().catch(() => []),
      ]);

      setRuns(
        runsRes.filter((r) => r.status === 'completed' || r.status == 'failed'),
      );
      setMetrics(metricsRes);
      setF1History(f1Res);
      setDistribution(distRes);
    } catch {
      setError('Could not establish connection to the analytics server.');
    } finally {
      setLoading(false);
    }
  }

  // ── Chart Mathematics (Line Chart) ──
  const rawPoints = f1History?.data_points || [];
  const minEpoch =
    rawPoints.length > 0 ? Math.min(...rawPoints.map((p) => p.epoch)) : 0;
  const maxEpoch =
    rawPoints.length > 0 ? Math.max(...rawPoints.map((p) => p.epoch)) : 50;
  const rangeEpoch = maxEpoch - minEpoch || 1;

  const points = rawPoints.map((p) => {
    const x = PAD_L + ((p.epoch - minEpoch) / rangeEpoch) * CHART_W;
    const y = PAD_T + (1.0 - p.f1_score) * CHART_H; // standard F1 is 0.0 - 1.0 (inverted in SVG)
    return { ...p, x, y };
  });

  // SVG Line Command
  const linePath =
    points.length > 0
      ? points
          .map(
            (p, i) =>
              `${i === 0 ? 'M' : 'L'} ${p.x.toFixed(1)} ${p.y.toFixed(1)}`,
          )
          .join(' ')
      : '';

  // SVG Closed Path for linear-gradient area fill
  const bottomY = PAD_T + CHART_H;
  const areaPath =
    points.length > 0
      ? `${linePath} L ${points[points.length - 1].x.toFixed(1)} ${bottomY.toFixed(1)} L ${points[0].x.toFixed(1)} ${bottomY.toFixed(1)} Z`
      : '';

  // Y-axis grid markers
  const f1Ticks = [0.0, 0.25, 0.5, 0.75, 1.0];

  // X-axis epoch grid markers (5 total)
  const epochTicks: number[] = [];
  if (rawPoints.length > 0) {
    const step = rangeEpoch / 4;
    for (let i = 0; i <= 4; i++) {
      epochTicks.push(Math.round(minEpoch + i * step));
    }
  }

  // Handle SVG hover tracking
  function handleMouseMove(e: React.MouseEvent<SVGSVGElement>) {
    if (points.length === 0) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const svgX = (mouseX / rect.width) * VIEW_W;

    let closest = points[0];
    let minDist = Math.abs(points[0].x - svgX);
    for (let i = 1; i < points.length; i++) {
      const dist = Math.abs(points[i].x - svgX);
      if (dist < minDist) {
        minDist = dist;
        closest = points[i];
      }
    }

    if (svgX >= PAD_L - 10 && svgX <= PAD_L + CHART_W + 10) {
      setHoveredPoint(closest);
    } else {
      setHoveredPoint(null);
    }
  }

  // ── Chart Mathematics (Bar Chart) ──
  const barData = distribution?.species_counts || [];
  const maxCount =
    barData.length > 0 ? Math.max(...barData.map((d) => d.count)) : 10;

  // Deployed Model Version Info
  const hasActiveModel =
    f1History &&
    f1History.model_version &&
    f1History.model_version !== 'unknown';
  const modelName = hasActiveModel
    ? f1History.model_version
    : 'No Deployed Model';

  return (
    <div className="dash-page">
      <NavBar />

      <main className="dash-main">
        <div className="dash-container">
          {/* Header row */}
          <header className="dash-header">
            <div className="dash-header-left">
              <span className="dash-breadcrumb">MONITORING &amp; KPI</span>
              <h1 className="dash-title">Dashboard Analytics</h1>
              <p className="dash-subtitle">
                System telemetry, retraining validation, and dataset annotation
                indices.
              </p>
            </div>
            <div className="dash-header-right">
              <div className="dash-run-filter">
                <label className="dash-run-filter-label" htmlFor="run-select">
                  TRAINING RUN
                </label>
                <div className="dash-run-filter-control">
                  <select
                    id="run-select"
                    className="dash-run-select"
                    value={selectedRunId ?? ''}
                    onChange={(e) => {
                      const newRunId = e.target.value || null;
                      setSelectedRunId(newRunId);
                      void fetchDashboardDataForRun(newRunId);
                    }}
                    disabled={loading}
                  >
                    <option value="">All Runs</option>
                    {runs.map((r) => (
                      <option key={r.run_id} value={r.run_id}>
                        {r.status === 'completed' ? '✓' : '✗'}{' '}
                        {new Date(r.started_at).toLocaleDateString('en-GB', {
                          day: '2-digit',
                          month: 'short',
                        })}{' '}
                        — F1: {((r.final_f1 ?? 0) * 100).toFixed(1)}%
                      </option>
                    ))}
                  </select>
                </div>
              </div>
            </div>
          </header>

          {error && (
            <div className="dash-alert dash-alert--warning">{error}</div>
          )}

          {loading && !metrics ? (
            <div className="dash-loading-screen">
              <span className="spinner" aria-hidden="true" />
              <p>Gathering system telemetry...</p>
            </div>
          ) : (
            <>
              {/* ── Section 1: KPI Cards ───────────────────────────────────── */}
              <div className="dash-kpis-grid">
                {/* F1 Radial Card */}
                <div className="dash-card dash-card--kpi">
                  <div className="dash-card-kpi-left">
                    <span className="kpi-label">VALIDATION F1 SCORE</span>
                    <span className="kpi-value">
                      {metrics && metrics.f1_score > 0
                        ? metrics.f1_score
                        : '0.00'}
                    </span>
                    {metrics && metrics.f1_score > 0 ? (
                      <span className="kpi-badge kpi-badge--success">
                        {selectedRunId
                          ? 'This training run'
                          : 'Best across all runs'}
                      </span>
                    ) : (
                      <span className="kpi-badge kpi-badge--muted">
                        No retraining data
                      </span>
                    )}
                  </div>
                  <div className="dash-card-kpi-right">
                    <div className="dash-radial-gauge">
                      <svg width="68" height="68" viewBox="0 0 36 36">
                        <circle
                          cx="18"
                          cy="18"
                          r="15.915"
                          fill="none"
                          stroke="var(--color-border-subtle)"
                          strokeWidth="2.5"
                        />
                        <circle
                          cx="18"
                          cy="18"
                          r="15.915"
                          fill="none"
                          stroke={
                            metrics && metrics.f1_score > 0
                              ? 'var(--color-primary-light)'
                              : 'var(--color-text-muted)'
                          }
                          strokeWidth="2.5"
                          strokeDasharray={`${metrics && metrics.f1_score > 0 ? metrics.f1_score * 100 : 0} ${100 - (metrics && metrics.f1_score > 0 ? metrics.f1_score * 100 : 0)}`}
                          strokeDashoffset="25"
                          strokeLinecap="round"
                          className="radial-progress-fill"
                        />
                      </svg>
                      <span className="radial-inner-label">U-Net</span>
                    </div>
                  </div>
                </div>

                {/* Throughput Gauge Card */}
                <div className="dash-card dash-card--kpi">
                  <div className="dash-card-kpi-left">
                    <span className="kpi-label">PIPELINE THROUGHPUT</span>
                    <span className="kpi-value">
                      {metrics
                        ? metrics.throughput_img_per_s >= 1
                          ? `${metrics.throughput_img_per_s.toFixed(3)} img/s`
                          : `${(metrics.throughput_img_per_s * 60).toFixed(3)} img/min`
                        : '—'}
                    </span>
                    {metrics && metrics.throughput_img_per_s > 0 ? (
                      <span className="kpi-badge kpi-badge--info">
                        {selectedRunId
                          ? 'This run benchmark'
                          : 'Latest benchmark'}
                      </span>
                    ) : (
                      <span className="kpi-badge kpi-badge--muted">
                        Calculation Failed
                      </span>
                    )}
                  </div>
                  <div className="dash-card-kpi-right">
                    <div
                      className="throughput-gauge-icon"
                      style={{
                        opacity:
                          metrics && metrics.throughput_img_per_s > 0 ? 1 : 0.4,
                      }}
                    >
                      <svg
                        width="36"
                        height="36"
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="var(--color-primary)"
                        strokeWidth="1.8"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                      >
                        <path d="M12 2v20" />
                        <path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6" />
                      </svg>
                    </div>
                  </div>
                </div>

                {/* Model Details Summary */}
                <div
                  className={`dash-card dash-card--kpi dash-card--model ${hasActiveModel ? '' : 'dash-card--model-inactive'}`}
                >
                  <div className="model-summary-info">
                    <span className="kpi-label">ACTIVE MODEL CONFIG</span>
                    <span className="model-version-name">{modelName}</span>
                    {hasActiveModel ? (
                      <span className="model-status-badge">
                        DEPLOYED &amp; ACTIVE
                      </span>
                    ) : (
                      <span className="model-status-badge model-status-badge--standby">
                        STANDBY STATE
                      </span>
                    )}
                  </div>
                </div>
              </div>

              {/* ── Section 2: Charts Grid ─────────────────────────────────── */}
              {f1History?.data_points.length === 0 ? (
                <div className="chart-empty-state">
                  <svg
                    width="48"
                    height="48"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="var(--color-text-muted)"
                    strokeWidth="1.5"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  >
                    <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
                  </svg>
                  <h4>No Training History Yet</h4>
                  <p>
                    Start a training run from the{' '}
                    <a href="/models?tab=retraining">Models page</a> to see
                    validation metrics here.
                  </p>
                </div>
              ) : (
                <div className="dash-charts-grid">
                  {/* F1 Epoch Line Chart */}
                  <div className="dash-card dash-chart-card">
                    <div className="chart-header">
                      <div className="chart-header-title">
                        <h3>
                          {selectedRunId
                            ? `Validation F1 — Run ${selectedRunId.slice(0, 8).toUpperCase()}`
                            : 'Validation F1 — All Runs'}
                        </h3>
                        <p>
                          {selectedRunId
                            ? 'Per-epoch validation F1 for the selected training run.'
                            : 'All recorded validation epochs across every training run.'}
                        </p>
                      </div>
                    </div>

                    <div className="line-chart-wrapper">
                      {rawPoints.length === 0 ? (
                        <div className="chart-empty-state">
                          <svg
                            width="48"
                            height="48"
                            viewBox="0 0 24 24"
                            fill="none"
                            stroke="var(--color-text-muted)"
                            strokeWidth="1.5"
                            strokeLinecap="round"
                            strokeLinejoin="round"
                          >
                            <line x1="18" y1="20" x2="18" y2="10" />
                            <line x1="12" y1="20" x2="12" y2="4" />
                            <line x1="6" y1="20" x2="6" y2="14" />
                            <line x1="2" y1="20" x2="22" y2="20" />
                          </svg>
                          <h4>No Retraining F1 Logged</h4>
                          <p>
                            Establish a training pipeline run to visualize F1
                            validation score progression across epochs.
                          </p>
                        </div>
                      ) : (
                        <svg
                          viewBox={`0 0 ${VIEW_W} ${VIEW_H}`}
                          className="line-chart-svg"
                          onMouseMove={handleMouseMove}
                          onMouseLeave={() => setHoveredPoint(null)}
                        >
                          {/* Gradient Definitions */}
                          <defs>
                            <linearGradient
                              id="chartAreaGrad"
                              x1="0"
                              y1="0"
                              x2="0"
                              y2="1"
                            >
                              <stop
                                offset="0%"
                                stopColor="var(--color-primary-light)"
                                stopOpacity="0.4"
                              />
                              <stop
                                offset="100%"
                                stopColor="var(--color-primary-light)"
                                stopOpacity="0.0"
                              />
                            </linearGradient>
                          </defs>

                          {/* Horizontal Grid lines */}
                          {f1Ticks.map((val) => {
                            const y = PAD_T + (1.0 - val) * CHART_H;
                            return (
                              <g key={val}>
                                <line
                                  x1={PAD_L}
                                  y1={y}
                                  x2={PAD_L + CHART_W}
                                  y2={y}
                                  stroke="var(--color-border-subtle)"
                                  strokeWidth="1"
                                  strokeDasharray="4,4"
                                />
                                <text
                                  x={PAD_L - 12}
                                  y={y + 3}
                                  textAnchor="end"
                                  className="chart-axis-label"
                                >
                                  {(val * 100).toFixed(0)}%
                                </text>
                              </g>
                            );
                          })}

                          {/* X-axis tick epoch tags */}
                          {epochTicks.map((val, idx) => {
                            const x =
                              PAD_L + ((val - minEpoch) / rangeEpoch) * CHART_W;
                            return (
                              <g key={idx}>
                                <text
                                  x={x}
                                  y={PAD_T + CHART_H + 18}
                                  textAnchor="middle"
                                  className="chart-axis-label"
                                >
                                  Epoch {val}
                                </text>
                              </g>
                            );
                          })}

                          {/* Line Area Fill (Gradient) */}
                          {areaPath && (
                            <path d={areaPath} fill="url(#chartAreaGrad)" />
                          )}

                          {/* Main Line Stroke */}
                          {linePath && (
                            <path
                              d={linePath}
                              fill="none"
                              stroke="var(--color-primary)"
                              strokeWidth="2.5"
                              strokeLinecap="round"
                              strokeLinejoin="round"
                            />
                          )}

                          {/* Data Dots */}
                          {points.map((p, idx) => (
                            <circle
                              key={idx}
                              cx={p.x}
                              cy={p.y}
                              r="3"
                              fill="var(--color-bg-white)"
                              stroke="var(--color-primary)"
                              strokeWidth="1.8"
                            />
                          ))}

                          {/* Interactive Hover Ruler & Guideline */}
                          {hoveredPoint && (
                            <>
                              <line
                                x1={hoveredPoint.x}
                                y1={PAD_T}
                                x2={hoveredPoint.x}
                                y2={PAD_T + CHART_H}
                                stroke="var(--color-primary-light)"
                                strokeWidth="1.2"
                                strokeDasharray="3,3"
                              />
                              <circle
                                cx={hoveredPoint.x}
                                cy={hoveredPoint.y}
                                r="7"
                                fill="var(--color-primary-light)"
                                opacity="0.3"
                              />
                              <circle
                                cx={hoveredPoint.x}
                                cy={hoveredPoint.y}
                                r="3.5"
                                fill="#ffffff"
                                stroke="var(--color-primary)"
                                strokeWidth="2"
                              />
                            </>
                          )}
                        </svg>
                      )}

                      {/* Glowing HTML Floating Tooltip mapped to coordinates */}
                      {hoveredPoint && (
                        <div
                          className="line-chart-tooltip"
                          style={{
                            left: `${((hoveredPoint.x - 6) / VIEW_W) * 100}%`,
                            top: `${((hoveredPoint.y - 48) / VIEW_H) * 100}%`,
                          }}
                        >
                          <span className="tooltip-title">
                            Epoch {hoveredPoint.epoch}
                          </span>
                          <span className="tooltip-val">
                            F1: {(hoveredPoint.f1_score * 100).toFixed(2)}%
                          </span>
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Annotation Species Distribution Bar Chart */}
                  <div className="dash-card dash-chart-card">
                    <div className="chart-header">
                      <div className="chart-header-title">
                        <h3>Annotation Species Distribution</h3>
                        <p>
                          Aggregated sample count of verified annotations per
                          species inside our database.
                        </p>
                      </div>
                    </div>

                    <div className="bar-chart-wrapper">
                      {barData.length === 0 ? (
                        <div className="chart-empty-state">
                          <svg
                            width="48"
                            height="48"
                            viewBox="0 0 24 24"
                            fill="none"
                            stroke="var(--color-text-muted)"
                            strokeWidth="1.5"
                            strokeLinecap="round"
                            strokeLinejoin="round"
                          >
                            <circle cx="12" cy="12" r="10" />
                            <line x1="12" y1="8" x2="12" y2="12" />
                            <line x1="12" y1="16" x2="12.01" y2="16" />
                          </svg>
                          <h4>No Annotation Metrics Indexed</h4>
                          <p>
                            Verify and save predictions in the Inference Lab to
                            populate plant species sample indices.
                          </p>
                        </div>
                      ) : (
                        <div className="bar-chart-list">
                          {barData.map((d, idx) => {
                            const pctWidth = (
                              (d.count / maxCount) *
                              100
                            ).toFixed(1);
                            return (
                              <div key={idx} className="bar-row">
                                <div className="bar-label-group">
                                  <span className="bar-label-species">
                                    {d.species}
                                  </span>
                                  <span className="bar-label-count">
                                    {d.count} confirmed
                                  </span>
                                </div>
                                <div className="bar-track">
                                  <div
                                    className="bar-fill"
                                    style={{
                                      width: `${pctWidth}%`,
                                      animationDelay: `${idx * 100}ms`,
                                    }}
                                  />
                                </div>
                              </div>
                            );
                          })}
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      </main>
    </div>
  );
}

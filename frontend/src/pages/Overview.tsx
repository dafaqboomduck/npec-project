/**
 * pages/Overview.tsx
 *
 * Home / Overview page — matches the Figma design.
 * Fetches: /overview/status, /overview/activity, /overview/model-health,
 *          /overview/environment, /models
 *
 * Gracefully degrades when the API is offline — shows "—" placeholders.
 * Route: /overview (protected)
 */

import { useEffect, useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import NavBar from '../components/NavBar';
import { getUsername, getRole } from '../api/auth';
import {
  getOverviewStatus,
  getOverviewActivity,
  getModelHealth,
  getEnvironment,
  getModels,
} from '../api/overview';
import type {
  OverviewStatusResponse,
  ActivityItem,
  ModelHealthResponse,
  EnvironmentResponse,
  ModelSummary,
  ActivityEventType,
} from '../types/api';
import './Overview.css';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function relativeTime(iso: string): string {
  const diff = Date.now() - new Date(iso).getTime();
  const m = Math.floor(diff / 60000);
  const h = Math.floor(m / 60);
  const d = Math.floor(h / 24);
  if (m < 60) return `${m}m ago`;
  if (h < 24) return `${h}h ago`;
  if (d === 1) return 'Yesterday';
  return `${d}d ago`;
}

function fmtTb(tb: number): string {
  return tb >= 1 ? `${tb.toFixed(1)}TB` : `${(tb * 1000).toFixed(0)}GB`;
}

// ---------------------------------------------------------------------------
// Activity icon per event type
// ---------------------------------------------------------------------------

function ActivityIconSvg({ type }: { type: ActivityEventType }) {
  if (type === 'retraining')
    return (
      <svg
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        <path d="M23 4v6h-6" />
        <path d="M1 20v-6h6" />
        <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15" />
      </svg>
    );
  if (type === 'upload')
    return (
      <svg
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        <polyline points="16 16 12 12 8 16" />
        <line x1="12" y1="12" x2="12" y2="21" />
        <path d="M20.39 18.39A5 5 0 0 0 18 9h-1.26A8 8 0 1 0 3 16.3" />
      </svg>
    );
  if (type === 'annotation')
    return (
      <svg
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2" />
        <circle cx="9" cy="7" r="4" />
        <path d="M23 21v-2a4 4 0 0 0-3-3.87" />
        <path d="M16 3.13a4 4 0 0 1 0 7.75" />
      </svg>
    );
  // deployment
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" />
    </svg>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export default function Overview() {
  const navigate = useNavigate();
  const username = getUsername();
  const role = getRole();
  // Display name: strip the domain from an email, capitalise first letter
  const displayName = username.includes('@')
    ? username
      .split('@')[0]
      .replace(/[._-]/g, ' ')
      .replace(/\b\w/g, (c) => c.toUpperCase())
    : username;

  // ── Data state ──
  const [status, setStatus] = useState<OverviewStatusResponse | null>(null);
  const [activity, setActivity] = useState<ActivityItem[]>([]);
  const [health, setHealth] = useState<ModelHealthResponse | null>(null);
  const [env, setEnv] = useState<EnvironmentResponse | null>(null);
  const [models, setModels] = useState<ModelSummary[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    void Promise.allSettled([
      getOverviewStatus()
        .then(setStatus)
        .catch(() => undefined),
      getOverviewActivity()
        .then((r) => setActivity(r.items))
        .catch(() => undefined),
      getModelHealth()
        .then(setHealth)
        .catch(() => undefined),
      getEnvironment()
        .then(setEnv)
        .catch(() => undefined),
      getModels()
        .then(setModels)
        .catch(() => undefined),
    ]).finally(() => setLoading(false));
  }, []);

  const deployedModels = models.filter((m) => m.status?.toLowerCase() === 'deployed');
  const currentModel = deployedModels[0] ?? null;

  // ── Health bar fill widths ──
  const f1Pct = health ? Math.round(health.f1_score * 100) : null;
  const latencyPct = health
    ? Math.min((Math.log10(health.latency_ms + 1) / Math.log10(60_000)) * 100, 100)
    : null;
  const latencyDanger = health ? health.latency_ms > 32_000 : false;


  return (
    <div className="overview-page">
      <NavBar />

      <main className="overview-main">
        <div className="overview-container">
          {/* ── Welcome ─────────────────────────────────────────────────── */}
          <section className="overview-welcome">
            <div className="overview-welcome-text">
              <h1 className="overview-heading">Welcome back, {displayName}</h1>
              <p className="overview-subheading">
                Rhizome Lab current status:&nbsp;
                <span className="status-active">Active</span>.
                {status
                  ? ` ${status.total_experiments.toLocaleString()} experiments on record.`
                  : ' Connecting to backend…'}
              </p>
            </div>
            <button
              className="btn-new-experiment"
              id="btn-new-experiment"
              onClick={() => navigate('/inference')}
            >
              + New Experiment
            </button>
          </section>

          {/* ── Stats row ───────────────────────────────────────────────── */}
          <div className="overview-stats" aria-label="Lab statistics">
            {/* Total experiments */}
            <div className="stat-card stat-card--light">
              <span className="stat-watermark">π</span>
              <span className="stat-label">TOTAL EXPERIMENTS</span>
              <span className="stat-value">
                {loading
                  ? '—'
                  : (status?.total_experiments.toLocaleString() ?? '—')}
              </span>
              <span className="stat-trend">↗ Running total</span>
            </div>

            {/* Active models */}
            <div className="stat-card stat-card--dark">
              <span className="stat-label">ACTIVE MODELS</span>
              <span className="stat-value stat-value--light">
                {loading
                  ? '—'
                  : String(
                    status?.active_model_count ?? deployedModels.length,
                  ).padStart(2, '0')}
              </span>
              <div className="model-badges">
                {deployedModels.length > 0 ? (
                  deployedModels.slice(0, 3).map((m) => (
                    <span className="model-badge" key={m.model_id}>
                      {m.name}
                    </span>
                  ))
                ) : (
                  <span className="model-badge model-badge--empty">
                    No models active
                  </span>
                )}
              </div>
            </div>

            {/* Data collected */}
            <div className="stat-card stat-card--light">
              <span className="stat-label">DATA COLLECTED</span>
              <span className="stat-value">
                {loading ? '—' : status ? fmtTb(status.data_collected_tb) : '—'}
              </span>
              <div className="stat-progress-track">
                <div
                  className="stat-progress-fill"
                  style={{ width: `${status?.quota_percent ?? 0}%` }}
                />
              </div>
              <span className="stat-trend">
                {status
                  ? `${status.quota_percent}% of assigned lab quota`
                  : 'Lab quota unavailable'}
              </span>
            </div>
          </div>

          {/* ── Hero banner ─────────────────────────────────────────────── */}
          <div className="overview-hero" aria-label="Lab environment">
            <img
              src="/botanical_hero_1776696272801.png"
              alt="Botanical laboratory"
              className="hero-img"
            />
            <div className="hero-overlay" />
            {/* Floating model card */}
            <div className="hero-model-card">
              <span className="hero-card-label">CURRENT MODEL</span>
              <div className="hero-card-model">
                <span className="hero-card-icon" aria-hidden="true">
                  <svg
                    width="22"
                    height="22"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="1.8"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  >
                    <circle cx="12" cy="12" r="3" />
                    <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" />
                  </svg>
                </span>
                <span className="hero-card-name">
                  {currentModel ? currentModel.name : 'No deployed models'}
                </span>
              </div>
            </div>
          </div>

          {/* ── Bottom split ────────────────────────────────────────────── */}
          <div className="overview-bottom">
            {/* Recent Activity */}
            <section className="activity-section">
              <h2 className="section-heading">Recent Activity</h2>

              {activity.length === 0 && !loading ? (
                <p className="activity-empty">No recent activity to display.</p>
              ) : (
                <ul className="activity-list" role="list">
                  {activity.slice(0, 5).map((item, idx) => (
                    <li key={idx} className="activity-item">
                      <div
                        className={`activity-icon activity-icon--${item.event_type}`}
                      >
                        <ActivityIconSvg type={item.event_type} />
                      </div>
                      <div className="activity-content">
                        <div className="activity-title-row">
                          <span className="activity-title">{item.title}</span>
                          <span className="activity-time">
                            {relativeTime(item.timestamp)}
                          </span>
                        </div>
                        <p className="activity-description">
                          {item.description}
                        </p>
                        {(() => {
                          const link = (() => {
                            if (!item.resource_url) return null;
                            const match = item.resource_url.match(/([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})/i);
                            const id = match ? match[1] : '';
                            if (item.event_type === 'upload') return id ? `/history?prediction_id=${id}` : '/history';
                            if (item.event_type === 'annotation') return '/library?tab=annotations';
                            if (item.event_type === 'deployment') return '/models?tab=registry';
                            if (item.event_type === 'retraining') return '/models?tab=retraining';
                            return null;
                          })();
                          return link ? (
                            <Link className="activity-link" to={link}>
                              View files ↗
                            </Link>
                          ) : null;
                        })()}
                      </div>
                    </li>
                  ))}
                </ul>
              )}

              <button
                className="btn-view-logs"
                id="btn-view-logs"
                onClick={() => navigate('/history')}
              >
                VIEW ALL LABORATORY LOGS
              </button>
            </section>

            {/* Health + Environment */}
            <aside className="health-aside">
              <h2 className="section-heading">Model Health</h2>

              <div className="health-metrics">
                {/* f1 */}
                <div className="health-metric">
                  <div className="health-metric-header">
                    <span className="health-metric-label">VALIDATION F1</span>
                    <span className="health-metric-value">
                      {f1Pct !== null ? `${f1Pct}%` : '—'}
                    </span>
                  </div>
                  <div className="health-bar-track">
                    <div
                      className="health-bar-fill health-bar-fill--green"
                      style={{ width: `${f1Pct ?? 0}%` }}
                    />
                  </div>
                </div>

                {/* Latency */}
                <div className="health-metric">
                  <div className="health-metric-header">
                    <span className="health-metric-label">LATENCY</span>
                    <span className="health-metric-value">
                      {health ? `${(health.latency_ms / 1000).toFixed(3)}s` : '—'}
                    </span>
                  </div>
                  <div className="health-bar-track">
                    <div
                      className={`health-bar-fill ${latencyDanger ? 'health-bar-fill--danger' : 'health-bar-fill--dark'}`}
                      style={{ width: `${latencyPct ?? 0}%` }}
                    />
                  </div>
                </div>
              </div>

              {/* Environment stats card */}
              {role === 'admin' && (
                <div className="env-card">
                  <div className="env-card-header">
                    <h3 className="env-card-title">Environment Stats</h3>
                    <span
                      className={`env-status-badge ${env?.system_status === 'optimal' ? 'env-status-badge--ok' : 'env-status-badge--warn'}`}
                    >
                      <span className="env-dot" />
                      {env ? capitalize(env.system_status) : 'Unknown'}
                    </span>
                  </div>
                  <div className="env-grid">
                    {env?.gpu_temp_celsius != null && (
                      <div className="env-stat">
                        <span className="env-stat-label">GPU TEMP</span>
                        <span className="env-stat-value">
                          {env.gpu_temp_celsius.toFixed(1)}°C
                        </span>
                      </div>
                    )}
                    {env?.gpu_utilisation != null && (
                      <div className="env-stat">
                        <span className="env-stat-label">GPU UTIL</span>
                        <span className="env-stat-value">
                          {env.gpu_utilisation.toFixed(0)}%
                        </span>
                      </div>
                    )}
                    {env?.vram_percent != null && (
                      <div className="env-stat">
                        <span className="env-stat-label">VRAM</span>
                        <span className="env-stat-value">
                          {env.vram_percent.toFixed(0)}%
                        </span>
                      </div>
                    )}
                    {env?.cpu_percent != null && (
                      <div className="env-stat">
                        <span className="env-stat-label">CPU</span>
                        <span className="env-stat-value">
                          {env.cpu_percent.toFixed(0)}%
                        </span>
                      </div>
                    )}
                    <div className="env-stat">
                      <span className="env-stat-label">MEMORY</span>
                      <span className="env-stat-value">
                        {env ? `${env.memory_percent}%` : '—'}
                      </span>
                    </div>
                    <div className="env-stat">
                      <span className="env-stat-label">DISK</span>
                      <span className="env-stat-value">
                        {env ? `${env.disk_percent}%` : '—'}
                      </span>
                    </div>
                    <div className="env-stat">
                      <span className="env-stat-label">CUDA</span>
                      <span className="env-stat-value">
                        {env ? (env.cuda_available ? 'Yes' : 'No') : '—'}
                      </span>
                    </div>
                  </div>
                </div>
              )}
            </aside>
          </div>
        </div>
      </main>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

function capitalize(s: string): string {
  return s.charAt(0).toUpperCase() + s.slice(1);
}

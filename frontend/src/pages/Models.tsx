/**
 * pages/Models.tsx
 *
 * Dedicated Admin Models Management Console.
 * Restricts access to roles with the 'admin' authorization token.
 *
 * Integrates two administrative workspaces:
 * 1. U-Net Retraining: Triggers background pipelines and logs subprocesses.
 * 2. Model Registry: Registers new weights (.keras), audits metadata, and deploys/deletes versions.
 *
 * Route: /models (protected, admin only)
 */

import React, { useEffect, useRef, useState } from 'react';
import NavBar from '../components/NavBar';
import {
  listTrainingRuns,
  startTrainingRun,
  stopTrainingRun,
} from '../api/trainingApi';
import {
  listModels,
  registerModel,
  deleteModel,
  deployModel,
  getModelDetail,
} from '../api/modelsApi';
import type {
  TrainingRunDetail,
  TrainingStartRequest,
  ModelSummary,
  ModelDetail,
  ModelCreate,
} from '../types/api';
import './Models.css';

export default function Models() {
  // ── Tab Management ──
  const [activeTab, setActiveTab] = useState<'retraining' | 'registry'>(() => {
    const params = new URLSearchParams(window.location.search);
    const tabParam = params.get('tab');
    if (tabParam === 'retraining' || tabParam === 'registry') {
      return tabParam as 'retraining' | 'registry';
    }
    return 'retraining';
  });

  // ── UI States ──
  const [runs, setRuns] = useState<TrainingRunDetail[]>([]);
  const [models, setModels] = useState<ModelSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [actionLoading, setActionLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [infoMessage, setInfoMessage] = useState<string | null>(null);
  const [isNotImplemented, setIsNotImplemented] = useState(false);

  // ── Hyperparameter Form Controls (Retraining) ──
  const [learningRate, setLearningRate] = useState<string>('0.001');
  const [batchSize, setBatchSize] = useState<number>(16);
  const [epochs, setEpochs] = useState<number>(50);

  // ── Register Model Form Controls (Registry) ──
  const [regVersion, setRegVersion] = useState('');
  const [regName, setRegName] = useState('');
  const [regArtifactUri, setRegArtifactUri] = useState('');
  const [regDescription, setRegDescription] = useState('');
  const [regF1Score, setRegF1Score] = useState('');
  const [regPrecision, setRegPrecision] = useState('');
  const [regRecall, setRegRecall] = useState('');
  const [regLatency, setRegLatency] = useState('');

  // ── Expandable Detail Rows ──
  const [expandedRunId, setExpandedRunId] = useState<string | null>(null);
  const [expandedModelId, setExpandedModelId] = useState<string | null>(null);
  const [detailedModels, setDetailedModels] = useState<
    Record<string, ModelDetail>
  >({});

  const pollingRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    fetchData();
  }, [activeTab]);

  useEffect(() => {
    const hasRunningRun = runs.some((r) => r.status === 'running');

    if (hasRunningRun) {
      // Start polling if not already
      if (!pollingRef.current) {
        pollingRef.current = setInterval(() => {
          void fetchData();
        }, 5000);
      }
    } else {
      // Clear polling when nothing is running
      if (pollingRef.current) {
        clearInterval(pollingRef.current);
        pollingRef.current = null;
      }
    }

    // Cleanup on unmount
    return () => {
      if (pollingRef.current) {
        clearInterval(pollingRef.current);
        pollingRef.current = null;
      }
    };
  }, [runs]);

  async function fetchData() {
    setLoading(true);
    setError(null);
    setInfoMessage(null);
    setIsNotImplemented(false);

    try {
      // Always fetch models to identify the currently deployed base
      try {
        const liveModels = await listModels();
        setModels(liveModels || []);
      } catch (err) {
        console.error('Failed to fetch models list:', err);
      }

      if (activeTab === 'retraining') {
        const liveRuns = await listTrainingRuns();
        setRuns(liveRuns || []);
      }
    } catch (err: any) {
      const msg = err.message || '';
      if (
        activeTab === 'retraining' &&
        (msg.includes('501') || msg.includes('Not implemented'))
      ) {
        setIsNotImplemented(true);
        setError(
          'The model retraining subprocess is currently under integration (501 Not Implemented).',
        );
      } else {
        setError(
          err.message ||
            'Connection failed. Could not fetch models workspace data.',
        );
      }
    } finally {
      setLoading(false);
    }
  }

  // ── Retraining Actions ──

  async function handleLaunchTraining(e: React.FormEvent) {
    e.preventDefault();
    setActionLoading(true);
    setError(null);
    setInfoMessage(null);

    const config: TrainingStartRequest = {
      learning_rate: parseFloat(learningRate) || 0.001,
      batch_size: batchSize,
      epochs: epochs,
    };

    try {
      const res = await startTrainingRun(config);
      setInfoMessage(
        `✓ Retraining initiated: ${res.message} (Run ID: ${res.run_id.slice(0, 8)})`,
      );
      await fetchData();
    } catch (err: any) {
      const msg = err.message || '';
      if (msg.includes('501') || msg.includes('Not implemented')) {
        setError(
          'Retraining trigger failed: The training runner subprocess daemon is pending deployment (501).',
        );
      } else if (msg.includes('already in progress') || msg.includes('409')) {
        // Find the running run to show its short ID
        const runningRun = runs.find((r) => r.status === 'running');
        const runLabel = runningRun
          ? `run_${runningRun.run_id.slice(0, 8)}`
          : 'a run';
        setError(
          `${runLabel} is already running — wait for it to finish or terminate it using the button in the table below.`,
        );
      } else {
        setError(msg || 'Failed to start retraining run.');
      }
    } finally {
      setActionLoading(false);
    }
  }

  async function handleStopTraining(runId: string, e: React.MouseEvent) {
    e.stopPropagation();
    if (
      !window.confirm('Are you sure you want to terminate this retraining job?')
    )
      return;

    setActionLoading(true);
    setError(null);

    try {
      const res = await stopTrainingRun(runId);
      setInfoMessage(res.message);
      await fetchData();
    } catch (err: any) {
      const msg = err.message || '';
      if (msg.includes('501') || msg.includes('Not implemented')) {
        setError(
          'Stop action failed: Training runner subprocess management is pending deployment (501).',
        );
      } else {
        setError(err.message || 'Failed to stop retraining run.');
      }
    } finally {
      setActionLoading(false);
    }
  }

  // ── Model Registry Actions ──

  async function handleRegisterModel(e: React.FormEvent) {
    e.preventDefault();
    setActionLoading(true);
    setError(null);
    setInfoMessage(null);

    const payload: ModelCreate = {
      version: regVersion.trim(),
      name: regName.trim(),
      artifact_uri: regArtifactUri.trim(),
      description: regDescription.trim() || undefined,
      f1_score: regF1Score ? parseFloat(regF1Score) : undefined,
      precision: regPrecision ? parseFloat(regPrecision) : undefined,
      recall: regRecall ? parseFloat(regRecall) : undefined,
      inference_latency_ms: regLatency ? parseInt(regLatency) : undefined,
    };

    try {
      const res = await registerModel(payload);
      setInfoMessage(`✓ Model version ${res.version} successfully registered!`);
      // Clear form
      setRegVersion('');
      setRegName('');
      setRegArtifactUri('');
      setRegDescription('');
      setRegF1Score('');
      setRegPrecision('');
      setRegRecall('');
      setRegLatency('');
      await fetchData();
    } catch (err: any) {
      setError(err.message || 'Failed to register new model version.');
    } finally {
      setActionLoading(false);
    }
  }

  async function handleDeployModel(modelId: string, e: React.MouseEvent) {
    e.stopPropagation();
    if (
      !window.confirm(
        'Deploy this U-Net model version? This will set it as the active inference engine.',
      )
    )
      return;

    setActionLoading(true);
    setError(null);
    setInfoMessage(null);

    try {
      const res = await deployModel(modelId);
      setInfoMessage(`🚀 Deployment successful: ${res.message}`);
      await fetchData();
    } catch (err: any) {
      setError(err.message || 'Failed to deploy model.');
    } finally {
      setActionLoading(false);
    }
  }

  async function handleDeleteModel(modelId: string, e: React.MouseEvent) {
    e.stopPropagation();
    if (
      !window.confirm(
        'Are you sure you want to delete this registered model? This cannot be undone.',
      )
    )
      return;

    setActionLoading(true);
    setError(null);
    setInfoMessage(null);

    try {
      const res = await deleteModel(modelId);
      setInfoMessage(res.message);
      await fetchData();
    } catch (err: any) {
      setError(err.message || 'Failed to delete model.');
    } finally {
      setActionLoading(false);
    }
  }

  async function toggleExpandModelRow(modelId: string) {
    if (expandedModelId === modelId) {
      setExpandedModelId(null);
      return;
    }

    setExpandedModelId(modelId);
    if (!detailedModels[modelId]) {
      try {
        const detail = await getModelDetail(modelId);
        setDetailedModels((prev) => ({ ...prev, [modelId]: detail }));
      } catch (err) {
        console.error('Failed to fetch model details', err);
      }
    }
  }

  function toggleExpandRow(runId: string) {
    if (expandedRunId === runId) {
      setExpandedRunId(null);
    } else {
      setExpandedRunId(runId);
    }
  }

  return (
    <div className="models-page">
      <NavBar />

      <main className="models-main">
        <div className="models-container">
          {/* Header section */}
          <header className="models-header">
            <div className="models-header-left">
              <span className="models-breadcrumb">ADMIN CONSOLE</span>
              <h1 className="models-title">Models Administration</h1>
              <p className="models-subtitle">
                Supervise automated U-Net retraining pipelines, register model
                weights, and deploy inference versions.
              </p>
            </div>
          </header>

          {/* Tab Navigation */}
          <div className="models-tabs">
            <button
              className={`tab-btn ${activeTab === 'retraining' ? 'tab-btn--active' : ''}`}
              onClick={() => setActiveTab('retraining')}
            >
              U-Net Retraining Runs
            </button>
            <button
              className={`tab-btn ${activeTab === 'registry' ? 'tab-btn--active' : ''}`}
              onClick={() => setActiveTab('registry')}
            >
              Model Registry & Deployments
            </button>
          </div>

          {/* 501 Integration Banner */}
          {activeTab === 'retraining' && isNotImplemented && (
            <div className="models-notice-banner">
              <div className="notice-icon">
                <svg
                  width="20"
                  height="20"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
                  <line x1="12" y1="9" x2="12" y2="13" />
                  <line x1="12" y1="17" x2="12.01" y2="17" />
                </svg>
              </div>
              <div className="notice-content">
                <strong>U-Net Retraining Subprocess Offline (Task 2097)</strong>
                <p>
                  The U-Net training daemon router is pending server
                  infrastructure integration. Real-time controls will activate
                  once deployed.
                </p>
              </div>
            </div>
          )}

          {error && (
            <div className="models-alert models-alert--danger">{error}</div>
          )}
          {infoMessage && (
            <div className="models-alert models-alert--success">
              {infoMessage}
            </div>
          )}

          {/* ── Tab Content ───────────────────────────────────────────────── */}
          {activeTab === 'retraining' ? (
            <div className="models-grid">
              {/* Left: Hyperparameter Configuration */}
              <div className="models-card models-form-card">
                <h3 className="card-title">Retraining Launch Control</h3>
                <p className="card-desc">
                  Configure variables to invoke a new U-Net training background
                  subprocess.
                </p>

                {(() => {
                  const deployedModel = models.find(
                    (m) => m.status === 'deployed',
                  );
                  return (
                    <div
                      style={{
                        background: 'rgba(255, 255, 255, 0.03)',
                        border: '1px solid var(--color-border-subtle)',
                        borderRadius: '6px',
                        padding: '10px 12px',
                        marginBottom: '15px',
                        fontSize: '12px',
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'center',
                      }}
                    >
                      <span
                        style={{
                          color: 'var(--color-text-muted)',
                          fontWeight: 600,
                        }}
                      >
                        RETRAINING ANCHOR BASE:
                      </span>
                      <span
                        style={{
                          fontFamily: 'var(--font-mono)',
                          fontWeight: 700,
                          color: 'var(--color-primary-light)',
                        }}
                      >
                        {deployedModel ? `${deployedModel.name}` : 'U-Net Base'}
                      </span>
                    </div>
                  );
                })()}

                <form onSubmit={handleLaunchTraining} className="models-form">
                  <div className="form-group">
                    <label htmlFor="learning-rate">Learning Rate (&eta;)</label>
                    <input
                      id="learning-rate"
                      type="number"
                      step="0.0001"
                      min="0.0001"
                      max="0.1"
                      value={learningRate}
                      onChange={(e) => setLearningRate(e.target.value)}
                      disabled={actionLoading || isNotImplemented}
                      required
                    />
                    <small className="form-help">
                      Governs weights convergence. Recommended: 0.001
                    </small>
                  </div>

                  <div className="form-group">
                    <label htmlFor="batch-size">Batch Size</label>
                    <select
                      id="batch-size"
                      value={batchSize}
                      onChange={(e) => setBatchSize(Number(e.target.value))}
                      disabled={actionLoading || isNotImplemented}
                    >
                      <option value={8}>8 samples</option>
                      <option value={16}>16 samples (Default)</option>
                      <option value={32}>32 samples</option>
                      <option value={64}>64 samples</option>
                    </select>
                    <small className="form-help">
                      Sub-sampling constraints per forward pass.
                    </small>
                  </div>

                  <div className="form-group">
                    <div className="slider-header">
                      <label htmlFor="epochs-range">Target Epochs</label>
                      <span className="slider-badge">{epochs} epochs</span>
                    </div>
                    <input
                      id="epochs-range"
                      type="range"
                      min="1"
                      max="100"
                      step="1"
                      value={epochs}
                      onChange={(e) => setEpochs(Number(e.target.value))}
                      disabled={actionLoading || isNotImplemented}
                    />
                    <small className="form-help">
                      Total full passes across dataset records.
                    </small>
                  </div>

                  <button
                    type="submit"
                    className="models-submit-btn"
                    disabled={actionLoading || isNotImplemented}
                  >
                    {isNotImplemented
                      ? 'Retraining Pipeline Offline'
                      : actionLoading
                        ? 'Initializing Daemon...'
                        : 'Launch Retraining Pipeline'}
                  </button>
                </form>
              </div>

              {/* Right: Retraining Jobs History */}
              <div className="models-card models-table-card">
                <h3 className="card-title">Training Jobs Monitor</h3>
                <h3 className="card-title">
                  Retraining History
                  {runs.some((r) => r.status === 'running') && (
                    <span className="live-indicator">
                      <span className="pulse-dot" />
                      LIVE
                    </span>
                  )}
                </h3>
                <p className="card-desc">
                  Supervise active U-Net tasks and trace learning historical
                  statistics.
                </p>

                {loading ? (
                  <div className="models-table-loading">
                    <span className="spinner" aria-hidden="true" />
                    <p>Resolving runs database...</p>
                  </div>
                ) : isNotImplemented ? (
                  <div className="models-table-empty">
                    <svg
                      width="40"
                      height="40"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="var(--color-text-muted)"
                      strokeWidth="1.5"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    >
                      <circle cx="12" cy="12" r="10" />
                      <line x1="8" y1="12" x2="16" y2="12" />
                    </svg>
                    <h4>Training Daemon Unreachable</h4>
                    <p>
                      The U-Net retraining logs could not be retrieved due to
                      server-side Task 2097 integration.
                    </p>
                  </div>
                ) : runs.length === 0 ? (
                  <div className="models-table-empty">
                    <svg
                      width="40"
                      height="40"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="var(--color-text-muted)"
                      strokeWidth="1.5"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    >
                      <circle cx="12" cy="12" r="10" />
                      <line x1="12" y1="8" x2="12" y2="16" />
                      <line x1="8" y1="12" x2="16" y2="12" />
                    </svg>
                    <h4>No Retraining Runs Found</h4>
                    <p>
                      Invoke a retraining run on the left to spawn U-Net
                      historical logs.
                    </p>
                  </div>
                ) : (
                  <div className="models-table-wrapper">
                    <table className="models-table">
                      <thead>
                        <tr>
                          <th>RUN ID</th>
                          <th>STATUS</th>
                          <th>F1 SCORE</th>
                          <th>STARTED AT</th>
                          <th aria-label="Actions"></th>
                        </tr>
                      </thead>
                      <tbody>
                        {runs.map((run) => {
                          const isExpanded = expandedRunId === run.run_id;
                          const runShort = run.run_id.slice(0, 8);
                          const isRunning = run.status === 'running';

                          return (
                            <React.Fragment key={run.run_id}>
                              <tr
                                className={`run-row ${isExpanded ? 'run-row--expanded' : ''}`}
                                onClick={() => toggleExpandRow(run.run_id)}
                              >
                                <td className="cell-id">
                                  <code>run_{runShort}</code>
                                </td>
                                <td>
                                  <span
                                    className={`status-badge status-badge--${run.status}`}
                                  >
                                    {isRunning && (
                                      <span className="pulse-dot" />
                                    )}
                                    {run.status}
                                  </span>
                                </td>
                                <td className="cell-f1">
                                  {run.final_f1
                                    ? (run.final_f1 * 100).toFixed(2) + '%'
                                    : '—'}
                                </td>
                                <td className="cell-time">
                                  {new Date(run.started_at).toLocaleTimeString(
                                    [],
                                    {
                                      hour: '2-digit',
                                      minute: '2-digit',
                                      second: '2-digit',
                                    },
                                  )}
                                </td>
                                <td className="cell-actions">
                                  {isRunning ? (
                                    <button
                                      className="terminate-btn"
                                      onClick={(e) =>
                                        handleStopTraining(run.run_id, e)
                                      }
                                      title="Terminate Retraining Pipeline Job"
                                      disabled={actionLoading}
                                    >
                                      Terminate
                                    </button>
                                  ) : (
                                    <span className="row-expand-chevron">
                                      {isExpanded ? '▲' : '▼'}
                                    </span>
                                  )}
                                </td>
                              </tr>

                              {isExpanded && (
                                <tr className="detail-row">
                                  <td colSpan={5}>
                                    <div className="run-expanded-panel">
                                      <div className="panel-col">
                                        <h5>Configuration Metrics</h5>
                                        <ul>
                                          <li>
                                            <span>Learning Rate:</span>
                                            <strong>
                                              {run.hyperparameters
                                                ?.learning_rate !== undefined
                                                ? String(
                                                    run.hyperparameters
                                                      .learning_rate,
                                                  )
                                                : '0.001'}
                                            </strong>
                                          </li>
                                          <li>
                                            <span>Batch Size:</span>
                                            <strong>
                                              {run.hyperparameters
                                                ?.batch_size !== undefined
                                                ? String(
                                                    run.hyperparameters
                                                      .batch_size,
                                                  )
                                                : '16'}{' '}
                                              samples
                                            </strong>
                                          </li>
                                          <li>
                                            <span>Target Epochs:</span>
                                            <strong>
                                              {run.hyperparameters?.epochs !==
                                              undefined
                                                ? String(
                                                    run.hyperparameters.epochs,
                                                  )
                                                : '50'}{' '}
                                              epochs
                                            </strong>
                                          </li>
                                        </ul>
                                      </div>

                                      <div className="panel-col">
                                        <h5>Subprocess Telemetry</h5>
                                        <ul>
                                          <li>
                                            <span>Start Timestamp:</span>
                                            <strong>
                                              {new Date(
                                                run.started_at,
                                              ).toLocaleString()}
                                            </strong>
                                          </li>
                                          <li>
                                            <span>End Timestamp:</span>
                                            <strong>
                                              {run.completed_at
                                                ? new Date(
                                                    run.completed_at,
                                                  ).toLocaleString()
                                                : '—'}
                                            </strong>
                                          </li>
                                        </ul>
                                      </div>
                                    </div>
                                  </td>
                                </tr>
                              )}
                            </React.Fragment>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            </div>
          ) : (
            /* Registry Workspaces */
            <div className="models-grid">
              {/* Left Side: Register New Model */}
              <div className="models-card models-form-card">
                <h3 className="card-title">Register Model Version</h3>
                <p className="card-desc">
                  Integrate new weights and register parameters directly into
                  the U-Net registry.
                </p>

                <form onSubmit={handleRegisterModel} className="models-form">
                  <div className="form-group">
                    <label htmlFor="reg-name">Model Name</label>
                    <input
                      id="reg-name"
                      type="text"
                      placeholder="e.g. U-Net Botanical Sage"
                      value={regName}
                      onChange={(e) => setRegName(e.target.value)}
                      disabled={actionLoading}
                      required
                    />
                  </div>

                  <div className="form-group">
                    <label htmlFor="reg-version">Version string</label>
                    <input
                      id="reg-version"
                      type="text"
                      placeholder="e.g. v2.1.0"
                      value={regVersion}
                      onChange={(e) => setRegVersion(e.target.value)}
                      disabled={actionLoading}
                      required
                    />
                  </div>

                  <div className="form-group">
                    <label htmlFor="reg-artifact">Artifact Storage URI</label>
                    <input
                      id="reg-artifact"
                      type="text"
                      placeholder="e.g. s3://models/unet_sage.keras"
                      value={regArtifactUri}
                      onChange={(e) => setRegArtifactUri(e.target.value)}
                      disabled={actionLoading}
                      required
                    />
                  </div>

                  <div className="form-group">
                    <label htmlFor="reg-desc">Description</label>
                    <input
                      id="reg-desc"
                      type="text"
                      placeholder="Optional notes or training attributes"
                      value={regDescription}
                      onChange={(e) => setRegDescription(e.target.value)}
                      disabled={actionLoading}
                    />
                  </div>

                  <div className="form-row-metrics">
                    <div className="form-group">
                      <label htmlFor="reg-f1">F1 Score</label>
                      <input
                        id="reg-f1"
                        type="number"
                        step="0.0001"
                        min="0"
                        max="1"
                        placeholder="e.g. 0.942"
                        value={regF1Score}
                        onChange={(e) => setRegF1Score(e.target.value)}
                        disabled={actionLoading}
                      />
                    </div>
                    <div className="form-group">
                      <label htmlFor="reg-latency">Latency (ms)</label>
                      <input
                        id="reg-latency"
                        type="number"
                        placeholder="e.g. 150"
                        value={regLatency}
                        onChange={(e) => setRegLatency(e.target.value)}
                        disabled={actionLoading}
                      />
                    </div>
                  </div>

                  <button
                    type="submit"
                    className="models-submit-btn"
                    disabled={actionLoading}
                  >
                    {actionLoading
                      ? 'Saving Registry record...'
                      : 'Register Model Version'}
                  </button>
                </form>
              </div>

              {/* Right Side: Registered Models Table */}
              <div className="models-card models-table-card">
                <h3 className="card-title">Model Registry Monitor</h3>
                <p className="card-desc">
                  Supervise registered model versions, audit metrics, and deploy
                  live.
                </p>

                {loading ? (
                  <div className="models-table-loading">
                    <span className="spinner" aria-hidden="true" />
                    <p>Resolving models database...</p>
                  </div>
                ) : models.length === 0 ? (
                  <div className="models-table-empty">
                    <svg
                      width="40"
                      height="40"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="var(--color-text-muted)"
                      strokeWidth="1.5"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    >
                      <circle cx="12" cy="12" r="10" />
                      <line x1="12" y1="8" x2="12" y2="16" />
                      <line x1="8" y1="12" x2="16" y2="12" />
                    </svg>
                    <h4>No Registered Models Found</h4>
                    <p>
                      Register your first trained weights file on the left to
                      spawn U-Net registry logs.
                    </p>
                  </div>
                ) : (
                  <div className="models-table-wrapper">
                    <table className="models-table">
                      <thead>
                        <tr>
                          <th>VERSION</th>
                          <th>NAME</th>
                          <th>STATUS</th>
                          <th>CREATED AT</th>
                          <th aria-label="Actions"></th>
                        </tr>
                      </thead>
                      <tbody>
                        {models.map((model) => {
                          const isExpanded = expandedModelId === model.model_id;
                          const detail = detailedModels[model.model_id];
                          const isDeployed = model.status === 'deployed';

                          return (
                            <React.Fragment key={model.model_id}>
                              <tr
                                className={`run-row ${isExpanded ? 'run-row--expanded' : ''}`}
                                onClick={() =>
                                  toggleExpandModelRow(model.model_id)
                                }
                              >
                                <td className="cell-id">
                                  <code>{model.model_id.slice(0, 8)}</code>
                                </td>
                                <td className="cell-name-val">
                                  <strong>{model.name}</strong>
                                </td>
                                <td>
                                  <span
                                    className={`status-badge status-badge--${model.status}`}
                                  >
                                    {isDeployed && (
                                      <span
                                        className="pulse-dot"
                                        style={{
                                          backgroundColor:
                                            'var(--color-success)',
                                        }}
                                      />
                                    )}
                                    {model.status}
                                  </span>
                                </td>
                                <td className="cell-time">
                                  {new Date(
                                    model.created_at,
                                  ).toLocaleDateString([], {
                                    month: 'short',
                                    day: 'numeric',
                                    year: 'numeric',
                                  })}
                                </td>
                                <td className="cell-actions">
                                  <div className="action-row-buttons">
                                    {!isDeployed && (
                                      <button
                                        className="deploy-btn"
                                        onClick={(e) =>
                                          handleDeployModel(model.model_id, e)
                                        }
                                        title="Deploy this version active"
                                        disabled={actionLoading}
                                      >
                                        Deploy
                                      </button>
                                    )}
                                    <button
                                      className="delete-btn"
                                      onClick={(e) =>
                                        handleDeleteModel(model.model_id, e)
                                      }
                                      title="Delete model version"
                                      disabled={actionLoading}
                                    >
                                      Delete
                                    </button>
                                  </div>
                                </td>
                              </tr>

                              {isExpanded && (
                                <tr className="detail-row">
                                  <td colSpan={5}>
                                    <div className="run-expanded-panel">
                                      <div className="panel-col">
                                        <h5>Model Attributes</h5>
                                        <ul>
                                          <li>
                                            <span>Artifact URI:</span>
                                            <strong className="artifact-strong">
                                              {detail?.artifact_uri || '—'}
                                            </strong>
                                          </li>
                                          <li>
                                            <span>Description:</span>
                                            <strong
                                              style={{ textAlign: 'right' }}
                                            >
                                              {detail?.description ||
                                                'No description provided.'}
                                            </strong>
                                          </li>
                                          <li>
                                            <span>Registration Time:</span>
                                            <strong>
                                              {detail
                                                ? new Date(
                                                    detail.created_at,
                                                  ).toLocaleString()
                                                : '—'}
                                            </strong>
                                          </li>
                                        </ul>
                                      </div>

                                      <div className="panel-col">
                                        <h5>Evaluation Metrics</h5>
                                        <ul>
                                          <li>
                                            <span>F1 Validation:</span>
                                            <strong>
                                              {detail?.f1_score
                                                ? (
                                                    detail.f1_score * 100
                                                  ).toFixed(2) + '%'
                                                : '—'}
                                            </strong>
                                          </li>
                                          <li>
                                            <span>Precision:</span>
                                            <strong>
                                              {detail?.precision
                                                ? (
                                                    detail.precision * 100
                                                  ).toFixed(2) + '%'
                                                : '—'}
                                            </strong>
                                          </li>
                                          <li>
                                            <span>Recall:</span>
                                            <strong>
                                              {detail?.recall
                                                ? (detail.recall * 100).toFixed(
                                                    2,
                                                  ) + '%'
                                                : '—'}
                                            </strong>
                                          </li>
                                          <li>
                                            <span>Inference Latency:</span>
                                            <strong>
                                              {detail?.inference_latency_ms
                                                ? detail.inference_latency_ms +
                                                  ' ms'
                                                : '—'}
                                            </strong>
                                          </li>
                                        </ul>
                                      </div>
                                    </div>
                                  </td>
                                </tr>
                              )}
                            </React.Fragment>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}

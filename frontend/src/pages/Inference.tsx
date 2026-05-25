/**
 * pages/Inference.tsx
 *
 * Inference Lab page — matches the Figma design.
 * Left sidebar: model info, parameters, session.
 * Right panel: drag-and-drop upload zone, results.
 * "Select From Server" shows a "Coming soon" modal.
 *
 * Route: /inference (protected)
 */

import { ChangeEvent, DragEvent, useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import NavBar from '../components/NavBar';
import { predict, getModelsForInference, confirmPrediction, deletePrediction } from '../api/inferenceApi';
import type { PredictionResponse, ModelSummary } from '../types/api';
import { useInferenceJobs } from '../context/InferenceJobsContext';
import { getUsername, getRole } from '../api/auth';
import { getAnnotations, getDatasets, runDatasetInference } from '../api/libraryApi';
import type { DatasetSummary } from '../api/libraryApi';
import ResultsView from '../components/ResultsView';
import './Inference.css';

// ---------------------------------------------------------------------------
// Main Inference component
// ---------------------------------------------------------------------------

export default function Inference() {
  const navigate = useNavigate();

  // ── Sidebar state ──
  const [models, setModels] = useState<ModelSummary[]>([]);

  // ── Dropzone state ──
  const [dragging, setDragging] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [filePreviewUrl, setFilePreviewUrl] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // ── Inference state ──
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const { jobs, addJob, removeJob, cancelJob } = useInferenceJobs();

  // ── Modals ──
  const [showDiscardConfirm, setShowDiscardConfirm] = useState(false);
  const [showSaveModal, setShowSaveModal] = useState(false);

  // ── Select Dataset Modal States ──
  const [showSelectDatasetModal, setShowSelectDatasetModal] = useState(false);
  const [datasets, setDatasets] = useState<DatasetSummary[]>([]);
  const [loadingDatasets, setLoadingDatasets] = useState(false);

  // ── Batch Inference Progress States ──
  const [showBatchProgress, setShowBatchProgress] = useState(false);
  const [batchDatasetName, setBatchDatasetName] = useState('');
  const [batchTotalCount, setBatchTotalCount] = useState(0);
  const [batchCurrentIndex, setBatchCurrentIndex] = useState(0);
  const [batchPercent, setBatchPercent] = useState(0);
  const [batchCompleted, setBatchCompleted] = useState(false);

  // ── Save form fields ──
  const [saveSpecies, setSaveSpecies] = useState('');
  const [saveAgeDays, setSaveAgeDays] = useState('');
  const [saveTreatment, setSaveTreatment] = useState('');
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [discarding, setDiscarding] = useState(false);
  const [pendingFeedback, setPendingFeedback] = useState<'correct' | 'incorrect' | null>(null);
  const [datasetPendingCounts, setDatasetPendingCounts] = useState<Record<string, number>>({});

  // ── Active Session Stats ──
  const username = getUsername();
  const role = getRole();
  const displayName = username.includes('@')
    ? username.split('@')[0].replace(/[._-]/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase())
    : username;

  async function runBatchInference(dsId: string, dsName: string, count: number) {
    setShowSelectDatasetModal(false);
    setShowBatchProgress(true);
    setBatchDatasetName(dsName);
    setBatchTotalCount(count);
    setBatchCurrentIndex(0);
    setBatchPercent(0);
    setBatchCompleted(false);

    try {
      // Trigger background inference
      const res = await runDatasetInference(dsId);

      // If no pending images
      if (res.message.toLowerCase().includes('no pending')) {
        setBatchCompleted(true);
        setBatchPercent(100);
        setBatchCurrentIndex(count);
        return;
      }

      const initialAnnotations = await getAnnotations({
        dataset_id: dsId,
        page_size: 100,
      });
      const realTotal = initialAnnotations.length;
      setBatchTotalCount(realTotal);

      const initialPending = initialAnnotations.filter(a => a.status === 'pending').length;
      const initialDone = realTotal - initialPending;
      setBatchCurrentIndex(initialDone);
      setBatchPercent(realTotal > 0 ? Math.round((initialDone / realTotal) * 100) : 0);

      // Poll annotations until none are pending
      const MAX_POLLS = 120; // 10 minutes at 5s intervals
      let polls = 0;

      const interval = setInterval(async () => {
        polls++;
        try {
          const annotations = await getAnnotations({
            dataset_id: dsId,
            page_size: 100,
          });

          const total = annotations.length;
          const pending = annotations.filter(a => a.status === 'pending').length;
          const done = total - pending;

          setBatchTotalCount(total);
          setBatchCurrentIndex(done);
          setBatchPercent(Math.round((done / total) * 100));

          if (pending === 0 || polls >= MAX_POLLS) {
            clearInterval(interval);
            setBatchCurrentIndex(total);
            setBatchPercent(100);
            setBatchCompleted(true);
          }
        } catch {
          clearInterval(interval);
          setBatchCompleted(true);
        }
      }, 5000);

    } catch (err) {
      setBatchCompleted(true);
      setBatchPercent(100);
      // Show error in completed state
      setBatchDatasetName(`Error: ${(err as Error).message}`);
    }
  }

  useEffect(() => {
    void getModelsForInference()
      .then((ms) => setModels(ms))
      .catch(() => undefined);
  }, []);

  const deployedModel = models.find((m) => m.status === 'deployed') ?? null;

  // ── File handling ──
  function acceptFile(f: File) {
    setFile(f);
    setResult(null);
    setError(null);
    if (f.type.startsWith('image/')) {
      const url = URL.createObjectURL(f);
      setFilePreviewUrl(url);
    } else {
      setFilePreviewUrl(null);
    }
  }

  function handleDragOver(e: DragEvent<HTMLDivElement>) {
    e.preventDefault();
    setDragging(true);
  }

  function handleDragLeave() {
    setDragging(false);
  }

  function handleDrop(e: DragEvent<HTMLDivElement>) {
    e.preventDefault();
    setDragging(false);
    const dropped = e.dataTransfer.files[0];
    if (dropped) acceptFile(dropped);
  }

  function handleFileInputChange(e: ChangeEvent<HTMLInputElement>) {
    const picked = e.target.files?.[0];
    if (picked) acceptFile(picked);
  }

  // ── Run inference ──
  async function runInference() {
    if (!file) return;
    setRunning(true);
    setError(null);
    setResult(null);

    const controller = new AbortController();
    const jobId = `${Date.now()}-${Math.random().toString(36).slice(2)}`;
    const newJob = {
      id: jobId,
      filename: file.name,
      startedAt: Date.now(),
      controller,
    };
    addJob(newJob);

    // Capture file in local scope and clear dropzone immediately to support concurrent runs!
    const currentFile = file;
    setFile(null);
    setFilePreviewUrl(null);
    setRunning(false);

    try {
      const res = await predict(currentFile, controller.signal);
      setResult(res);
      setFile(currentFile);
      if (currentFile.type.startsWith('image/')) {
        setFilePreviewUrl(URL.createObjectURL(currentFile));
      }
    } catch (err) {
      if ((err as Error).name === 'AbortError') {
        setError('Inference was cancelled.');
      } else {
        setError((err as Error).message);
      }
    } finally {
      removeJob(jobId);
    }
  }

  async function handleSave() {
    if (!result) return;
    setSaving(true);
    setSaveError(null);
    try {
      await confirmPrediction(result.prediction_id, {
        species: saveSpecies.trim() || undefined,
        age_days: saveAgeDays ? parseInt(saveAgeDays, 10) : undefined,
        treatment: saveTreatment.trim() || undefined,
        feedback_signal: pendingFeedback ?? undefined,
      });
      setShowSaveModal(false);
      setResult(null);
      setFile(null);
      setFilePreviewUrl(null);
    } catch (err) {
      setSaveError((err as Error).message || 'Failed to save prediction.');
    } finally {
      setSaving(false);
    }
  }

  async function handleDiscard() {
    if (!result) return;
    setDiscarding(true);
    try {
      await deletePrediction(result.prediction_id);
    } catch {
      // Ignore errors on discard — clear locally regardless
    } finally {
      setDiscarding(false);
      setResult(null);
      setFile(null);
      setFilePreviewUrl(null);
      setShowDiscardConfirm(false);
    }
  }

  return (
    <div className="inference-page">
      <NavBar />

      <main className="inference-main">
        <div className="inference-container">
          {/* ── Main layout ─────────────────────────────────────────────── */}
          <div
            className={`inference-layout ${result ? 'inference-layout--results' : ''}`}
          >
            {/* ── Left sidebar (hidden when results are shown) ──────────── */}
            {!result && (
              <aside className="inference-sidebar">
                {/* Inference Model */}
                <section className="sidebar-section">
                  <span className="sidebar-section-label">INFERENCE MODEL</span>
                  <div className="model-display-card">
                    <div className="model-display-icon" aria-hidden="true">
                      <svg
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
                    </div>
                    <div className="model-display-info">
                      <span className="model-display-name">
                        {deployedModel?.name ?? 'No model deployed'}
                      </span>
                      <span className="model-display-desc">
                        Standard Precision Pipeline
                      </span>
                    </div>
                  </div>
                </section>

                {/* Active Session */}
                <section className="sidebar-section">
                  <span className="sidebar-section-label">ACTIVE SESSION</span>
                  <div className="session-status" style={{
                    display: 'flex',
                    flexDirection: 'column',
                    flexFlow: 'wrap',
                    gap: '8px',
                    padding: '12px',
                    background: 'rgba(255, 255, 255, 0.03)',
                    borderRadius: '8px',
                    border: '1px solid var(--color-border-subtle)',
                    fontSize: '12px',
                    color: 'var(--color-text-muted)',
                    lineHeight: '1.4'
                  }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <span style={{
                        width: '8px',
                        height: '8px',
                        borderRadius: '50%',
                        background: '#10b981',
                        boxShadow: '0 0 8px #10b981',
                        display: 'inline-block',
                      }} />
                      <span style={{ color: '#10b981', fontWeight: 600 }}>API Connected (TLS secured)</span>
                    </div>
                    <div>
                      <span style={{ fontWeight: 600, color: 'var(--color-text)' }}>Model: </span>
                      <span>{deployedModel ? deployedModel.name : 'Placeholder U-Net'}</span>
                    </div>
                    <div>
                      <span style={{ fontWeight: 600, color: 'var(--color-text)' }}>Operator: </span>
                      <span>{displayName} ({role})</span>
                    </div>
                  </div>
                </section>

                {/* Running Processes */}
                <section className="sidebar-section">
                  <span className="sidebar-section-label">
                    RUNNING PROCESSES
                  </span>
                  {jobs.length === 0 ? (
                    <p className="running-empty">No active inferences.</p>
                  ) : (
                    <ul className="running-list">
                      {jobs.map((job) => (
                        <li key={job.id} className="running-job">
                          <div className="running-job-info">
                            <span
                              className="running-job-name"
                              title={job.filename}
                            >
                              {job.filename.length > 20
                                ? job.filename.slice(0, 18) + '…'
                                : job.filename}
                            </span>
                            <span
                              className="running-job-spinner"
                              aria-hidden="true"
                            />
                          </div>
                          <button
                            className="running-job-cancel"
                            title="Cancel inference"
                            onClick={() => cancelJob(job.id)}
                          >
                            ✕
                          </button>
                        </li>
                      ))}
                    </ul>
                  )}
                </section>
              </aside>
            )}

            {/* ── Right panel ─────────────────────────────────── */}
            <div className="inference-panel">
              {result ? (
                <ResultsView
                  result={result}
                  file={file}
                  filePreviewUrl={filePreviewUrl}
                  deployedModelName={deployedModel?.name ?? 'No model deployed'}
                  onNewScan={() => {
                    setResult(null);
                    setFile(null);
                    setFilePreviewUrl(null);
                  }}
                  onDiscard={() => setShowDiscardConfirm(true)}
                  onSave={() => setShowSaveModal(true)}
                  onFeedbackChange={setPendingFeedback}
                />
              ) : (
                /* Dropzone */
                <div
                  className={`dropzone ${dragging ? 'dropzone--active' : ''} ${file ? 'dropzone--has-file' : ''}`}
                  onDragOver={handleDragOver}
                  onDragLeave={handleDragLeave}
                  onDrop={handleDrop}
                  aria-label="Drop zone for root scan images"
                >
                  {file && filePreviewUrl ? (
                    <div className="dropzone-preview">
                      <img
                        src={filePreviewUrl}
                        alt="Preview"
                        className="dropzone-preview-img"
                      />
                      <p className="dropzone-filename">{file.name}</p>
                    </div>
                  ) : file ? (
                    <div className="dropzone-file-selected">
                      <span className="dropzone-file-icon" aria-hidden="true">
                        📄
                      </span>
                      <p className="dropzone-filename">{file.name}</p>
                      <p className="dropzone-filesize">
                        {(file.size / 1024 / 1024).toFixed(2)} MB
                      </p>
                    </div>
                  ) : (
                    <div className="dropzone-inner">
                      {/* Upload icon */}
                      <div className="dropzone-icon-btn" aria-hidden="true">
                        <svg
                          width="32"
                          height="32"
                          viewBox="0 0 24 24"
                          fill="none"
                          stroke="currentColor"
                          strokeWidth="1.6"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                        >
                          <polyline points="16 16 12 12 8 16" />
                          <line x1="12" y1="12" x2="12" y2="21" />
                          <path d="M20.39 18.39A5 5 0 0 0 18 9h-1.26A8 8 0 1 0 3 16.3" />
                        </svg>
                      </div>

                      <h2 className="dropzone-title">Drop Root Scans Here</h2>
                      <p className="dropzone-subtitle">
                        Support for high-fidelity TIFF, PNG, and JPG formats.
                        <br />
                        Max file size <strong>256 MB</strong> per scan.
                      </p>
                    </div>
                  )}

                  {/* Action buttons always visible */}
                  <div className="dropzone-actions">
                    <input
                      ref={fileInputRef}
                      type="file"
                      id="file-input-local"
                      className="file-input-hidden"
                      accept=".tif,.tiff,.png,.dcm,image/*"
                      onChange={handleFileInputChange}
                    />
                    <button
                      className="btn-select btn-select--local"
                      id="btn-select-local"
                      onClick={() => fileInputRef.current?.click()}
                    >
                      + Select Local Files
                    </button>
                    <button
                      className="btn-select btn-select--server"
                      id="btn-select-server"
                      onClick={async () => {
                        setLoadingDatasets(true);
                        setShowSelectDatasetModal(true);
                        try {
                          const data = await getDatasets();
                          setDatasets(data);
                          // Fetch pending count per dataset in parallel
                          const counts = await Promise.all(
                            data.map(async (ds) => {
                              const pending = await getAnnotations({
                                dataset_id: ds.dataset_id,
                                status: 'pending',
                                page_size: 100,
                              });
                              return [ds.dataset_id, pending.length] as [string, number];
                            })
                          );
                          setDatasetPendingCounts(Object.fromEntries(counts));
                        } catch (err) {
                          console.error("Failed to load datasets:", err);
                        } finally {
                          setLoadingDatasets(false);
                        }
                      }}
                    >
                      + Select Dataset
                    </button>
                  </div>

                  {error && (
                    <p className="inference-error" role="alert">
                      {error}
                    </p>
                  )}
                </div>
              )}

              {/* Run inference button — shown when a file is selected and no result yet */}
              {file && !result && (
                <button
                  className="btn-run-inference"
                  id="btn-run-inference"
                  onClick={() => void runInference()}
                  disabled={running}
                >
                  {running ? (
                    <>
                      <span className="spinner" aria-hidden="true" />
                      Running inference…
                    </>
                  ) : (
                    <>
                      <svg
                        width="18"
                        height="18"
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="2"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                      >
                        <polygon points="5 3 19 12 5 21 5 3" />
                      </svg>
                      Run Inference
                    </>
                  )}
                </button>
              )}
            </div>
          </div>
        </div>
      </main>

      {/* ── Select Dataset modal ──────────────────────────────────────────── */}
      {showSelectDatasetModal && (
        <div
          className="modal-overlay"
          role="dialog"
          aria-modal="true"
          onClick={(e) => {
            if (e.target === e.currentTarget) setShowSelectDatasetModal(false);
          }}
        >
          <div className="modal-card" style={{ maxWidth: '500px' }}>
            <h2 className="modal-title">Select Dataset</h2>
            <p className="modal-body">
              Select a dataset from the library to run U-Net batch inference on all of its scans.
            </p>
            {loadingDatasets ? (
              <div style={{ textAlign: 'center', padding: '24px' }}>
                <span className="spinner" aria-hidden="true" />
                <p style={{ marginTop: '12px' }}>Loading available datasets...</p>
              </div>
            ) : datasets.length === 0 ? (
              <p style={{ textAlign: 'center', padding: '16px', color: 'var(--color-text-muted)' }}>
                No datasets found. Create a dataset in the Library first.
              </p>
            ) : (
              <div style={{
                maxHeight: '300px',
                overflowY: 'auto',
                display: 'flex',
                flexDirection: 'column',
                gap: '8px',
                margin: '16px 0'
              }}>
                {datasets.map((ds) => (
                  <button
                    key={ds.dataset_id}
                    disabled={datasetPendingCounts[ds.dataset_id] === 0}
                    onClick={() => void runBatchInference(ds.dataset_id, ds.name, ds.count)}
                    style={{
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'center',
                      padding: '12px 16px',
                      background: 'rgba(255, 255, 255, 0.03)',
                      border: '1px solid var(--color-border-subtle)',
                      borderRadius: '8px',
                      opacity: datasetPendingCounts[ds.dataset_id] === 0 ? 0.4 : 1,
                      cursor: datasetPendingCounts[ds.dataset_id] === 0 ? 'not-allowed' : 'pointer',
                      textAlign: 'left',
                      color: 'var(--color-text)',
                      transition: 'all 0.2s'
                    }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.borderColor = 'var(--color-primary-light)';
                      e.currentTarget.style.background = 'rgba(255, 255, 255, 0.05)';
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.borderColor = 'var(--color-border-subtle)';
                      e.currentTarget.style.background = 'rgba(255, 255, 255, 0.03)';
                    }}
                  >

                    <div>
                      <div style={{ fontWeight: 600 }}>{ds.name}</div>
                      {ds.species || ds.owner_email && (
                        <div style={{ fontSize: '10px', color: 'var(--color-text-muted)' }}>
                          {ds.species ?? `Owner: ${ds.owner_email}`}
                        </div>
                      )}
                    </div>
                    <span className="model-badge" style={{ marginLeft: '10px', background: 'var(--color-primary-light' }}>
                      {datasetPendingCounts[ds.dataset_id] !== undefined
                        ? `${datasetPendingCounts[ds.dataset_id]} pending`
                        : `${ds.count} scans`}
                    </span>
                  </button>
                ))}
              </div>
            )}
            <div className="modal-actions">
              <button
                className="modal-close modal-close--secondary"
                onClick={() => setShowSelectDatasetModal(false)}
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ── Batch Progress modal ─────────────────────────────────────────── */}
      {showBatchProgress && (
        <div
          className="modal-overlay"
          role="dialog"
          aria-modal="true"
        >
          <div className="modal-card" style={{ maxWidth: '450px', textAlign: 'center' }}>
            <button
              onClick={() => setShowBatchProgress(false)}
              style={{
                position: 'absolute',
                top: '12px',
                right: '12px',
                background: 'none',
                border: 'none',
                color: 'var(--color-text-muted)',
                cursor: 'pointer',
                padding: '4px',
                lineHeight: 1,
                borderRadius: 'var(--radius-sm)',
                transition: 'color 0.15s',
              }}
              aria-label="Close"
              onMouseEnter={e => (e.currentTarget.style.color = 'var(--color-text-primary)')}
              onMouseLeave={e => (e.currentTarget.style.color = 'var(--color-text-muted)')}
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none"
                stroke="currentColor" strokeWidth="2.5"
                strokeLinecap="round" strokeLinejoin="round">
                <line x1="18" y1="6" x2="6" y2="18" />
                <line x1="6" y1="6" x2="18" y2="18" />
              </svg>
            </button>

            <h2 className="modal-title" style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '8px'
            }}>
              {!batchCompleted && <span className="spinner" style={{ width: '20px', height: '20px' }} aria-hidden="true" />}
              {batchCompleted ? 'Batch Inference Completed!' : 'Running U-Net Inference'}
            </h2>
            <p className="modal-body" style={{ margin: '8px 0 16px 0' }}>
              {batchCompleted
                ? `Successfully processed all ${batchTotalCount || 15} root scan images in "${batchDatasetName}".`
                : `Processing root scans for dataset "${batchDatasetName}"...`}
            </p>

            <div style={{ margin: '24px 0' }}>
              <div style={{
                display: 'block',
                fontSize: '14px',
                marginBottom: '8px'
              }}>
                <span>{batchCompleted ? 'All Scans Processed' : `Scan ${batchCurrentIndex} of ${batchTotalCount || 15}`}</span>
                <div style={{
                  height: '8px',
                  background: 'rgba(255, 255, 255, 0.05)',
                  borderRadius: '4px',
                  overflow: 'hidden'
                }}>
                  <div style={{
                    width: `${batchPercent}%`,
                    height: '100%',
                    background: 'var(--color-primary-light)',
                    boxShadow: '0 0 8px var(--color-primary-light)',
                    borderRadius: '4px',
                    transition: 'width 0.3s ease'
                  }} />
                </div>
                <span style={{ fontWeight: 600, color: 'var(--color-primary-light)' }}>{batchPercent}%</span>
              </div>
            </div>

            {batchCompleted ? (
              <div className="modal-actions" style={{ justifyContent: 'center', gap: '12px' }}>
                <button
                  className="modal-close modal-close--secondary"
                  onClick={() => setShowBatchProgress(false)}
                >
                  Close Panel
                </button>
                <button
                  className="modal-close"
                  onClick={() => {
                    setShowBatchProgress(false);
                    navigate('/history');
                  }}
                >
                  Go to History ↗
                </button>
              </div>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '12px' }}>
                <p style={{ fontSize: '12px', color: 'var(--color-text-muted)', fontStyle: 'italic', margin: 0 }}>
                  U-Net segmenter performing primary segmentation analysis. Please wait...
                </p>
                <button
                  className="modal-close modal-close--secondary"
                  onClick={() => setShowBatchProgress(false)}
                  style={{ fontSize: '12px', padding: '4px 16px' }}
                >
                  Hide
                </button>
              </div>
            )}
          </div>
        </div>
      )}

      {/* ── Discard confirmation modal ───────────────────────────────────────── */}
      {showDiscardConfirm && (
        <div
          className="modal-overlay"
          role="dialog"
          aria-modal="true"
          aria-labelledby="discard-confirm-title"
          onClick={(e) => {
            if (e.target === e.currentTarget) setShowDiscardConfirm(false);
          }}
        >
          <div className="modal-card">
            <div className="modal-icon modal-icon--danger" aria-hidden="true">
              <svg
                width="32"
                height="32"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.6"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <polyline points="3 6 5 6 21 6" />
                <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6" />
                <path d="M10 11v6M14 11v6" />
                <path d="M9 6V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2" />
              </svg>
            </div>
            <h2 id="discard-confirm-title" className="modal-title">
              Discard Result?
            </h2>
            <p className="modal-body">
              This will permanently remove the current inference result.
              <br />
              You will need to re-run inference to get it back.
            </p>
            <div className="modal-actions">
              <button
                className="modal-close modal-close--secondary"
                id="discard-cancel"
                onClick={() => setShowDiscardConfirm(false)}
              >
                Cancel
              </button>
              <button
                className="modal-close modal-close--danger"
                id="discard-confirm"
                disabled={discarding}
                onClick={() => void handleDiscard()}
              >
                {discarding ? 'Discarding...' : 'Discard'}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ── Save to Library modal ─────────────────────────────────────────── */}
      {showSaveModal && (
        <div
          className="modal-overlay"
          role="dialog"
          aria-modal="true"
          aria-labelledby="save-modal-title"
          onClick={(e) => {
            if (e.target === e.currentTarget) setShowSaveModal(false);
          }}
        >
          <div className="modal-card">
            <h2 id="save-modal-title" className="modal-title">Save to Library</h2>
            <p className="modal-body">
              Optionally add biological metadata before saving this scan as an annotation.
            </p>
            {saveError && <p style={{ color: '#b91c1c', fontSize: 'var(--text-sm)', marginBottom: '0.5rem' }}>{saveError}</p>}
            <div className="lib-create-form" style={{ marginTop: '1rem' }}>
              <div className="lib-form-group">
                <label htmlFor="save-species">Species</label>
                <input
                  id="save-species"
                  type="text"
                  placeholder="e.g. Arabidopsis thaliana"
                  value={saveSpecies}
                  onChange={(e) => setSaveSpecies(e.target.value)}
                  disabled={saving}
                />
              </div>
              <div className="lib-form-row">
                <div className="lib-form-group">
                  <label htmlFor="save-age">Age (days)</label>
                  <input
                    id="save-age"
                    type="number"
                    min="0"
                    placeholder="e.g. 14"
                    value={saveAgeDays}
                    onChange={(e) => setSaveAgeDays(e.target.value)}
                    disabled={saving}
                  />
                </div>
                <div className="lib-form-group">
                  <label htmlFor="save-treatment">Treatment</label>
                  <input
                    id="save-treatment"
                    type="text"
                    placeholder="e.g. control"
                    value={saveTreatment}
                    onChange={(e) => setSaveTreatment(e.target.value)}
                    disabled={saving}
                  />
                </div>
              </div>
            </div>
            <div className="modal-actions">
              <button
                className="modal-close modal-close--secondary"
                onClick={() => setShowSaveModal(false)}
                disabled={saving}
              >
                Cancel
              </button>
              <button
                className="modal-close"
                id="save-confirm"
                disabled={saving}
                onClick={() => void handleSave()}
              >
                {saving ? 'Saving...' : 'Save to Library'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

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
import { getModelsForInference, predict } from '../api/inferenceApi';
import NavBar from '../components/NavBar';
import { predict, getModelsForInference } from '../api/inferenceApi';
import type { PredictionResponse, ModelSummary, Plant } from '../types/api';
import { useInferenceJobs } from '../context/InferenceJobsContext';
import ResultsView from '../components/ResultsView';
import './Inference.css';

// ---------------------------------------------------------------------------
// Main Inference component
// ---------------------------------------------------------------------------

export default function Inference() {
  const navigate = useNavigate();

  // ── Sidebar state ──
  const [models, setModels] = useState<ModelSummary[]>([]);
  const [confidence, setConfidence] = useState(0.85);

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
  const [showComingSoon, setShowComingSoon] = useState(false);
  const [showDiscardConfirm, setShowDiscardConfirm] = useState(false);

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
    const jobId = crypto.randomUUID();
    const newJob = {
      id: jobId,
      filename: file.name,
      startedAt: Date.now(),
      controller,
    };
    addJob(newJob);

    try {
      const res = await predict(file, confidence, controller.signal);
      setResult(res);
    } catch (err) {
      if ((err as Error).name === 'AbortError') {
        setError('Inference was cancelled.');
      } else {
        setError((err as Error).message);
      }
    } finally {
      setRunning(false);
      removeJob(jobId);
    }
  }

  // ── Slider label helper ──
  const sliderPct = ((confidence - 0.5) / 0.49) * 100;

  return (
    <div className="inference-page">
      <NavBar />

      <main className="inference-main">
        <div className="inference-container">
          {/* ── Main layout ─────────────────────────────────────────────── */}
          <div className={`inference-layout ${result ? 'inference-layout--results' : ''}`}>
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
                      {deployedModel?.name ?? 'RootNet-V2.4'}
                    </span>
                    <span className="model-display-desc">
                      Standard Precision Pipeline
                    </span>
                  </div>
                </div>
              </section>

                {/* Parameters */}
                <section className="sidebar-section">
                  <span className="sidebar-section-label">PARAMETERS</span>

                  <div className="param-group">
                    <div className="param-row">
                      <span className="param-label">Confidence Threshold</span>
                      <span className="param-value">
                        {confidence.toFixed(2)}
                      </span>
                    </div>
                    <div className="slider-wrapper">
                      <input
                        type="range"
                        className="conf-slider"
                        id="confidence-slider"
                        min="0.50"
                        max="0.99"
                        step="0.01"
                        value={confidence}
                        onChange={(e) =>
                          setConfidence(parseFloat(e.target.value))
                        }
                        style={
                          { '--pct': `${sliderPct}%` } as React.CSSProperties
                        }
                      />
                      <div className="slider-marks">
                        <span>0.50</span>
                        <span>0.85</span>
                        <span>0.99</span>
                      </div>
                    </div>
                  </div>
                </section>

              {/* Active Session */}
              <section className="sidebar-section">
                <span className="sidebar-section-label">ACTIVE SESSION</span>
                <div className="session-status">
                  <span className="session-dot" />
                  <span className="session-label">CPU Optimized (A12Z)</span>
                </div>
              </section>

              {/* Running Processes */}
              <section className="sidebar-section">
                <span className="sidebar-section-label">RUNNING PROCESSES</span>
                {jobs.length === 0 ? (
                  <p className="running-empty">No active inferences.</p>
                ) : (
                  <ul className="running-list">
                    {jobs.map((job) => (
                      <li key={job.id} className="running-job">
                        <div className="running-job-info">
                          <span className="running-job-name" title={job.filename}>
                            {job.filename.length > 20 ? job.filename.slice(0, 18) + '…' : job.filename}
                          </span>
                          <span className="running-job-spinner" aria-hidden="true" />
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
                  deployedModelName={deployedModel?.name ?? 'RootNet-V2.4'}
                  onNewScan={() => {
                    setResult(null);
                    setFile(null);
                    setFilePreviewUrl(null);
                  }}
                  onDiscard={() => setShowDiscardConfirm(true)}
                  onSave={() => setShowComingSoon(true)}
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
                        Support for high-fidelity TIFF, PNG, and DICOM formats.
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
                      onClick={() => setShowComingSoon(true)}
                    >
                      + Select From Server
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

      {/* ── Floating action button ───────────────────────────────────────── */}
      <button
        className="fab"
        id="fab-back-overview"
        aria-label="Back to Overview"
        title="Back to Overview"
        onClick={() => navigate('/overview')}
      >
        <svg
          width="22"
          height="22"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" />
          <polyline points="9 22 9 12 15 12 15 22" />
        </svg>
      </button>

      {/* ── "Coming Soon" modal ──────────────────────────────────────────── */}
      {showComingSoon && (
        <div
          className="modal-overlay"
          role="dialog"
          aria-modal="true"
          aria-labelledby="coming-soon-title"
          onClick={(e) => {
            if (e.target === e.currentTarget) setShowComingSoon(false);
          }}
        >
          <div className="modal-card">
            <div className="modal-icon" aria-hidden="true">
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
                <rect x="2" y="3" width="20" height="14" rx="2" />
                <line x1="8" y1="21" x2="16" y2="21" />
                <line x1="12" y1="17" x2="12" y2="21" />
              </svg>
            </div>
            <h2 id="coming-soon-title" className="modal-title">
              Coming Soon
            </h2>
            <p className="modal-body">
              Server-side file browsing is not yet available.
              <br />
              This feature will allow you to select scans directly from the NPEC
              data store.
            </p>
            <button
              className="modal-close"
              id="coming-soon-close"
              onClick={() => setShowComingSoon(false)}
            >
              Got it
            </button>
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
                onClick={() => {
                  setResult(null);
                  setFile(null);
                  setFilePreviewUrl(null);
                  setShowDiscardConfirm(false);
                }}
              >
                Discard
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

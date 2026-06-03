/**
 * pages/Library.tsx
 *
 * Library page — browse datasets, annotations, and exports.
 * Left sidebar: user info, new dataset button, navigation, filters.
 * Main panel: dataset grid or empty state.
 *
 * Route: /library (protected)
 */

import { useEffect, useState, useRef } from 'react';
import NavBar from '../components/NavBar';
import { getUsername, getRole } from '../api/auth';
import {
  getDatasets,
  createDataset,
  deleteDataset,
  getDatasetDetail,
  uploadDatasetImages,
  importDataset,
  getAnnotations,
  deleteAnnotation,
  runDatasetInference,
  updateAnnotationMetadata,
  submitAnnotationFeedback,
  getAnnotationDetail,
  runAnnotationInference,
  addAnnotationToDataset,
} from '../api/libraryApi';
import type {
  DatasetSummary,
  DatasetDetail,
  DatasetImportResponse,
} from '../api/libraryApi';
import type { AnnotationResponse, PredictionResponse } from '../types/api';
import {
  getPrediction,
  getUnconfirmedPredictions,
  confirmPrediction,
  deletePrediction,
} from '../api/inferenceApi';
import { getArtifactUrl } from '../api/client';
import ResultsView from '../components/ResultsView';
import { useInferenceJobs } from '../context/InferenceJobsContext';
import './Library.css';
import { useNavigate } from 'react-router-dom';

// ---------------------------------------------------------------------------
// Annotations Tab sub-component
// ---------------------------------------------------------------------------

interface AnnotationsTabProps {
  onSelect: (ann: AnnotationResponse) => void;
  statusFilter: string;
  growthFilter: string;
  speciesFilter: string;
  refreshKey?: number;
  onAnnotationsFetched?: (annotations: AnnotationResponse[]) => void;
  onHasAnnotationsChange?: (hasAny: boolean) => void;
  onClearFilters?: () => void;
}

function AnnotationsTab({
  onSelect,
  statusFilter,
  growthFilter,
  speciesFilter,
  refreshKey,
  onAnnotationsFetched,
  onHasAnnotationsChange,
  onClearFilters,
}: AnnotationsTabProps) {
  const [annotations, setAnnotations] = useState<AnnotationResponse[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [page, setPage] = useState(1);
  const [deleting, setDeleting] = useState<string | null>(null);
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [deleteTarget, setDeleteTarget] = useState<string | null>(null);

  useEffect(() => {
    void fetchAnnotations(page, statusFilter, growthFilter, speciesFilter);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [page, statusFilter, growthFilter, speciesFilter, refreshKey]);

  const prevFilters = useRef({ statusFilter, growthFilter, speciesFilter });
  useEffect(() => {
    const prev = prevFilters.current;
    if (
      prev.statusFilter !== statusFilter ||
      prev.growthFilter !== growthFilter ||
      prev.speciesFilter !== speciesFilter
    ) {
      prevFilters.current = { statusFilter, growthFilter, speciesFilter };
      setPage(1);
    }
  }, [statusFilter, growthFilter, speciesFilter]);

  async function fetchAnnotations(
    pageNum: number,
    status: string,
    growth: string,
    species: string,
  ) {
    setLoading(true);
    setError(null);
    try {
      const data = await getAnnotations({
        page: pageNum,
        page_size: 20,
        status: status || undefined,
        growth_stage: growth || undefined,
        species: species || undefined,
      });
      setAnnotations(data);
      onAnnotationsFetched?.(data);
      if (!status && !growth && !species) {
        onHasAnnotationsChange?.(data.length > 0);
      }
    } catch (err) {
      setError((err as Error).message || 'Failed to load annotations.');
    } finally {
      setLoading(false);
    }
  }

  function handleDelete(e: React.MouseEvent, annotationId: string) {
    e.stopPropagation();
    setDeleteTarget(annotationId);
    setShowDeleteModal(true);
  }

  async function executeDelete() {
    if (!deleteTarget) return;
    setDeleting(deleteTarget);
    try {
      await deleteAnnotation(deleteTarget);
      setShowDeleteModal(false);
      setDeleteTarget(null);
      await fetchAnnotations(page, statusFilter, growthFilter, speciesFilter);
    } catch (err) {
      setError((err as Error).message || 'Failed to delete annotation.');
    } finally {
      setDeleting(null);
    }
  }

  const statusColors: Record<string, string> = {
    pending: '#b45309',
    annotated: 'var(--color-tag-text)',
    review_required: '#b91c1c',
  };

  if (loading) {
    return (
      <div className="lib-empty">
        <span className="spinner" aria-hidden="true" />
        <p>Loading annotations...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="lib-empty">
        <p className="lib-empty-text" style={{ color: '#b91c1c' }}>
          {error}
        </p>
        <button
          className="modal-close"
          onClick={() =>
            void fetchAnnotations(
              page,
              statusFilter,
              growthFilter,
              speciesFilter,
            )
          }
          style={{ marginTop: '1rem' }}
        >
          Retry
        </button>
      </div>
    );
  }

  if (annotations.length === 0) {
    const hasAnyFilter =
      statusFilter !== '' || growthFilter !== '' || speciesFilter !== '';
    if (hasAnyFilter) {
      return (
        <div className="lib-empty">
          <p className="lib-empty-text">No annotations match your filters.</p>
          {onClearFilters && (
            <button
              className="modal-close"
              onClick={onClearFilters}
              style={{ marginTop: '1rem' }}
            >
              Clear Filters
            </button>
          )}
        </div>
      );
    }
    return (
      <div className="lib-empty">
        <p className="lib-empty-text">No annotations have been created yet.</p>
        <p className="lib-empty-hint">
          Run inference and save a result to create annotations.
        </p>
      </div>
    );
  }

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        gap: 'var(--space-4)',
      }}
    >
      <div style={{ overflowX: 'auto' }}>
        <table className="history-table">
          <thead>
            <tr>
              <th>Preview</th>
              <th>Annotation ID</th>
              <th>Prediction</th>
              <th>Status</th>
              <th>Species</th>
              <th>Age (days)</th>
              <th>Treatment</th>
              <th>Created</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {annotations.map((ann) => (
              <tr
                key={ann.annotation_id}
                onClick={() => onSelect(ann)}
                style={{ cursor: 'pointer' }}
                className="lib-annotation-row"
              >
                <td>
                  <img
                    src={getArtifactUrl(ann.image_uri)}
                    alt="Preview"
                    style={{
                      width: '40px',
                      height: '40px',
                      objectFit: 'cover',
                      borderRadius: '4px',
                      border: '1px solid var(--color-border-subtle)',
                      background: 'rgba(0,0,0,0.2)',
                    }}
                    onError={(e) => {
                      (e.target as HTMLImageElement).src =
                        'data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="40" height="40" viewBox="0 0 40 40"><rect width="40" height="40" fill="%23222"/><text x="20" y="24" font-family="sans-serif" font-size="8" fill="%23666" text-anchor="middle">N/A</text></svg>';
                    }}
                  />
                </td>
                <td
                  style={{
                    fontFamily: 'var(--font-mono)',
                    fontSize: 'var(--text-xs)',
                    color: 'var(--color-text-muted)',
                  }}
                >
                  {ann.annotation_id.slice(0, 8)}&hellip;
                </td>
                <td
                  style={{
                    fontFamily: 'var(--font-mono)',
                    fontSize: 'var(--text-xs)',
                    color: 'var(--color-text-muted)',
                  }}
                >
                  {ann.prediction_id
                    ? `${ann.prediction_id.slice(0, 8)}...`
                    : '—'}
                </td>
                <td>
                  <span
                    style={{
                      fontSize: 'var(--text-xs)',
                      fontWeight: 700,
                      color: statusColors[ann.status] ?? 'inherit',
                      textTransform: 'uppercase',
                      letterSpacing: '0.06em',
                    }}
                  >
                    {ann.status.replace('_', ' ')}
                  </span>
                </td>
                <td>{ann.species ?? '—'}</td>
                <td>{ann.age_days ?? '—'}</td>
                <td>{ann.treatment ?? '—'}</td>
                <td
                  style={{
                    fontSize: 'var(--text-xs)',
                    color: 'var(--color-text-muted)',
                  }}
                >
                  {new Date(ann.created_at).toLocaleDateString('en-GB', {
                    day: '2-digit',
                    month: 'short',
                    year: 'numeric',
                  })}
                </td>
                <td>
                  <button
                    className="delete-btn-icon"
                    disabled={deleting === ann.annotation_id}
                    onClick={(e) => handleDelete(e, ann.annotation_id)}
                    title="Delete annotation"
                  >
                    {deleting === ann.annotation_id ? (
                      <span
                        className="spinner"
                        style={{ width: '12px', height: '12px', margin: 0 }}
                        aria-hidden="true"
                      />
                    ) : (
                      <svg
                        xmlns="http://www.w3.org/2000/svg"
                        width="14"
                        height="14"
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="2"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        aria-hidden="true"
                      >
                        <path d="M3 6h18" />
                        <path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6" />
                        <path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2" />
                      </svg>
                    )}
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="lib-pagination">
        <button
          className="lib-page-btn"
          disabled={page === 1}
          onClick={() => setPage((p) => p - 1)}
          aria-label="Previous page"
        >
          <svg
            width="14"
            height="14"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2.5"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <polyline points="15 18 9 12 15 6" />
          </svg>
          Prev
        </button>

        <span className="lib-page-indicator">
          Page <strong>{page}</strong>
        </span>

        <button
          className="lib-page-btn"
          disabled={annotations.length < 20}
          onClick={() => setPage((p) => p + 1)}
          aria-label="Next page"
        >
          Next
          <svg
            width="14"
            height="14"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2.5"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <polyline points="9 18 15 12 9 6" />
          </svg>
        </button>
      </div>

      {showDeleteModal && deleteTarget && (
        <div
          className="modal-overlay"
          role="dialog"
          aria-modal="true"
          onClick={(e) => {
            if (e.target === e.currentTarget) {
              setShowDeleteModal(false);
              setDeleteTarget(null);
            }
          }}
        >
          <div className="modal-card" style={{ maxWidth: '400px' }}>
            <h2 className="modal-title" style={{ color: '#ef4444' }}>
              Delete Annotation
            </h2>
            <p className="modal-body">
              Are you sure you want to delete this annotation? This action
              cannot be undone.
            </p>
            <div
              className="modal-actions"
              style={{
                marginTop: '1.5rem',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'flex-end',
                gap: 'var(--space-3)',
              }}
            >
              <button
                className="modal-close modal-close--secondary"
                style={{
                  display: 'inline-flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  height: '38px',
                  padding: '0 20px',
                  borderRadius: 'var(--radius-md)',
                  fontWeight: 600,
                  fontSize: 'var(--text-sm)',
                  boxSizing: 'border-box',
                  margin: 0,
                  border: '1px solid transparent',
                }}
                onClick={() => {
                  setShowDeleteModal(false);
                  setDeleteTarget(null);
                }}
                disabled={deleting !== null}
              >
                Cancel
              </button>
              <button
                className="delete-btn"
                style={{
                  display: 'inline-flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  height: '38px',
                  padding: '0 20px',
                  borderRadius: 'var(--radius-md)',
                  fontWeight: 600,
                  fontSize: 'var(--text-sm)',
                  boxSizing: 'border-box',
                  margin: 0,
                  border: '1px solid #fca5a5',
                }}
                disabled={deleting !== null}
                onClick={() => void executeDelete()}
              >
                {deleting !== null ? 'Deleting…' : 'Delete'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

const _inferenceRunningDatasets = new Set<string>();
const _inferenceRunningAnnotations = new Set<string>();

// ---------------------------------------------------------------------------
// UnconfirmedPredictionsTab sub-component
// ---------------------------------------------------------------------------

interface UnconfirmedPredictionsTabProps {
  onCountChange: (count: number) => void;
}

function UnconfirmedPredictionsTab({
  onCountChange,
}: UnconfirmedPredictionsTabProps) {
  const [predictions, setPredictions] = useState<PredictionResponse[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [page, setPage] = useState(1);
  const PAGE_SIZE = 20;

  // Expanded result for inline review
  const [expandedId, setExpandedId] = useState<string | null>(null);

  // Save modal state
  const [showSaveModal, setShowSaveModal] = useState(false);
  const [saveTarget, setSaveTarget] = useState<string | null>(null);
  const [saveSpecies, setSaveSpecies] = useState('');
  const [saveAgeDays, setSaveAgeDays] = useState('');
  const [saveTreatment, setSaveTreatment] = useState('');
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);

  // Discard state
  const [discardingId, setDiscardingId] = useState<string | null>(null);
  const [showDiscardModal, setShowDiscardModal] = useState(false);
  const [discardTarget, setDiscardTarget] = useState<string | null>(null);
  const [discarding, setDiscarding] = useState(false);

  async function load(pageNum: number) {
    setLoading(true);
    setError(null);
    try {
      const data = await getUnconfirmedPredictions(PAGE_SIZE * pageNum);
      // Slice to the current page
      const pageData = data.slice(
        (pageNum - 1) * PAGE_SIZE,
        pageNum * PAGE_SIZE,
      );
      setPredictions(pageData);
      onCountChange(data.length);
    } catch (err) {
      setError(
        (err as Error).message || 'Failed to load unconfirmed predictions.',
      );
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    Promise.resolve().then(() => {
      void load(page);
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [page]);

  function openSave(predId: string) {
    setSaveTarget(predId);
    setSaveSpecies('');
    setSaveAgeDays('');
    setSaveTreatment('');
    setSaveError(null);
    setShowSaveModal(true);
  }

  async function handleSave() {
    if (!saveTarget) return;
    setSaving(true);
    setSaveError(null);
    try {
      await confirmPrediction(saveTarget, {
        species: saveSpecies.trim() || undefined,
        age_days: saveAgeDays ? parseInt(saveAgeDays, 10) : undefined,
        treatment: saveTreatment.trim() || undefined,
      });
      setShowSaveModal(false);
      setExpandedId(null);
      setSaveTarget(null);
      await load(page);
    } catch (err) {
      setSaveError((err as Error).message || 'Failed to save prediction.');
    } finally {
      setSaving(false);
    }
  }

  function handleDiscard(predId: string) {
    setDiscardTarget(predId);
    setShowDiscardModal(true);
  }

  async function handleDiscardConfirm() {
    if (!discardTarget) return;
    setDiscarding(true);
    setDiscardingId(discardTarget);
    try {
      await deletePrediction(discardTarget);
      if (expandedId === discardTarget) setExpandedId(null);
      setShowDiscardModal(false);
      setDiscardTarget(null);
      await load(page);
    } catch {
      // ignore
    } finally {
      setDiscarding(false);
      setDiscardingId(null);
    }
  }

  function relTime(iso: string): string {
    // eslint-disable-next-line react-hooks/purity
    const diff = Date.now() - new Date(iso).getTime();
    const m = Math.floor(diff / 60000);
    const h = Math.floor(m / 60);
    const d = Math.floor(h / 24);
    if (m < 1) return 'just now';
    if (m < 60) return `${m}m ago`;
    if (h < 24) return `${h}h ago`;
    if (d === 1) return 'Yesterday';
    return `${d}d ago`;
  }

  if (loading) {
    return (
      <div className="lib-empty">
        <span className="spinner" aria-hidden="true" />
        <p>Loading unconfirmed predictions…</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="lib-empty">
        <p className="lib-empty-text" style={{ color: '#b91c1c' }}>
          {error}
        </p>
        <button
          className="modal-close"
          onClick={() => void load(page)}
          style={{ marginTop: '1rem' }}
        >
          Retry
        </button>
      </div>
    );
  }

  if (predictions.length === 0) {
    return (
      <div>
        <div className="lib-content-header">
          <div className="lib-content-header-left">
            <span className="lib-breadcrumb">
              LIBRARY / UNCONFIRMED PREDICTIONS
            </span>
            <h1 className="lib-content-title">Unconfirmed Predictions</h1>
          </div>
        </div>
        <div className="lib-empty">
          <svg
            width="48"
            height="48"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.2"
            strokeLinecap="round"
            strokeLinejoin="round"
            style={{ opacity: 0.3 }}
          >
            <circle cx="12" cy="12" r="10" />
            <polyline points="12 6 12 12 16 14" />
          </svg>
          <p className="lib-empty-text">No unconfirmed predictions.</p>
          <p className="lib-empty-hint">
            Run inference and the result will appear here until you save or
            discard it.
          </p>
        </div>
      </div>
    );
  }

  const expandedPred =
    predictions.find((p) => p.prediction_id === expandedId) ?? null;

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        gap: 'var(--space-5)',
      }}
    >
      {/* Header */}
      <div className="lib-content-header">
        <div className="lib-content-header-left">
          <span className="lib-breadcrumb">
            LIBRARY / UNCONFIRMED PREDICTIONS
          </span>
          <h1 className="lib-content-title">Unconfirmed Predictions</h1>
        </div>
        <button
          className="modal-close modal-close--secondary"
          onClick={() => void load(page)}
          style={{ alignSelf: 'flex-end' }}
        >
          ↻ Refresh
        </button>
      </div>

      {/* Expanded result view */}
      {expandedPred && (
        <div
          style={{
            background: 'var(--color-bg-white)',
            border: '1px solid var(--color-primary-light)',
            borderRadius: 'var(--radius-lg)',
            padding: 'var(--space-5)',
            marginBottom: 'var(--space-2)',
          }}
        >
          <ResultsView
            result={expandedPred}
            file={null}
            filePreviewUrl={null}
            deployedModelName={expandedPred.metadata.model_version}
            onNewScan={() => setExpandedId(null)}
            onDiscard={() => void handleDiscard(expandedPred.prediction_id)}
            onSave={() => openSave(expandedPred.prediction_id)}
          />
        </div>
      )}

      {/* Table */}
      <div style={{ overflowX: 'auto' }}>
        <table className="history-table">
          <thead>
            <tr>
              <th>Preview</th>
              <th>Filename</th>
              <th>Plants</th>
              <th>Model</th>
              <th>When</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {predictions.map((pred) => (
              <tr
                key={pred.prediction_id}
                className={`lib-annotation-row ${expandedId === pred.prediction_id ? 'lib-pred-row--active' : ''}`}
                onClick={() =>
                  setExpandedId(
                    expandedId === pred.prediction_id
                      ? null
                      : pred.prediction_id,
                  )
                }
                style={{ cursor: 'pointer' }}
              >
                <td>
                  <img
                    src={getArtifactUrl(pred.input_url)}
                    alt="Scan preview"
                    style={{
                      width: '40px',
                      height: '40px',
                      objectFit: 'cover',
                      borderRadius: '4px',
                      border: '1px solid var(--color-border-subtle)',
                      background: 'rgba(0,0,0,0.2)',
                    }}
                    onError={(e) => {
                      (e.target as HTMLImageElement).src =
                        'data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="40" height="40" viewBox="0 0 40 40"><rect width="40" height="40" fill="%23222"/><text x="20" y="24" font-family="sans-serif" font-size="8" fill="%23666" text-anchor="middle">N/A</text></svg>';
                    }}
                  />
                </td>
                <td
                  style={{
                    fontFamily: 'var(--font-mono)',
                    fontSize: 'var(--text-xs)',
                    maxWidth: '200px',
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    whiteSpace: 'nowrap',
                  }}
                  title={pred.input_filename ?? pred.prediction_id}
                >
                  {pred.input_filename ?? pred.prediction_id.slice(0, 8) + '…'}
                </td>
                <td>
                  <span
                    style={{
                      background: 'var(--color-tag-bg)',
                      color: 'var(--color-primary)',
                      borderRadius: 'var(--radius-full)',
                      padding: '2px 8px',
                      fontSize: 'var(--text-xs)',
                      fontWeight: 700,
                    }}
                  >
                    {pred.plants.length} plants
                  </span>
                </td>
                <td
                  style={{
                    fontSize: 'var(--text-xs)',
                    color: 'var(--color-text-muted)',
                  }}
                >
                  {pred.metadata.model_version}
                </td>
                <td
                  style={{
                    fontSize: 'var(--text-xs)',
                    color: 'var(--color-text-muted)',
                  }}
                >
                  {relTime(
                    pred.metadata.inference_timestamp as unknown as string,
                  )}
                </td>
                <td onClick={(e) => e.stopPropagation()}>
                  <div
                    style={{
                      display: 'flex',
                      gap: '8px',
                      alignItems: 'center',
                    }}
                  >
                    <button
                      className="modal-close"
                      style={{
                        display: 'inline-flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        fontSize: 'var(--text-xs)',
                        padding: '0 14px',
                        fontWeight: 600,
                        letterSpacing: '0.03em',
                        height: '28px',
                        boxSizing: 'border-box',
                        marginTop: '0px',
                      }}
                      onClick={() => openSave(pred.prediction_id)}
                    >
                      Save
                    </button>
                    <button
                      className="delete-btn"
                      style={{
                        display: 'inline-flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        padding: '6px',
                        borderRadius: 'var(--radius-md)',
                        background: 'rgba(239, 68, 68, 0.1)',
                        color: '#ef4444',
                        border: '1px solid rgba(239, 68, 68, 0.2)',
                        cursor: 'pointer',
                        transition: 'all 0.2s ease',
                      }}
                      disabled={discardingId === pred.prediction_id}
                      onClick={() => handleDiscard(pred.prediction_id)}
                      title="Discard prediction"
                    >
                      {discardingId === pred.prediction_id ? (
                        <span
                          className="spinner"
                          style={{ width: '12px', height: '12px' }}
                        />
                      ) : (
                        <svg
                          width="14"
                          height="14"
                          viewBox="0 0 24 24"
                          fill="none"
                          stroke="currentColor"
                          strokeWidth="2.5"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                        >
                          <polyline points="3 6 5 6 21 6"></polyline>
                          <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
                          <line x1="10" y1="11" x2="10" y2="17"></line>
                          <line x1="14" y1="11" x2="14" y2="17"></line>
                        </svg>
                      )}
                    </button>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Pagination */}
      <div className="lib-pagination">
        <button
          className="lib-page-btn"
          disabled={page === 1}
          onClick={() => setPage((p) => p - 1)}
          aria-label="Previous page"
        >
          <svg
            width="14"
            height="14"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2.5"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <polyline points="15 18 9 12 15 6" />
          </svg>
          Prev
        </button>
        <span className="lib-page-indicator">
          Page <strong>{page}</strong>
        </span>
        <button
          className="lib-page-btn"
          disabled={predictions.length < PAGE_SIZE}
          onClick={() => setPage((p) => p + 1)}
          aria-label="Next page"
        >
          Next
          <svg
            width="14"
            height="14"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2.5"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <polyline points="9 18 15 12 9 6" />
          </svg>
        </button>
      </div>

      {/* Save modal */}
      {showSaveModal && (
        <div
          className="modal-overlay"
          role="dialog"
          aria-modal="true"
          onClick={(e) => {
            if (e.target === e.currentTarget) setShowSaveModal(false);
          }}
        >
          <div className="modal-card">
            <h2 className="modal-title">Save to Library</h2>
            <p className="modal-body">
              Optionally add biological metadata before saving this scan as an
              annotation.
            </p>
            {saveError && (
              <p
                style={{
                  color: '#b91c1c',
                  fontSize: 'var(--text-sm)',
                  marginBottom: '0.5rem',
                }}
              >
                {saveError}
              </p>
            )}
            <div className="lib-create-form" style={{ marginTop: '1rem' }}>
              <div className="lib-form-group">
                <label htmlFor="upred-species">Species</label>
                <input
                  id="upred-species"
                  type="text"
                  placeholder="e.g. Arabidopsis thaliana"
                  value={saveSpecies}
                  onChange={(e) => setSaveSpecies(e.target.value)}
                  disabled={saving}
                />
              </div>
              <div className="lib-form-row">
                <div className="lib-form-group">
                  <label htmlFor="upred-age">Age (days)</label>
                  <input
                    id="upred-age"
                    type="number"
                    min="0"
                    placeholder="e.g. 14"
                    value={saveAgeDays}
                    onChange={(e) => setSaveAgeDays(e.target.value)}
                    disabled={saving}
                  />
                </div>
                <div className="lib-form-group">
                  <label htmlFor="upred-treatment">Treatment</label>
                  <input
                    id="upred-treatment"
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
                id="upred-save-confirm"
                disabled={saving}
                onClick={() => void handleSave()}
              >
                {saving ? 'Saving…' : 'Save to Library'}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Discard confirmation modal */}
      {showDiscardModal && (
        <div
          className="modal-overlay"
          role="dialog"
          aria-modal="true"
          onClick={(e) => {
            if (e.target === e.currentTarget) setShowDiscardModal(false);
          }}
        >
          <div className="modal-card" style={{ maxWidth: '400px' }}>
            <h2 className="modal-title" style={{ color: '#ef4444' }}>
              Discard Prediction
            </h2>
            <p className="modal-body">
              Are you sure you want to discard this prediction? This action
              cannot be undone and the scan result will be permanently deleted.
            </p>
            <div
              className="modal-actions"
              style={{
                marginTop: '1.5rem',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'flex-end',
                gap: 'var(--space-3)',
              }}
            >
              <button
                className="modal-close modal-close--secondary"
                style={{
                  display: 'inline-flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  height: '38px',
                  padding: '0 20px',
                  borderRadius: 'var(--radius-md)',
                  fontWeight: 600,
                  fontSize: 'var(--text-sm)',
                  boxSizing: 'border-box',
                  margin: 0,
                  border: '1px solid transparent',
                }}
                onClick={() => setShowDiscardModal(false)}
                disabled={discarding}
              >
                Cancel
              </button>
              <button
                className="delete-btn"
                style={{
                  display: 'inline-flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  height: '38px',
                  padding: '0 20px',
                  borderRadius: 'var(--radius-md)',
                  fontWeight: 600,
                  fontSize: 'var(--text-sm)',
                  boxSizing: 'border-box',
                  margin: 0,
                  border: '1px solid #fca5a5',
                }}
                disabled={discarding}
                onClick={() => void handleDiscardConfirm()}
              >
                {discarding ? 'Discarding…' : 'Yes, Discard'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

type LibraryTab = 'datasets' | 'annotations' | 'predictions';

export default function Library() {
  const navigate = useNavigate();
  const username = getUsername();
  const role = getRole();
  const displayName = username.includes('@')
    ? username
        .split('@')[0]
        .replace(/[._-]/g, ' ')
        .replace(/\b\w/g, (c) => c.toUpperCase())
    : username;

  const { jobs, addJob, removeJob, isDatasetRunning } = useInferenceJobs();

  // ── State ──
  const [activeTab, setActiveTab] = useState<LibraryTab>(() => {
    const params = new URLSearchParams(window.location.search);
    const tabParam = params.get('tab');
    if (
      tabParam === 'annotations' ||
      tabParam === 'datasets' ||
      tabParam === 'predictions'
    ) {
      return tabParam as LibraryTab;
    }
    return 'datasets';
  });

  const [unconfirmedCount, setUnconfirmedCount] = useState(0);
  const [hasDatasetsOverall, setHasDatasetsOverall] = useState(true);
  const [hasAnnotations, setHasAnnotations] = useState(true);

  const [datasets, setDatasets] = useState<DatasetSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [speciesFilter, setSpeciesFilter] = useState('');
  const [growthStages, setGrowthStages] = useState<string[]>([]);
  const [dbUp, setDbUp] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const [sortOrder, setSortOrder] = useState<'desc' | 'asc'>('desc');
  const [annotStatusFilter, setAnnotStatusFilter] = useState<string>('');
  const [annotSpeciesFilter, setAnnotSpeciesFilter] = useState('');
  const [annotGrowthFilter, setAnnotGrowthFilter] = useState('');
  const [annotRefreshKey, setAnnotRefreshKey] = useState(0);
  const [runningAnnotationIds, setRunningAnnotationIds] = useState<Set<string>>(
    new Set(),
  );
  const [feedbackSaving, setFeedbackSaving] = useState(false);
  const annotationModalClosedRef = useRef<string | null>(null);
  const [targetDatasetId, setTargetDatasetId] = useState('');
  const [addingToDataset, setAddingToDataset] = useState(false);
  const [addToDatasetError, setAddToDatasetError] = useState<string | null>(
    null,
  );
  const [annotationFeedback, setAnnotationFeedback] = useState<
    'correct' | 'incorrect' | null
  >(null);

  const [inferenceRunning, setInferenceRunning] = useState(false);
  const [inferenceMessage, setInferenceMessage] = useState<string | null>(null);
  const pollingRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const [showCreateModal, setShowCreateModal] = useState(false);
  const [createData, setCreateData] = useState({
    name: '',
    description: '',
    species: '',
    growth_stage: '',
  });
  const [createFiles, setCreateFiles] = useState<File[]>([]);
  const createFileInputRef = useRef<HTMLInputElement>(null);
  const [createError, setCreateError] = useState<string | null>(null);
  const [creating, setCreating] = useState(false);

  const [datasetToDelete, setDatasetToDelete] = useState<DatasetSummary | null>(
    null,
  );
  const [deleteConfirmName, setDeleteConfirmName] = useState('');
  const [deleteError, setDeleteError] = useState<string | null>(null);
  const [deleting, setDeleting] = useState(false);

  const [selectedDatasetId, setSelectedDatasetId] = useState<string | null>(
    null,
  );
  const [datasetDetail, setDatasetDetail] = useState<DatasetDetail | null>(
    null,
  );
  const [detailLoading, setDetailLoading] = useState(false);
  const [detailScans, setDetailScans] = useState<AnnotationResponse[]>([]);

  // Selected annotation modal details
  const [selectedAnnotation, setSelectedAnnotation] =
    useState<AnnotationResponse | null>(null);
  const [linkedPrediction, setLinkedPrediction] =
    useState<PredictionResponse | null>(null);
  const [linkedPredictionLoading, setLinkedPredictionLoading] = useState(false);
  const [imageView, setImageView] = useState<'scan' | 'mask' | 'tip'>('scan');
  const [lightboxSrc, setLightboxSrc] = useState<string | null>(null);
  const [scanPage, setScanPage] = useState(1);
  const SCANS_PER_PAGE = 20;

  // Editing annotation details
  const [editingMetaField, setEditingMetaField] = useState<string | null>(null);
  const [editingMetaValue, setEditingMetaValue] = useState('');
  const [metaSaving, setMetaSaving] = useState(false);

  // Upload Modal State
  const [showUploadModal, setShowUploadModal] = useState(false);
  const [uploadFiles, setUploadFiles] = useState<File[]>([]);
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Import Dataset Wizard State
  const [showImportModal, setShowImportModal] = useState(false);
  const [importStep, setImportStep] = useState<1 | 2 | 3>(1);
  const [importZipFile, setImportZipFile] = useState<File | null>(null);
  const [importTargetMode, setImportTargetMode] = useState<'new' | 'existing'>(
    'new',
  );
  const [importDatasetId, setImportDatasetId] = useState<string>('');
  const [importNewName, setImportNewName] = useState('');
  const [importNewDesc, setImportNewDesc] = useState('');
  const [importNewSpecies, setImportNewSpecies] = useState('');
  const [importNewGrowth, setImportNewGrowth] = useState('');
  const [importProgress, setImportProgress] = useState(0);
  const [importProcessing, setImportProcessing] = useState(false);
  const [importReport, setImportReport] =
    useState<DatasetImportResponse | null>(null);
  const [importError, setImportError] = useState<string | null>(null);
  const zipInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const fetchDatasets = async () => {
      await Promise.resolve();
      setLoading(true);
      try {
        const data = await getDatasets({
          species: speciesFilter || undefined,
          growth_stage: growthStages.length === 1 ? growthStages[0] : undefined,
        });
        setDatasets(data);
        setDbUp(true);
        if (!speciesFilter && growthStages.length === 0) {
          setHasDatasetsOverall(data.length > 0);
        }
      } catch (err) {
        setDatasets([]);
        const message = (err as Error).message ?? '';
        setDbUp(
          !message.includes('fetch') && !message.includes('NetworkError'),
        );
      } finally {
        setLoading(false);
      }
    };

    fetchDatasets();
  }, [speciesFilter, growthStages]);

  // Fetch unconfirmed count for badge with a periodic 5-second poll so it updates dynamically as inference jobs complete
  useEffect(() => {
    const fetchCount = () => {
      void getUnconfirmedPredictions(100)
        .then((preds) => setUnconfirmedCount(preds.length))
        .catch(() => undefined);
    };
    fetchCount();
    const interval = setInterval(fetchCount, 5000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    Promise.resolve().then(() => {
      setScanPage(1);
    });
    const fetchDetail = async () => {
      if (selectedDatasetId) {
        await Promise.resolve();
        setDetailLoading(true);
        try {
          const detail = await getDatasetDetail(selectedDatasetId);
          setDatasetDetail(detail);
          const scans = await getAnnotations({
            dataset_id: selectedDatasetId,
            page: 1,
            page_size: SCANS_PER_PAGE,
          });
          setDetailScans(scans);
        } catch (error) {
          console.error(error);
        } finally {
          setDetailLoading(false);
        }
      } else {
        setDatasetDetail(null);
        setDetailScans([]);
      }
    };

    fetchDetail();
  }, [selectedDatasetId]);

  useEffect(() => {
    Promise.resolve().then(() => {
      setScanPage(1);
      setInferenceMessage(null);
      setInferenceRunning(false);

      if (pollingRef.current) {
        clearInterval(pollingRef.current);
        pollingRef.current = null;
      }

      if (!selectedDatasetId) return;

      const isRunning =
        _inferenceRunningDatasets.has(selectedDatasetId) ||
        isDatasetRunning(selectedDatasetId);

      if (isRunning) {
        setInferenceRunning(true);
        setInferenceMessage(
          'Processing images in the background. The grid will update automatically.',
        );

        // Find the active jobId from the global context to allow removal when finished
        const globalJob = jobs.find(
          (j) => j.type === 'dataset' && j.datasetId === selectedDatasetId,
        );
        const jobId = globalJob?.id;

        // Poll every 5s
        const interval = setInterval(async () => {
          try {
            const scans = await getAnnotations({
              dataset_id: selectedDatasetId,
              page: 1, // check first page for pending scans
              page_size: SCANS_PER_PAGE,
            });
            setDetailScans(scans);

            const stillPending = scans.some((s) => s.status === 'pending');
            if (!stillPending) {
              clearInterval(interval);
              pollingRef.current = null;
              _inferenceRunningDatasets.delete(selectedDatasetId);
              setInferenceRunning(false);
              setInferenceMessage('All images processed successfully.');
              if (jobId) removeJob(jobId);
              // Refresh dataset details to update overall stats
              const detail = await getDatasetDetail(selectedDatasetId);
              setDatasetDetail(detail);
            }
          } catch {
            clearInterval(interval);
            pollingRef.current = null;
            _inferenceRunningDatasets.delete(selectedDatasetId);
            setInferenceRunning(false);
            if (jobId) removeJob(jobId);
          }
        }, 5000);

        pollingRef.current = interval;
      }
    });

    return () => {
      if (pollingRef.current) {
        clearInterval(pollingRef.current);
        pollingRef.current = null;
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedDatasetId, isDatasetRunning]);

  useEffect(() => {
    Promise.resolve().then(() => {
      setImageView('scan');
      setTargetDatasetId('');
      setAddToDatasetError(null);
    });
    const fetchLinkedPrediction = async () => {
      if (selectedAnnotation && selectedAnnotation.prediction_id) {
        annotationModalClosedRef.current = null;
        setLinkedPredictionLoading(true);
        try {
          const pred = await getPrediction(selectedAnnotation.prediction_id);
          setLinkedPrediction(pred);
          setAnnotationFeedback(pred.feedback_signal ?? null);
        } catch (err) {
          console.error('Failed to load linked prediction:', err);
          setLinkedPrediction(null);
          setAnnotationFeedback(null);
        } finally {
          setLinkedPredictionLoading(false);
        }
      } else {
        setLinkedPrediction(null);
        setAnnotationFeedback(null);
      }
    };
    fetchLinkedPrediction();
  }, [selectedAnnotation]);

  useEffect(() => {
    function handleKey(e: KeyboardEvent) {
      if (e.key === 'Escape') {
        if (lightboxSrc) setLightboxSrc(null);
      }
    }
    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, [lightboxSrc, selectedAnnotation]);

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    params.set('tab', activeTab);
    window.history.replaceState(null, '', `?${params.toString()}`);
  }, [activeTab]);

  useEffect(() => {
    if (!selectedDatasetId) return;
    const fetchScans = async () => {
      setDetailLoading(true);
      try {
        const scans = await getAnnotations({
          dataset_id: selectedDatasetId,
          page: scanPage,
          page_size: SCANS_PER_PAGE,
        });
        setDetailScans(scans);
      } catch (err) {
        console.error(err);
      } finally {
        setDetailLoading(false);
      }
    };
    void fetchScans();
  }, [scanPage, selectedDatasetId]);

  useEffect(() => {
    return () => {
      if (pollingRef.current) clearInterval(pollingRef.current);
    };
  }, []);

  function switchTab(tab: LibraryTab) {
    setActiveTab(tab);
    setSelectedDatasetId(null);
    setDatasetDetail(null);
    setDetailScans([]);
    if (tab !== 'annotations' && tab !== 'predictions') {
      setAnnotStatusFilter('');
      setAnnotGrowthFilter('');
      setAnnotSpeciesFilter('');
    }
  }

  async function handleCreateDataset(e: React.FormEvent) {
    e.preventDefault();
    setCreateError(null);
    setCreating(true);
    try {
      const newDataset = await createDataset({
        name: createData.name,
        description: createData.description || undefined,
        species: createData.species || undefined,
        growth_stage: createData.growth_stage || undefined,
      });

      // If user selected any files to add, upload them to the newly created dataset!
      if (createFiles.length > 0) {
        await uploadDatasetImages(newDataset.dataset_id, createFiles);
      }

      setShowCreateModal(false);
      setCreateData({
        name: '',
        description: '',
        species: '',
        growth_stage: '',
      });
      setCreateFiles([]); // Clear selected files
      // Reload datasets
      const data = await getDatasets({
        species: speciesFilter || undefined,
        growth_stage: growthStages.length === 1 ? growthStages[0] : undefined,
      });
      setDatasets(data);
    } catch (err) {
      setCreateError((err as Error).message);
    } finally {
      setCreating(false);
    }
  }

  async function handleMetaSave(field: string) {
    if (!selectedAnnotation) return;
    setMetaSaving(true);
    try {
      const body: Record<string, string | number> = {};
      if (field === 'species') body.species = editingMetaValue;
      if (field === 'treatment') body.treatment = editingMetaValue;
      if (field === 'age_days') body.age_days = parseInt(editingMetaValue) || 0;

      const updated = await updateAnnotationMetadata(
        selectedAnnotation.annotation_id,
        body,
      );
      setSelectedAnnotation(updated);
      setAnnotRefreshKey((k) => k + 1);
    } catch {
      // optionally show an error
    } finally {
      setMetaSaving(false);
      setEditingMetaField(null);
    }
  }

  async function handleRunInference() {
    if (!selectedDatasetId || !datasetDetail) return;
    if (
      _inferenceRunningDatasets.has(selectedDatasetId) ||
      isDatasetRunning(selectedDatasetId)
    )
      return;

    setInferenceRunning(true);
    setInferenceMessage(null);
    _inferenceRunningDatasets.add(selectedDatasetId);

    try {
      const res = await runDatasetInference(selectedDatasetId);

      // Parse count from message if present
      const countMatch = res.message.match(/(\d+)\s+image/);
      const count = countMatch ? parseInt(countMatch[1]) : null;

      if (res.message.toLowerCase().includes('already running')) {
        setInferenceMessage(
          'Inference is already in progress for this dataset.',
        );
        _inferenceRunningDatasets.delete(selectedDatasetId);
        setInferenceRunning(false);
        return;
      }

      if (res.message.toLowerCase().includes('no pending')) {
        setInferenceMessage('No pending images to process.');
        _inferenceRunningDatasets.delete(selectedDatasetId);
        setInferenceRunning(false);
        return;
      }

      setInferenceMessage(
        count
          ? `Processing ${count} image${count !== 1 ? 's' : ''} in the background. The grid will update automatically.`
          : 'Processing images in the background. The grid will update automatically.',
      );

      // Add to global jobs list to kick off the polling and sidebar status
      const jobId = `ds-${selectedDatasetId}-${Date.now()}`;
      addJob({
        id: jobId,
        filename: datasetDetail.name,
        startedAt: Date.now(),
        controller: new AbortController(),
        type: 'dataset',
        datasetId: selectedDatasetId,
        datasetName: datasetDetail.name,
      });
    } catch (err) {
      _inferenceRunningDatasets.delete(selectedDatasetId);
      setInferenceRunning(false);
      setInferenceMessage(
        `Failed to start inference: ${(err as Error).message}`,
      );
    }
  }

  async function handleRunAnnotationInference() {
    if (!selectedAnnotation) return;
    const annotationId = selectedAnnotation.annotation_id;
    if (_inferenceRunningAnnotations.has(annotationId)) return;

    addRunningAnnotation(annotationId);
    setSelectedAnnotation((prev) =>
      prev ? { ...prev, status: 'pending' } : null,
    );
    setAnnotRefreshKey((k) => k + 1);

    try {
      const res = await runAnnotationInference(annotationId);

      if (res.message.toLowerCase().includes('already running')) {
        removeRunningAnnotation(annotationId);
        setSelectedAnnotation((prev) =>
          prev ? { ...prev, status: 'annotated' } : null,
        );
        return;
      }

      // Poll every 2s for up to 5 minutes
      const interval = setInterval(async () => {
        try {
          const updated = await getAnnotationDetail(annotationId);
          if (annotationModalClosedRef.current !== annotationId) {
            setSelectedAnnotation((current) => {
              if (current && current.annotation_id === annotationId) {
                return updated;
              }
              return current;
            });
          }

          if (
            updated.status === 'annotated' ||
            updated.status === 'review_required'
          ) {
            clearInterval(interval);
            removeRunningAnnotation(annotationId);
            setAnnotRefreshKey((k) => k + 1);

            if (annotationModalClosedRef.current !== annotationId) {
              setSelectedAnnotation((current) => {
                if (current && current.annotation_id === annotationId) {
                  return updated;
                }
                return current;
              });
            }
          }
        } catch {
          clearInterval(interval);
          removeRunningAnnotation(annotationId);
        }
      }, 2000);

      // Safety timeout after 5 minutes
      setTimeout(
        () => {
          clearInterval(interval);
          removeRunningAnnotation(annotationId);
        },
        5 * 60 * 1000,
      );
    } catch {
      removeRunningAnnotation(annotationId);
      // Revert optimistic status update
      setSelectedAnnotation((prev) =>
        prev ? { ...prev, status: 'annotated' } : null,
      );
    }
  }

  function addRunningAnnotation(id: string) {
    _inferenceRunningAnnotations.add(id);
    setRunningAnnotationIds((prev) => new Set(prev).add(id));
  }

  function removeRunningAnnotation(id: string) {
    _inferenceRunningAnnotations.delete(id);
    setRunningAnnotationIds((prev) => {
      const next = new Set(prev);
      next.delete(id);
      return next;
    });
  }

  async function handleDeleteDataset(e: React.FormEvent) {
    e.preventDefault();
    if (!datasetToDelete) return;
    if (deleteConfirmName !== datasetToDelete.name) return;

    setDeleteError(null);
    setDeleting(true);
    try {
      await deleteDataset(datasetToDelete.dataset_id);
      setDatasetToDelete(null);
      setDeleteConfirmName('');
      // Reload datasets
      const data = await getDatasets({
        species: speciesFilter || undefined,
        growth_stage: growthStages.length === 1 ? growthStages[0] : undefined,
      });
      setDatasets(data);
    } catch (err) {
      setDeleteError((err as Error).message);
    } finally {
      setDeleting(false);
    }
  }

  function handleDragOver(e: React.DragEvent) {
    e.preventDefault();
    setIsDragging(true);
  }

  function handleDragLeave(e: React.DragEvent) {
    e.preventDefault();
    setIsDragging(false);
  }

  function handleDrop(e: React.DragEvent) {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      setUploadFiles((prev) => [...prev, ...Array.from(e.dataTransfer.files)]);
    }
  }

  function handleFileSelect(e: React.ChangeEvent<HTMLInputElement>) {
    if (e.target.files && e.target.files.length > 0) {
      setUploadFiles((prev) => [...prev, ...Array.from(e.target.files!)]);
    }
  }

  async function handleUpload() {
    if (!selectedDatasetId || uploadFiles.length === 0) return;

    // Validate before setting loading state
    const MAX_SIZE_MB = 50;
    const ACCEPTED_TYPES = [
      'image/png',
      'image/tiff',
      'image/tif',
      'image/jpeg',
      'image/jpg',
    ];

    for (const f of uploadFiles) {
      if (f.size > MAX_SIZE_MB * 1024 * 1024) {
        setUploadError(`File "${f.name}" exceeds the ${MAX_SIZE_MB}MB limit.`);
        return;
      }
      if (!ACCEPTED_TYPES.includes(f.type)) {
        setUploadError(
          `File "${f.name}" has an unsupported format. Use PNG, TIFF, or JPEG.`,
        );
        return;
      }
    }

    setIsUploading(true);
    setUploadError(null);
    try {
      const result = await uploadDatasetImages(selectedDatasetId, uploadFiles);

      if (result.failed > 0 && result.success === 0) {
        // All failed
        setUploadError(result.errors.join('\n'));
        return; // don't close modal
      }

      if (result.failed > 0 && result.success > 0) {
        // Partial success — close modal but show warning
        setShowUploadModal(false);
        setUploadFiles([]);
        setUploadError(
          `${result.success} image(s) uploaded. ${result.failed} failed:\n${result.errors.join('\n')}`,
        );
      } else {
        // Full success
        setShowUploadModal(false);
        setUploadFiles([]);
      }

      // Refresh either way if anything succeeded
      if (result.success > 0) {
        const detail = await getDatasetDetail(selectedDatasetId);
        setDatasetDetail(detail);
        setScanPage(1);
        const scans = await getAnnotations({
          dataset_id: selectedDatasetId,
          page: 1,
          page_size: SCANS_PER_PAGE,
        });
        setDetailScans(scans);
        const data = await getDatasets({
          species: speciesFilter || undefined,
          growth_stage: growthStages.length === 1 ? growthStages[0] : undefined,
        });
        setDatasets(data);
      }
    } catch (err) {
      setUploadError((err as Error).message);
    } finally {
      setIsUploading(false);
    }
  }

  function handleZipDragOver(e: React.DragEvent) {
    e.preventDefault();
    setIsDragging(true);
  }

  function handleZipDragLeave(e: React.DragEvent) {
    e.preventDefault();
    setIsDragging(false);
  }

  function handleZipDrop(e: React.DragEvent) {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const file = e.dataTransfer.files[0];
      if (file.name.endsWith('.zip')) {
        setImportZipFile(file);
        setImportStep(2);
      } else {
        setImportError('Please select a valid .zip file.');
      }
    }
  }

  function handleZipSelect(e: React.ChangeEvent<HTMLInputElement>) {
    if (e.target.files && e.target.files.length > 0) {
      const file = e.target.files[0];
      if (file.name.endsWith('.zip')) {
        setImportZipFile(file);
        setImportStep(2);
      } else {
        setImportError('Please select a valid .zip file.');
      }
    }
  }

  async function handleImportSubmit() {
    if (!importZipFile) return;
    setImportStep(3);
    setImportProcessing(true);
    setImportProgress(10);
    setImportError(null);

    // Animate progress up to 90% while uploading/waiting
    const interval = setInterval(() => {
      setImportProgress((prev) => {
        if (prev >= 90) {
          clearInterval(interval);
          return 90;
        }
        return prev + 15;
      });
    }, 400);

    try {
      const res = await importDataset(importZipFile, {
        targetMode: importTargetMode,
        datasetId:
          importTargetMode === 'existing' ? importDatasetId : undefined,
        name: importTargetMode === 'new' ? importNewName : undefined,
        description: importTargetMode === 'new' ? importNewDesc : undefined,
        species: importTargetMode === 'new' ? importNewSpecies : undefined,
        growthStage: importTargetMode === 'new' ? importNewGrowth : undefined,
      });

      clearInterval(interval);
      setImportProgress(100);
      setImportReport(res);

      // Refresh list
      const data = await getDatasets({
        species: speciesFilter || undefined,
        growth_stage: growthStages.length === 1 ? growthStages[0] : undefined,
      });
      setDatasets(data);
    } catch (err) {
      clearInterval(interval);
      setImportError((err as Error).message);
      setImportStep(2); // Go back to config
    } finally {
      setImportProcessing(false);
    }
  }

  function resetImportWizard() {
    setShowImportModal(false);
    setImportStep(1);
    setImportZipFile(null);
    setImportTargetMode('new');
    setImportDatasetId('');
    setImportNewName('');
    setImportNewDesc('');
    setImportNewSpecies('');
    setImportNewGrowth('');
    setImportProgress(0);
    setImportProcessing(false);
    setImportReport(null);
    setImportError(null);
  }

  const filteredDatasets = datasets
    .filter((ds) => {
      if (!searchQuery) return true;
      const lowerQ = searchQuery.toLowerCase();
      return (
        ds.name.toLowerCase().includes(lowerQ) ||
        ds.dataset_id.toLowerCase().includes(lowerQ)
      );
    })
    .sort((a, b) => {
      const timeA = new Date(a.created_at).getTime();
      const timeB = new Date(b.created_at).getTime();
      return sortOrder === 'desc' ? timeB - timeA : timeA - timeB;
    });

  // Tab labels for sidebar nav
  const navItems: {
    key: LibraryTab;
    label: string;
    icon: JSX.Element;
    badge?: number;
  }[] = [
    {
      key: 'datasets',
      label: 'Datasets',
      icon: (
        <svg
          width="16"
          height="16"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.8"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <ellipse cx="12" cy="5" rx="9" ry="3" />
          <path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3" />
          <path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5" />
        </svg>
      ),
    },
    {
      key: 'annotations',
      label: 'Annotations',
      icon: (
        <svg
          width="16"
          height="16"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.8"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <path d="M12 20h9" />
          <path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z" />
        </svg>
      ),
    },
    {
      key: 'predictions' as LibraryTab,
      label: 'Unconfirmed',
      badge: unconfirmedCount,
      icon: (
        <svg
          width="16"
          height="16"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.8"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <circle cx="12" cy="12" r="10" />
          <polyline points="12 6 12 12 16 14" />
        </svg>
      ),
    },
  ];

  return (
    <div className="library-page">
      <NavBar />

      <main className="library-main">
        <div className="library-container">
          {/* ── Sidebar ────────────────────────────────────────── */}
          <aside className="lib-sidebar">
            {/* User card */}
            <div className="lib-user-card">
              <div className="lib-user-avatar">
                <svg
                  width="20"
                  height="20"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.8"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" />
                  <circle cx="12" cy="7" r="4" />
                </svg>
              </div>
              <div className="lib-user-info">
                <span className="lib-user-name">{displayName}</span>
                <span className="lib-user-db">
                  Database:{' '}
                  <span
                    className={`lib-db-badge ${dbUp ? 'lib-db-badge--up' : 'lib-db-badge--down'}`}
                  >
                    {dbUp ? 'UP' : 'DOWN'}
                  </span>
                </span>
              </div>
            </div>

            {/* New Dataset & Import buttons */}
            {role === 'admin' && (
              <div
                style={{
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '8px',
                  marginBottom: '1.5rem',
                  width: '100%',
                }}
              >
                <button
                  className="lib-btn-new"
                  id="btn-new-dataset"
                  onClick={() => setShowCreateModal(true)}
                >
                  + New Dataset
                </button>
                <button
                  className="lib-btn-import"
                  id="btn-import-dataset"
                  onClick={() => setShowImportModal(true)}
                >
                  <svg
                    width="14"
                    height="14"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2.5"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    style={{ marginRight: '6px' }}
                  >
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                    <polyline points="17 8 12 3 7 8" />
                    <line x1="12" y1="3" x2="12" y2="15" />
                  </svg>
                  Import Dataset
                </button>
              </div>
            )}

            {/* Navigation */}
            <div className="lib-nav-section">
              <span className="lib-nav-label">NAVIGATION</span>
              <ul className="lib-nav-list">
                {navItems.map((item) => (
                  <li key={item.key}>
                    <button
                      className={`lib-nav-item ${activeTab === item.key ? 'lib-nav-item--active' : ''}`}
                      onClick={() => switchTab(item.key)}
                    >
                      {item.icon}
                      {item.label}
                      {item.badge !== undefined && item.badge > 0 && (
                        <span className="lib-nav-badge">{item.badge}</span>
                      )}
                    </button>
                  </li>
                ))}
              </ul>
            </div>

            {/* Filters */}
            {((activeTab === 'datasets' && hasDatasetsOverall) ||
              (activeTab === 'annotations' && hasAnnotations)) && (
              <div className="lib-filter-section">
                <span className="lib-nav-label">FILTERS</span>

                {activeTab === 'datasets' ? (
                  <div className="lib-filter-group">
                    <span className="lib-filter-label">GROWTH STAGE</span>
                    <div className="lib-filter-chips">
                      {[
                        { value: '', label: 'All' },
                        { value: 'seedling', label: 'Seedling' },
                        { value: 'mature', label: 'Mature' },
                      ].map((opt) => (
                        <button
                          key={opt.value}
                          className={`lib-chip ${(growthStages.length === 0 && opt.value === '') || growthStages[0] === opt.value ? 'lib-chip--active' : ''}`}
                          onClick={() =>
                            setGrowthStages(opt.value === '' ? [] : [opt.value])
                          }
                        >
                          {opt.label}
                        </button>
                      ))}
                    </div>
                  </div>
                ) : (
                  <div
                    style={{
                      display: 'flex',
                      flexDirection: 'column',
                      gap: 'var(--space-3)',
                    }}
                  >
                    <div className="lib-filter-group">
                      <span className="lib-filter-label">STATUS</span>
                      <div className="lib-filter-chips">
                        {[
                          { value: '', label: 'All' },
                          { value: 'pending', label: 'Pending' },
                          { value: 'annotated', label: 'Annotated' },
                          { value: 'review_required', label: 'Review Req.' },
                        ].map((opt) => (
                          <button
                            key={opt.value}
                            className={`lib-chip ${annotStatusFilter === opt.value ? 'lib-chip--active' : ''}`}
                            onClick={() => setAnnotStatusFilter(opt.value)}
                          >
                            {opt.label}
                          </button>
                        ))}
                      </div>
                    </div>

                    <div className="lib-filter-group">
                      <span className="lib-filter-label">GROWTH STAGE</span>
                      <div className="lib-filter-chips">
                        {[
                          { value: '', label: 'All' },
                          { value: 'seedling', label: 'Seedling' },
                          { value: 'mature', label: 'Mature' },
                        ].map((opt) => (
                          <button
                            key={opt.value}
                            className={`lib-chip ${annotGrowthFilter === opt.value ? 'lib-chip--active' : ''}`}
                            onClick={() => setAnnotGrowthFilter(opt.value)}
                          >
                            {opt.label}
                          </button>
                        ))}
                      </div>
                    </div>

                    <div className="lib-filter-group">
                      <span className="lib-filter-label">SPECIES</span>
                      <input
                        type="text"
                        className="lib-search-input"
                        placeholder="e.g. Arabidopsis"
                        value={annotSpeciesFilter}
                        onChange={(e) => setAnnotSpeciesFilter(e.target.value)}
                        style={{
                          fontSize: 'var(--text-xs)',
                          padding: 'var(--space-2) var(--space-3)',
                        }}
                      />
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Footer links */}
            <div className="lib-sidebar-footer">
              <button className="lib-footer-link">
                <svg
                  width="14"
                  height="14"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.8"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  <circle cx="12" cy="12" r="10" />
                  <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3" />
                  <line x1="12" y1="17" x2="12.01" y2="17" />
                </svg>
                Support
              </button>
              <button className="lib-footer-link">
                <svg
                  width="14"
                  height="14"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.8"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  <path d="M21 8v13H3V8" />
                  <path d="M1 3h22v5H1z" />
                </svg>
                Archive
              </button>
            </div>
          </aside>

          {/* ── Content panel ──────────────────────────────────── */}
          <div className="lib-content">
            {activeTab === 'predictions' ? (
              <UnconfirmedPredictionsTab onCountChange={setUnconfirmedCount} />
            ) : selectedDatasetId && activeTab === 'datasets' ? (
              /* ── Detail View ── */
              detailLoading || !datasetDetail ? (
                <div className="lib-empty">
                  <span className="spinner" aria-hidden="true" />
                  <p>Loading dataset details...</p>
                </div>
              ) : (
                <div className="lib-detail-view">
                  <div className="lib-detail-header">
                    <button
                      className="lib-back-btn"
                      onClick={() => setSelectedDatasetId(null)}
                    >
                      ← DATASETS
                    </button>
                    <div className="lib-detail-header-top">
                      <h1 className="lib-content-title">
                        {datasetDetail.name}
                      </h1>
                      <button
                        className="lib-btn-add-scans"
                        onClick={() => {
                          setUploadError(null);
                          setShowUploadModal(true);
                        }}
                      >
                        <svg
                          width="16"
                          height="16"
                          viewBox="0 0 24 24"
                          fill="none"
                          stroke="currentColor"
                          strokeWidth="2"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                        >
                          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                          <polyline points="17 8 12 3 7 8" />
                          <line x1="12" y1="3" x2="12" y2="15" />
                        </svg>
                        Add Scans
                      </button>
                    </div>
                    <div className="lib-ds-list-meta">
                      <span className="lib-ds-list-meta-item">
                        <span className="lib-ds-meta-label">Species:</span>{' '}
                        <span className="lib-ds-meta-val">
                          {datasetDetail.species ?? 'Arabidopsis'}
                        </span>
                      </span>
                    </div>
                  </div>

                  <div className="lib-detail-body">
                    <div className="lib-detail-main">
                      <div className="lib-detail-card">
                        <h3 className="lib-detail-card-title">Description</h3>
                        <p className="lib-detail-desc">
                          {datasetDetail.description ||
                            'No description provided.'}
                        </p>
                      </div>
                      <div className="lib-detail-card">
                        <div
                          style={{
                            display: 'flex',
                            justifyContent: 'space-between',
                            alignItems: 'center',
                            marginBottom: 'var(--space-3)',
                          }}
                        >
                          <h3 className="lib-detail-card-title">
                            Scans ({datasetDetail.count.toLocaleString()})
                          </h3>
                          {/* Page indicator when there are multiple pages */}
                          {datasetDetail.count > SCANS_PER_PAGE && (
                            <span
                              style={{
                                fontSize: 'var(--text-xs)',
                                color: 'var(--color-text-muted)',
                              }}
                            >
                              {(scanPage - 1) * SCANS_PER_PAGE + 1}–
                              {Math.min(
                                scanPage * SCANS_PER_PAGE,
                                datasetDetail.count,
                              )}{' '}
                              of {datasetDetail.count}
                            </span>
                          )}
                        </div>
                        {uploadError && (
                          <div className="lib-upload-warning">
                            {uploadError.split('\n').map((line, i) => (
                              <p key={i} style={{ margin: '2px 0' }}>
                                {line}
                              </p>
                            ))}
                            <button
                              className="lib-upload-warning-dismiss"
                              onClick={() => setUploadError(null)}
                              aria-label="Dismiss"
                            >
                              ✕
                            </button>
                          </div>
                        )}
                        <div className="lib-scans-grid">
                          {detailScans.length === 0 ? (
                            <p
                              style={{
                                gridColumn: '1 / -1',
                                color: 'var(--color-text-muted)',
                                fontSize: '0.85rem',
                              }}
                            >
                              No scans uploaded yet. Click &quot;Add Scans&quot;
                              above to add images.
                            </p>
                          ) : (
                            detailScans.map((scan) => (
                              <div
                                key={scan.annotation_id}
                                className="lib-scan-card"
                                onClick={() => setSelectedAnnotation(scan)}
                                style={{
                                  cursor: 'pointer',
                                  background: 'rgba(255, 255, 255, 0.02)',
                                  border:
                                    '1px solid var(--color-border-subtle)',
                                  borderRadius: '8px',
                                  padding: '8px',
                                  display: 'flex',
                                  flexDirection: 'column',
                                  gap: '8px',
                                  transition:
                                    'transform 0.2s, border-color 0.2s',
                                }}
                                onMouseEnter={(e) => {
                                  e.currentTarget.style.borderColor =
                                    'var(--color-primary-light)';
                                  e.currentTarget.style.transform =
                                    'translateY(-2px)';
                                }}
                                onMouseLeave={(e) => {
                                  e.currentTarget.style.borderColor =
                                    'var(--color-border-subtle)';
                                  e.currentTarget.style.transform =
                                    'translateY(0)';
                                }}
                              >
                                <img
                                  src={getArtifactUrl(scan.image_uri)}
                                  alt="Root Scan"
                                  className="lib-scan-thumbnail"
                                  style={{
                                    width: '100%',
                                    aspectRatio: '1',
                                    objectFit: 'cover',
                                    borderRadius: '4px',
                                    background: '#000',
                                  }}
                                  onError={(e) => {
                                    (e.target as HTMLImageElement).src =
                                      'data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100" viewBox="0 0 100 100"><rect width="100" height="100" fill="%23222"/><text x="50" y="55" font-family="sans-serif" font-size="12" fill="%23666" text-anchor="middle">No Preview</text></svg>';
                                  }}
                                />
                                <div
                                  style={{
                                    display: 'flex',
                                    justifyContent: 'space-between',
                                    alignItems: 'center',
                                    fontSize: '11px',
                                  }}
                                >
                                  <span
                                    style={{
                                      fontSize: '8px',
                                      fontFamily: 'var(--font-mono)',
                                      color: 'var(--color-text-muted)',
                                    }}
                                  >
                                    {scan.annotation_id.slice(0, 8)}...
                                  </span>
                                  <span
                                    style={{
                                      fontWeight: 700,
                                      color:
                                        scan.status === 'pending'
                                          ? '#b45309'
                                          : scan.status === 'annotated'
                                            ? 'var(--color-tag-text)'
                                            : '#b91c1c',
                                      textTransform: 'uppercase',
                                      fontSize: '9px',
                                    }}
                                  >
                                    {scan.status.replace('_', ' ')}
                                  </span>
                                </div>
                              </div>
                            ))
                          )}
                        </div>

                        {/* Pagination */}
                        {datasetDetail.count > SCANS_PER_PAGE && (
                          <div
                            className="lib-pagination"
                            style={{ marginTop: 'var(--space-4)' }}
                          >
                            <button
                              className="lib-page-btn"
                              disabled={scanPage === 1}
                              onClick={() => setScanPage((p) => p - 1)}
                              aria-label="Previous page"
                            >
                              <svg
                                width="14"
                                height="14"
                                viewBox="0 0 24 24"
                                fill="none"
                                stroke="currentColor"
                                strokeWidth="2.5"
                                strokeLinecap="round"
                                strokeLinejoin="round"
                              >
                                <polyline points="15 18 9 12 15 6" />
                              </svg>
                              Prev
                            </button>

                            <span className="lib-page-indicator">
                              Page <strong>{scanPage}</strong> of{' '}
                              <strong>
                                {Math.ceil(
                                  datasetDetail.count / SCANS_PER_PAGE,
                                )}
                              </strong>
                            </span>

                            <button
                              className="lib-page-btn"
                              disabled={
                                scanPage * SCANS_PER_PAGE >= datasetDetail.count
                              }
                              onClick={() => setScanPage((p) => p + 1)}
                              aria-label="Next page"
                            >
                              Next
                              <svg
                                width="14"
                                height="14"
                                viewBox="0 0 24 24"
                                fill="none"
                                stroke="currentColor"
                                strokeWidth="2.5"
                                strokeLinecap="round"
                                strokeLinejoin="round"
                              >
                                <polyline points="9 18 15 12 9 6" />
                              </svg>
                            </button>
                          </div>
                        )}
                      </div>
                    </div>

                    <div className="lib-detail-side">
                      <div className="lib-detail-card">
                        <h3 className="lib-detail-card-title">STATS</h3>
                        <div className="lib-stats-grid">
                          <div className="lib-stat-box">
                            <span className="lib-stat-label">Total Scans</span>
                            <span className="lib-stat-val">
                              {datasetDetail.count.toLocaleString()}
                            </span>
                          </div>
                        </div>

                        <div className="lib-page-stats">
                          <span className="lib-page-stats-label">
                            This page
                          </span>
                          <div className="lib-page-stats-row">
                            <span className="lib-page-stat lib-page-stat--annotated">
                              <span className="lib-page-stat-num">
                                {
                                  detailScans.filter(
                                    (s) => s.status === 'annotated',
                                  ).length
                                }
                              </span>
                              annotated
                            </span>
                            <span className="lib-page-stat lib-page-stat--pending">
                              <span className="lib-page-stat-num">
                                {
                                  detailScans.filter(
                                    (s) => s.status === 'pending',
                                  ).length
                                }
                              </span>
                              pending
                            </span>
                            <span className="lib-page-stat lib-page-stat--review">
                              <span className="lib-page-stat-num">
                                {
                                  detailScans.filter(
                                    (s) => s.status === 'review_required',
                                  ).length
                                }
                              </span>
                              review
                            </span>
                          </div>
                        </div>
                      </div>

                      <div className="lib-detail-card">
                        <h3 className="lib-detail-card-title">METADATA</h3>
                        <div className="lib-meta-grid">
                          {datasetDetail.growth_stage != null && (
                            <div className="lib-meta-row">
                              <span className="lib-meta-k">Species</span>
                              <span className="lib-meta-v">
                                {datasetDetail.species || '—'}
                              </span>
                            </div>
                          )}
                          {datasetDetail.growth_stage != null && (
                            <div className="lib-meta-row">
                              <span className="lib-meta-k">Growth Stage</span>
                              <span className="lib-meta-v">
                                {datasetDetail.growth_stage || '—'}
                              </span>
                            </div>
                          )}
                          <div className="lib-meta-row">
                            <span className="lib-meta-k">Created</span>
                            <span className="lib-meta-v">
                              {new Date(
                                datasetDetail.created_at,
                              ).toLocaleDateString('en-GB', {
                                day: '2-digit',
                                month: 'short',
                                year: 'numeric',
                              })}
                            </span>
                          </div>
                          <div className="lib-meta-row">
                            <span className="lib-meta-k">Last Modified</span>
                            <span className="lib-meta-v">
                              {new Date(
                                datasetDetail.modified_at,
                              ).toLocaleDateString('en-GB', {
                                day: '2-digit',
                                month: 'short',
                                year: 'numeric',
                              })}
                            </span>
                          </div>
                        </div>
                      </div>

                      {/* Run Inference button */}
                      {detailScans.some((s) => s.status === 'pending') && (
                        <div>
                          <button
                            className="lib-btn-run-inference"
                            onClick={() => void handleRunInference()}
                            disabled={
                              inferenceRunning ||
                              _inferenceRunningDatasets.has(
                                selectedDatasetId ?? '',
                              ) ||
                              isDatasetRunning(selectedDatasetId ?? '')
                            }
                          >
                            {inferenceRunning ||
                            isDatasetRunning(selectedDatasetId ?? '') ? (
                              <>
                                <span
                                  className="spinner"
                                  style={{ width: '12px', height: '12px' }}
                                />
                                Running...
                              </>
                            ) : (
                              <>
                                <svg
                                  width="13"
                                  height="13"
                                  viewBox="0 0 24 24"
                                  fill="none"
                                  stroke="currentColor"
                                  strokeWidth="2.5"
                                  strokeLinecap="round"
                                  strokeLinejoin="round"
                                >
                                  <polygon points="5 3 19 12 5 21 5 3" />
                                </svg>
                                Run Inference
                              </>
                            )}
                          </button>
                          {inferenceMessage && (
                            <p className="lib-inference-msg">
                              {inferenceMessage}
                            </p>
                          )}
                        </div>
                      )}
                      {/* Review in Lab button */}
                      {detailScans.some(
                        (s) => s.status === 'review_required',
                      ) && (
                        <button
                          className="lib-btn-review-lab"
                          onClick={() =>
                            navigate(`/lab?dataset_id=${selectedDatasetId}`)
                          }
                        >
                          <svg
                            width="13"
                            height="13"
                            viewBox="0 0 24 24"
                            fill="none"
                            stroke="currentColor"
                            strokeWidth="2.5"
                            strokeLinecap="round"
                            strokeLinejoin="round"
                          >
                            <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7" />
                            <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z" />
                          </svg>
                          Review{' '}
                          {
                            detailScans.filter(
                              (s) => s.status === 'review_required',
                            ).length
                          }{' '}
                          in Lab
                        </button>
                      )}
                    </div>
                  </div>
                </div>
              )
            ) : (
              /* ── List View ── */
              <>
                {/* Header bar */}
                <div className="lib-content-header">
                  <div className="lib-content-header-left">
                    <span className="lib-breadcrumb">
                      LIBRARY / {activeTab.toUpperCase()}
                    </span>
                    <h1 className="lib-content-title">
                      {activeTab.charAt(0).toUpperCase() + activeTab.slice(1)}
                    </h1>
                  </div>
                </div>

                {/* Toolbar (Search, Filter, Sort) */}
                {activeTab === 'datasets' &&
                  (datasets.length > 0 ||
                    speciesFilter ||
                    searchQuery ||
                    growthStages.length > 0) && (
                    <div className="lib-toolbar">
                      <div className="lib-toolbar-left">
                        <input
                          type="text"
                          className="lib-search-input"
                          placeholder="Search by name or ID"
                          value={searchQuery}
                          onChange={(e) => setSearchQuery(e.target.value)}
                        />
                      </div>
                      <div className="lib-toolbar-right">
                        <button
                          className="lib-sort-btn"
                          id="btn-sort"
                          onClick={() =>
                            setSortOrder((prev) =>
                              prev === 'desc' ? 'asc' : 'desc',
                            )
                          }
                        >
                          <svg
                            width="14"
                            height="14"
                            viewBox="0 0 24 24"
                            fill="none"
                            stroke="currentColor"
                            strokeWidth="2"
                            strokeLinecap="round"
                            strokeLinejoin="round"
                          >
                            {sortOrder === 'desc' ? (
                              <>
                                <line x1="12" y1="5" x2="12" y2="19" />
                                <polyline points="19 12 12 19 5 12" />
                              </>
                            ) : (
                              <>
                                <line x1="12" y1="19" x2="12" y2="5" />
                                <polyline points="5 12 12 5 19 12" />
                              </>
                            )}
                          </svg>
                          Sort {sortOrder === 'desc' ? '(Newest)' : '(Oldest)'}
                        </button>
                      </div>
                    </div>
                  )}

                {/* Dataset grid or empty state */}
                {activeTab === 'datasets' && (
                  <div className="lib-dataset-area">
                    {loading ? (
                      <div className="lib-empty">
                        <span className="spinner" aria-hidden="true" />
                        <p>Loading datasets…</p>
                      </div>
                    ) : datasets.length === 0 &&
                      !speciesFilter &&
                      !searchQuery &&
                      growthStages.length === 0 ? (
                      <div className="lib-empty">
                        <p className="lib-empty-text">
                          No datasets have been created yet.
                        </p>
                        <p className="lib-empty-hint">
                          Click &quot;+ New Dataset&quot; to get started.
                        </p>
                      </div>
                    ) : filteredDatasets.length === 0 ? (
                      <div className="lib-empty">
                        <p className="lib-empty-text">
                          No datasets match your filters.
                        </p>
                        <button
                          className="modal-close"
                          onClick={() => {
                            setLoading(true);
                            setSearchQuery('');
                            setSpeciesFilter('');
                            setGrowthStages([]);
                          }}
                          style={{ marginTop: '1rem' }}
                        >
                          Clear Filters
                        </button>
                      </div>
                    ) : (
                      <div className="lib-dataset-list">
                        {filteredDatasets.map((ds) => (
                          <div
                            key={ds.dataset_id}
                            className="lib-dataset-list-item"
                            onClick={() => setSelectedDatasetId(ds.dataset_id)}
                          >
                            <div className="lib-ds-list-title">
                              {ds.name}
                              {role === 'admin' && (
                                <button
                                  className="lib-ds-delete-btn"
                                  title="Delete dataset"
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    setDatasetToDelete(ds);
                                    setDeleteConfirmName('');
                                    setDeleteError(null);
                                  }}
                                >
                                  <svg
                                    width="14"
                                    height="14"
                                    viewBox="0 0 24 24"
                                    fill="none"
                                    stroke="currentColor"
                                    strokeWidth="2"
                                    strokeLinecap="round"
                                    strokeLinejoin="round"
                                  >
                                    <polyline points="3 6 5 6 21 6" />
                                    <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
                                    <line x1="10" y1="11" x2="10" y2="17" />
                                    <line x1="14" y1="11" x2="14" y2="17" />
                                  </svg>
                                </button>
                              )}
                            </div>
                            <div className="lib-ds-list-meta">
                              <span className="lib-ds-list-meta-item">
                                <span className="lib-ds-meta-label">
                                  Species:
                                </span>{' '}
                                <span className="lib-ds-meta-val">
                                  {ds.species ?? '—'}
                                </span>
                              </span>
                              <span className="lib-ds-list-meta-item">
                                <span className="lib-ds-meta-label">
                                  Owner:
                                </span>{' '}
                                <span className="lib-ds-meta-val">
                                  {ds.owner_email}
                                </span>
                              </span>
                              <span className="lib-ds-list-meta-item">
                                <span className="lib-ds-meta-label">
                                  created at:
                                </span>{' '}
                                <span className="lib-ds-meta-val">
                                  {new Date(ds.created_at)
                                    .toISOString()
                                    .split('.')[0] + 'Z'}
                                </span>
                              </span>
                              <span className="lib-ds-list-meta-item">
                                <span className="lib-ds-meta-label">
                                  count:
                                </span>{' '}
                                <span className="lib-ds-meta-val">
                                  {ds.count}
                                </span>
                              </span>
                              {/* Inference status pill */}
                              {isDatasetRunning(ds.dataset_id) ? (
                                <span
                                  style={{
                                    display: 'inline-flex',
                                    alignItems: 'center',
                                    gap: '4px',
                                    fontSize: '0.65rem',
                                    fontWeight: 700,
                                    padding: '2px 8px',
                                    borderRadius: 'var(--radius-full)',
                                    background: 'rgba(27,67,50,0.12)',
                                    color: 'var(--color-primary)',
                                  }}
                                >
                                  <span
                                    className="running-job-spinner"
                                    style={{
                                      width: '7px',
                                      height: '7px',
                                      borderColor: 'rgba(27,67,50,0.25)',
                                      borderTopColor: 'var(--color-primary)',
                                    }}
                                    aria-hidden="true"
                                  />
                                  Running
                                </span>
                              ) : (
                                <span
                                  style={{
                                    display: 'inline-flex',
                                    alignItems: 'center',
                                    fontSize: '0.65rem',
                                    fontWeight: 700,
                                    padding: '2px 8px',
                                    borderRadius: 'var(--radius-full)',
                                    background:
                                      ds.count === 0
                                        ? 'var(--color-bg-subtle)'
                                        : 'var(--color-tag-bg)',
                                    color:
                                      ds.count === 0
                                        ? 'var(--color-text-muted)'
                                        : 'var(--color-primary)',
                                  }}
                                >
                                  {ds.count === 0 ? 'Empty' : 'Not started'}
                                </span>
                              )}
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )}

                {activeTab === 'annotations' && (
                  <AnnotationsTab
                    onSelect={(ann) => setSelectedAnnotation(ann)}
                    statusFilter={annotStatusFilter}
                    growthFilter={annotGrowthFilter}
                    speciesFilter={annotSpeciesFilter}
                    refreshKey={annotRefreshKey}
                    onAnnotationsFetched={(fresh) => {
                      setSelectedAnnotation((prev) => {
                        if (!prev) return null;
                        const updated = fresh.find(
                          (a) => a.annotation_id === prev.annotation_id,
                        );
                        return updated ?? prev;
                      });
                    }}
                    onHasAnnotationsChange={setHasAnnotations}
                    onClearFilters={() => {
                      setAnnotStatusFilter('');
                      setAnnotGrowthFilter('');
                      setAnnotSpeciesFilter('');
                    }}
                  />
                )}
              </>
            )}
          </div>
        </div>
      </main>

      {/* ── Create Dataset Modal ────────────────────────────────────── */}
      {showCreateModal && (
        <div
          className="modal-overlay"
          role="dialog"
          aria-modal="true"
          onClick={(e) => {
            if (e.target === e.currentTarget) setShowCreateModal(false);
          }}
        >
          <div className="modal-card">
            <h2 className="modal-title">Create New Dataset</h2>
            <p className="modal-body">
              Enter high-level details for this root-scan dataset.
            </p>
            <form
              className="lib-create-form"
              onSubmit={(e) => void handleCreateDataset(e)}
            >
              <div className="lib-form-group">
                <label htmlFor="ds-name">Dataset Name *</label>
                <input
                  type="text"
                  id="ds-name"
                  required
                  placeholder="e.g. Spring 2026 Trial"
                  value={createData.name}
                  onChange={(e) =>
                    setCreateData({ ...createData, name: e.target.value })
                  }
                />
              </div>

              <div className="lib-form-group">
                <label htmlFor="ds-desc">Description</label>
                <textarea
                  id="ds-desc"
                  placeholder="Optional context or notes..."
                  value={createData.description}
                  onChange={(e) =>
                    setCreateData({
                      ...createData,
                      description: e.target.value,
                    })
                  }
                  rows={3}
                />
              </div>

              <div className="lib-form-row">
                <div className="lib-form-group">
                  <label htmlFor="ds-species">Species</label>
                  <input
                    type="text"
                    id="ds-species"
                    placeholder="e.g. Arabidopsis thaliana"
                    value={createData.species}
                    onChange={(e) =>
                      setCreateData({ ...createData, species: e.target.value })
                    }
                  />
                </div>
                <div className="lib-form-group">
                  <label htmlFor="ds-growth">Growth Stage</label>
                  <select
                    id="ds-growth"
                    value={createData.growth_stage}
                    onChange={(e) =>
                      setCreateData({
                        ...createData,
                        growth_stage: e.target.value,
                      })
                    }
                  >
                    <option value="">(None)</option>
                    <option value="seedling">Seedling</option>
                    <option value="mature">Mature</option>
                  </select>
                </div>
              </div>

              <div className="lib-form-group" style={{ marginTop: '1rem' }}>
                <label>Add Images (Optional)</label>
                <input
                  ref={createFileInputRef}
                  type="file"
                  multiple
                  accept="image/*"
                  onChange={(e) => {
                    if (e.target.files && e.target.files.length > 0) {
                      setCreateFiles(Array.from(e.target.files));
                    }
                  }}
                  style={{ display: 'none' }}
                />
                <div
                  style={{ display: 'flex', gap: '8px', alignItems: 'center' }}
                >
                  <button
                    type="button"
                    className="lib-btn-new"
                    style={{ margin: 0, padding: '6px 12px', fontSize: '12px' }}
                    onClick={() => createFileInputRef.current?.click()}
                  >
                    Select Files...
                  </button>
                  <span
                    style={{
                      fontSize: '12px',
                      color: 'var(--color-text-muted)',
                    }}
                  >
                    {createFiles.length === 0
                      ? 'No files selected'
                      : `${createFiles.length} files selected`}
                  </span>
                </div>
                {createFiles.length > 0 && (
                  <div
                    style={{
                      maxHeight: '120px',
                      overflowY: 'auto',
                      marginTop: '8px',
                      padding: '8px',
                      background: 'rgba(255, 255, 255, 0.03)',
                      border: '1px solid var(--color-border-subtle)',
                      borderRadius: '4px',
                      fontSize: '11px',
                      display: 'flex',
                      flexDirection: 'column',
                      gap: '4px',
                    }}
                  >
                    {createFiles.map((file, idx) => (
                      <div
                        key={idx}
                        style={{
                          display: 'flex',
                          justifyContent: 'space-between',
                          color: 'var(--color-text-muted)',
                        }}
                      >
                        <span
                          style={{
                            overflow: 'hidden',
                            textOverflow: 'ellipsis',
                            whiteSpace: 'nowrap',
                            maxWidth: '250px',
                          }}
                        >
                          {file.name}
                        </span>
                        <span>{(file.size / 1024 / 1024).toFixed(2)} MB</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              {createError && <p className="lib-form-error">{createError}</p>}

              <div className="modal-actions">
                <button
                  type="button"
                  className="modal-close modal-close--secondary"
                  onClick={() => setShowCreateModal(false)}
                  disabled={creating}
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="modal-close"
                  disabled={creating || !createData.name.trim()}
                >
                  {creating ? 'Creating...' : 'Create Dataset'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* ── Delete Dataset Modal ────────────────────────────────────── */}
      {datasetToDelete && (
        <div
          className="modal-overlay"
          role="dialog"
          aria-modal="true"
          onClick={(e) => {
            if (e.target === e.currentTarget && !deleting)
              setDatasetToDelete(null);
          }}
        >
          <div className="modal-card modal-card--danger">
            <h2 className="modal-title">Delete Dataset</h2>
            <p className="modal-body">
              Are you sure you want to delete the dataset{' '}
              <strong>{datasetToDelete.name}</strong>? This action cannot be
              undone and will fail if the dataset has associated annotations.
            </p>
            <p
              className="modal-body"
              style={{ marginTop: '1rem', fontWeight: 600 }}
            >
              Please type the name of the dataset to confirm:
            </p>
            <form
              className="lib-create-form"
              onSubmit={(e) => void handleDeleteDataset(e)}
            >
              <div className="lib-form-group">
                <input
                  type="text"
                  required
                  placeholder={datasetToDelete.name}
                  value={deleteConfirmName}
                  onChange={(e) => setDeleteConfirmName(e.target.value)}
                  autoComplete="off"
                />
              </div>

              {deleteError && <p className="lib-form-error">{deleteError}</p>}

              <div className="modal-actions">
                <button
                  type="button"
                  className="modal-close modal-close--secondary"
                  onClick={() => setDatasetToDelete(null)}
                  disabled={deleting}
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="modal-close modal-close--danger"
                  disabled={
                    deleting || deleteConfirmName !== datasetToDelete.name
                  }
                >
                  {deleting ? 'Deleting...' : 'Delete Permanently'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* ── Add Images Upload Modal ─────────────────────────────────── */}
      {showUploadModal && (
        <div
          className="modal-overlay"
          role="dialog"
          aria-modal="true"
          onClick={(e) => {
            if (e.target === e.currentTarget && !isUploading) {
              setShowUploadModal(false);
            }
          }}
        >
          <div className="modal-card">
            <h2 className="modal-title">Upload Scans</h2>
            <p className="modal-body">
              Add new root scans to <strong>{datasetDetail?.name}</strong>.
              These will automatically be processed by the current active model.
            </p>

            <div
              className={`lib-upload-dropzone ${isDragging ? 'lib-upload-dropzone--active' : ''}`}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
            >
              <div className="lib-upload-icon">
                <svg
                  width="32"
                  height="32"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                  <polyline points="17 8 12 3 7 8" />
                  <line x1="12" y1="3" x2="12" y2="15" />
                </svg>
              </div>
              <p className="lib-upload-text">
                Click or drag images here to upload
              </p>
              <p className="lib-upload-hint">
                Supported formats: PNG, JPG, TIFF
              </p>
              <input
                type="file"
                multiple
                accept=".png,.jpg,.jpeg,.tiff"
                className="lib-file-input-hidden"
                ref={fileInputRef}
                onChange={handleFileSelect}
              />
            </div>

            {uploadFiles.length > 0 && (
              <div className="lib-upload-list">
                {uploadFiles.map((file, idx) => (
                  <div key={`${file.name}-${idx}`} className="lib-upload-item">
                    <span className="lib-upload-item-name" title={file.name}>
                      {file.name}
                    </span>
                    <span
                      className={`lib-upload-item-status ${isUploading ? 'uploading' : 'pending'}`}
                    >
                      {isUploading ? 'Uploading...' : 'Ready'}
                    </span>
                  </div>
                ))}
              </div>
            )}

            {uploadError && <p className="lib-form-error">{uploadError}</p>}

            <div className="modal-actions" style={{ marginTop: '1.5rem' }}>
              <button
                type="button"
                className="modal-close modal-close--secondary"
                onClick={() => {
                  setShowUploadModal(false);
                  setUploadFiles([]);
                }}
                disabled={isUploading}
              >
                Cancel
              </button>
              <button
                type="button"
                className="modal-close"
                onClick={() => void handleUpload()}
                disabled={isUploading || uploadFiles.length === 0}
              >
                {isUploading
                  ? 'Uploading...'
                  : `Upload ${uploadFiles.length > 0 ? uploadFiles.length + ' ' : ''}Images`}
              </button>
            </div>
          </div>
        </div>
      )}
      {/* ── Import Dataset Wizard Modal ───────────────────────────────── */}
      {showImportModal && (
        <div
          className="modal-overlay"
          role="dialog"
          aria-modal="true"
          onClick={(e) => {
            if (e.target === e.currentTarget && !importProcessing) {
              resetImportWizard();
            }
          }}
        >
          <div className="modal-card">
            <h2 className="modal-title">Import Dataset Wizard</h2>

            {/* Wizard Steps Indicator */}
            <div className="wizard-steps">
              <div
                className={`wizard-step ${importStep >= 1 ? 'wizard-step--completed' : ''} ${importStep === 1 ? 'wizard-step--active' : ''}`}
              >
                1
              </div>
              <div
                className={`wizard-step ${importStep >= 2 ? 'wizard-step--completed' : ''} ${importStep === 2 ? 'wizard-step--active' : ''}`}
              >
                2
              </div>
              <div
                className={`wizard-step ${importStep === 3 ? 'wizard-step--completed' : ''} ${importStep === 3 ? 'wizard-step--active' : ''}`}
              >
                3
              </div>
            </div>

            {/* STEP 1: UPLOAD ZIP */}
            {importStep === 1 && (
              <div>
                <p className="modal-body" style={{ marginBottom: '1.25rem' }}>
                  Upload a single <strong>.zip</strong> archive containing your
                  raw root scan images. Pre-existing masks matching filenames
                  (e.g. <code>image_mask.png</code>) will be imported as
                  validated annotations.
                </p>
                <div
                  className={`lib-upload-dropzone ${isDragging ? 'lib-upload-dropzone--active' : ''}`}
                  onDragOver={handleZipDragOver}
                  onDragLeave={handleZipDragLeave}
                  onDrop={handleZipDrop}
                  onClick={() => zipInputRef.current?.click()}
                >
                  <div className="lib-upload-icon">
                    <svg
                      width="32"
                      height="32"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="1.5"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    >
                      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                      <polyline points="17 8 12 3 7 8" />
                      <line x1="12" y1="3" x2="12" y2="15" />
                    </svg>
                  </div>
                  <p className="lib-upload-text">
                    Click or drag ZIP archive here
                  </p>
                  <p className="lib-upload-hint">
                    Supported archive: ZIP up to 200MB
                  </p>
                  <input
                    type="file"
                    accept=".zip"
                    className="lib-file-input-hidden"
                    ref={zipInputRef}
                    onChange={handleZipSelect}
                  />
                </div>
                {importError && <p className="lib-form-error">{importError}</p>}

                <div className="modal-actions">
                  <button
                    type="button"
                    className="modal-close modal-close--secondary"
                    onClick={resetImportWizard}
                  >
                    Cancel
                  </button>
                </div>
              </div>
            )}

            {/* STEP 2: CONFIGURATION & TARGET */}
            {importStep === 2 && (
              <div>
                <p className="modal-body" style={{ marginBottom: '1.25rem' }}>
                  Choose where to import the root scans from{' '}
                  <strong>{importZipFile?.name}</strong>:
                </p>

                <div className="target-mode-options">
                  <div
                    className={`target-mode-card ${importTargetMode === 'new' ? 'selected' : ''}`}
                    onClick={() => setImportTargetMode('new')}
                  >
                    <span className="target-card-title">
                      <input
                        type="radio"
                        name="importTargetMode"
                        checked={importTargetMode === 'new'}
                        onChange={() => setImportTargetMode('new')}
                      />
                      Create New
                    </span>
                    <span className="target-card-desc">
                      Extract files into a brand new dataset collection.
                    </span>
                  </div>

                  <div
                    className={`target-mode-card ${importTargetMode === 'existing' ? 'selected' : ''}`}
                    onClick={() => setImportTargetMode('existing')}
                  >
                    <span className="target-card-title">
                      <input
                        type="radio"
                        name="importTargetMode"
                        checked={importTargetMode === 'existing'}
                        onChange={() => setImportTargetMode('existing')}
                      />
                      Add to Existing
                    </span>
                    <span className="target-card-desc">
                      Append scans directly into a previously created dataset.
                    </span>
                  </div>
                </div>

                {importTargetMode === 'new' ? (
                  <div className="lib-create-form">
                    <div className="lib-form-group">
                      <label htmlFor="import-name">Dataset Name *</label>
                      <input
                        type="text"
                        id="import-name"
                        required
                        placeholder="e.g. Arabidopsis Salt Treatment"
                        value={importNewName}
                        onChange={(e) => setImportNewName(e.target.value)}
                      />
                    </div>
                    <div className="lib-form-group">
                      <label htmlFor="import-desc">Description</label>
                      <textarea
                        id="import-desc"
                        rows={2}
                        placeholder="Details about salt concentration, stage, etc."
                        value={importNewDesc}
                        onChange={(e) => setImportNewDesc(e.target.value)}
                      />
                    </div>
                    <div className="lib-form-row">
                      <div className="lib-form-group">
                        <label htmlFor="import-species">Species</label>
                        <input
                          type="text"
                          id="import-species"
                          placeholder="e.g. Arabidopsis thaliana"
                          value={importNewSpecies}
                          onChange={(e) => setImportNewSpecies(e.target.value)}
                        />
                      </div>
                      <div className="lib-form-group">
                        <label htmlFor="import-growth">Growth Stage</label>
                        <select
                          id="import-growth"
                          value={importNewGrowth}
                          onChange={(e) => setImportNewGrowth(e.target.value)}
                        >
                          <option value="">(None)</option>
                          <option value="seedling">Seedling</option>
                          <option value="mature">Mature</option>
                        </select>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="lib-create-form">
                    <div className="lib-form-group">
                      <label htmlFor="import-target-ds">
                        Select Destination Dataset *
                      </label>
                      <select
                        id="import-target-ds"
                        required
                        value={importDatasetId}
                        onChange={(e) => setImportDatasetId(e.target.value)}
                      >
                        <option value="">-- Choose a Dataset --</option>
                        {datasets.map((ds) => (
                          <option key={ds.dataset_id} value={ds.dataset_id}>
                            {ds.name} ({ds.count} scans)
                          </option>
                        ))}
                      </select>
                    </div>
                  </div>
                )}

                {importError && <p className="lib-form-error">{importError}</p>}

                <div className="modal-actions" style={{ marginTop: '1.5rem' }}>
                  <button
                    type="button"
                    className="modal-close modal-close--secondary"
                    onClick={() => setImportStep(1)}
                  >
                    Back
                  </button>
                  <button
                    type="button"
                    className="modal-close"
                    onClick={() => void handleImportSubmit()}
                    disabled={
                      importTargetMode === 'new'
                        ? !importNewName.trim()
                        : !importDatasetId
                    }
                  >
                    Start Import
                  </button>
                </div>
              </div>
            )}

            {/* STEP 3: PROGRESS & LOGS / REPORT */}
            {importStep === 3 && (
              <div>
                {importProcessing ? (
                  <div className="import-progress-container">
                    <div className="import-progress-text">
                      <span>Importing & Processing scans...</span>
                      <span>{importProgress}%</span>
                    </div>
                    <div className="import-progress-bar-bg">
                      <div
                        className="import-progress-bar-fill"
                        style={{ width: `${importProgress}%` }}
                      ></div>
                    </div>
                    <p
                      className="modal-body"
                      style={{
                        fontSize: '0.8rem',
                        color: 'var(--color-text-muted)',
                        marginTop: '0.5rem',
                      }}
                    >
                      Extracting files, executing U-Net models for
                      segmentations, and writing metadata records. Please do not
                      close this modal.
                    </p>
                  </div>
                ) : (
                  <div>
                    <div className="import-success-summary">
                      <div className="import-success-number">
                        {importReport?.scans_imported}
                      </div>
                      <div className="import-success-label">
                        Scans Successfully Imported
                      </div>
                      <p
                        className="modal-body"
                        style={{ marginTop: '0.5rem', fontSize: '0.82rem' }}
                      >
                        All root scans are loaded into the dataset{' '}
                        <strong>{importReport?.dataset_name}</strong>.
                      </p>
                    </div>

                    {importReport && importReport.warnings.length > 0 && (
                      <div className="import-report-box">
                        <div className="import-report-title">
                          Import Warnings ({importReport.warnings.length})
                        </div>
                        <ul className="import-report-list">
                          {importReport.warnings.map((warn, idx) => (
                            <li key={idx} className="import-report-item">
                              <span>⚠️</span> {warn}
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}

                    <div
                      className="modal-actions"
                      style={{ marginTop: '1.5rem' }}
                    >
                      <button
                        type="button"
                        className="modal-close"
                        onClick={resetImportWizard}
                      >
                        Finish
                      </button>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      )}

      {/* ── Annotation Detail Modal ─────────────────────────────────── */}
      {selectedAnnotation && (
        <div
          className="modal-overlay"
          role="dialog"
          aria-modal="true"
          onClick={(e) => {
            if (e.target === e.currentTarget) {
              annotationModalClosedRef.current =
                selectedAnnotation?.annotation_id ?? null;
              setSelectedAnnotation(null);
            }
          }}
        >
          <div className="ann-detail-modal">
            {/* ── Header ── */}
            <div className="ann-detail-header">
              <div>
                <span className="lib-breadcrumb">ANNOTATION</span>
                <h2 className="ann-detail-title">
                  {selectedAnnotation.annotation_id.slice(0, 8).toUpperCase()}
                  <span
                    className={`ann-status-pill ann-status-pill--${selectedAnnotation.status}`}
                  >
                    {selectedAnnotation.status.replace('_', ' ')}
                  </span>
                </h2>
              </div>
              <button
                className="ann-detail-close"
                onClick={() => {
                  annotationModalClosedRef.current =
                    selectedAnnotation.annotation_id;
                  setSelectedAnnotation(null);
                }}
              >
                <svg
                  width="16"
                  height="16"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  <line x1="18" y1="6" x2="6" y2="18" />
                  <line x1="6" y1="6" x2="18" y2="18" />
                </svg>
              </button>
            </div>

            <div className="ann-detail-body">
              {/* ── Left: Image viewer ── */}
              <div className="ann-detail-image-col">
                {/* Tab switcher */}
                <div className="ann-image-tabs">
                  <button
                    className={`ann-image-tab ${imageView === 'scan' ? 'ann-image-tab--active' : ''}`}
                    onClick={() => setImageView('scan')}
                  >
                    Original Scan
                  </button>
                  <button
                    className={`ann-image-tab ${imageView === 'mask' ? 'ann-image-tab--active' : ''}`}
                    onClick={() => setImageView('mask')}
                    disabled={!selectedAnnotation.mask_url}
                  >
                    Seg. Mask
                  </button>
                  <button
                    className={`ann-image-tab ${imageView === 'tip' ? 'ann-image-tab--active' : ''}`}
                    onClick={() => setImageView('tip')}
                    disabled={!selectedAnnotation.tip_photo_url}
                  >
                    Tip Image
                  </button>
                </div>

                {/* Image display */}
                <div className="ann-image-frame">
                  {imageView === 'scan' ? (
                    <img
                      src={getArtifactUrl(selectedAnnotation.image_uri)}
                      alt="Original Scan"
                      className="ann-image"
                      onClick={() =>
                        setLightboxSrc(
                          getArtifactUrl(selectedAnnotation.image_uri),
                        )
                      }
                      title="Click to enlarge"
                      onError={(e) => {
                        (e.target as HTMLImageElement).src =
                          'data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="200" height="200" viewBox="0 0 200 200"><rect width="200" height="200" fill="%23222"/><text x="100" y="105" font-family="sans-serif" font-size="14" fill="%23666" text-anchor="middle">No Preview</text></svg>';
                      }}
                    />
                  ) : imageView === 'mask' && selectedAnnotation.mask_url ? (
                    <div
                      style={{
                        position: 'relative',
                        width: '100%',
                        height: '100%',
                      }}
                    >
                      <img
                        src={
                          selectedAnnotation.mask_url
                            ? getArtifactUrl(selectedAnnotation.mask_url)
                            : getArtifactUrl(
                                linkedPrediction?.segmentation_mask_url,
                              )
                        }
                        alt="Segmentation Mask"
                        className="ann-image"
                        onClick={() =>
                          setLightboxSrc(
                            getArtifactUrl(selectedAnnotation.mask_url!),
                          )
                        }
                        title="Click to enlarge"
                        onError={(e) => {
                          (e.target as HTMLImageElement).src =
                            'data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="200" height="200" viewBox="0 0 200 200"><rect width="200" height="200" fill="%23222"/><text x="100" y="105" font-family="sans-serif" font-size="14" fill="%23666" text-anchor="middle">No Mask</text></svg>';
                        }}
                      />
                      <button
                        className="ann-edit-mask-btn"
                        title="Edit mask in Lab"
                        onClick={(e) => {
                          e.stopPropagation();
                          setSelectedAnnotation(null);
                          navigate(
                            `/lab?annotation_id=${selectedAnnotation.annotation_id}`,
                          );
                        }}
                      >
                        <svg
                          width="13"
                          height="13"
                          viewBox="0 0 24 24"
                          fill="none"
                          stroke="currentColor"
                          strokeWidth="2.5"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                        >
                          <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7" />
                          <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z" />
                        </svg>
                        Edit in Lab
                      </button>
                    </div>
                  ) : imageView === 'tip' &&
                    selectedAnnotation.tip_photo_url ? (
                    <img
                      src={selectedAnnotation.tip_photo_url}
                      alt="Tip Visualisation"
                      className="ann-image"
                      onClick={() =>
                        setLightboxSrc(selectedAnnotation.tip_photo_url!)
                      }
                      onError={(e) => {
                        (e.target as HTMLImageElement).src =
                          'data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="200" height="200" viewBox="0 0 200 200"><rect width="200" height="200" fill="%23222"/><text x="100" y="105" font-family="sans-serif" font-size="14" fill="%23666" text-anchor="middle">No Mask</text></svg>';
                      }}
                    />
                  ) : (
                    <div className="ann-image-empty">
                      <svg
                        width="32"
                        height="32"
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="var(--color-text-muted)"
                        strokeWidth="1.5"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                      >
                        <rect x="3" y="3" width="18" height="18" rx="2" />
                        <circle cx="8.5" cy="8.5" r="1.5" />
                        <polyline points="21 15 16 10 5 21" />
                      </svg>
                      <p>No mask generated yet.</p>
                      <span>
                        Run inference to generate a segmentation mask.
                      </span>
                    </div>
                  )}
                </div>

                {/* Run / Re-run inference button */}
                <button
                  className="lib-btn-run-inference"
                  disabled={runningAnnotationIds.has(
                    selectedAnnotation.annotation_id,
                  )}
                  onClick={() => void handleRunAnnotationInference()}
                >
                  {runningAnnotationIds.has(
                    selectedAnnotation.annotation_id,
                  ) ? (
                    <>
                      <span
                        className="spinner"
                        style={{ width: '12px', height: '12px' }}
                      />
                      Running...
                    </>
                  ) : selectedAnnotation.status === 'pending' ? (
                    <>
                      <svg
                        width="13"
                        height="13"
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="2.5"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                      >
                        <polygon points="5 3 19 12 5 21 5 3" />
                      </svg>
                      Run Inference
                    </>
                  ) : (
                    '↺ Re-run Prediction'
                  )}
                </button>
              </div>

              {/* ── Right: Info ── */}
              <div className="ann-detail-info-col">
                {/* Metadata */}
                <div className="ann-detail-section">
                  <h3 className="ann-detail-section-title">METADATA</h3>
                  <dl className="ann-meta-list">
                    {[
                      {
                        label: 'Species',
                        field: 'species',
                        value: selectedAnnotation.species || '—',
                        editable: true,
                      },
                      {
                        label: 'Age (days)',
                        field: 'age_days',
                        value: selectedAnnotation.age_days?.toString() || '—',
                        editable: true,
                      },
                      {
                        label: 'Treatment',
                        field: 'treatment',
                        value: selectedAnnotation.treatment || '—',
                        editable: true,
                      },
                      {
                        label: 'Owner',
                        field: 'owner_email',
                        value: selectedAnnotation.owner_email || '—',
                        editable: false,
                      },
                      {
                        label: 'Created',
                        field: null,
                        value: new Date(
                          selectedAnnotation.created_at,
                        ).toLocaleDateString('en-GB', {
                          day: '2-digit',
                          month: 'short',
                          year: 'numeric',
                        }),
                        editable: false,
                      },
                      {
                        label: 'Updated',
                        field: null,
                        value: new Date(
                          selectedAnnotation.updated_at,
                        ).toLocaleDateString('en-GB', {
                          day: '2-digit',
                          month: 'short',
                          year: 'numeric',
                        }),
                        editable: false,
                      },
                    ].map(({ label, field, value, editable }) => (
                      <div key={label} className="ann-meta-row">
                        <dt className="ann-meta-label">{label}</dt>
                        <dd className="ann-meta-value ann-meta-value--editable">
                          {editable && field && editingMetaField === field ? (
                            <input
                              className="ann-meta-input"
                              value={editingMetaValue}
                              autoFocus
                              disabled={metaSaving}
                              onChange={(e) =>
                                setEditingMetaValue(e.target.value)
                              }
                              onBlur={() => void handleMetaSave(field)}
                              onKeyDown={(e) => {
                                if (e.key === 'Enter')
                                  void handleMetaSave(field);
                                if (e.key === 'Escape')
                                  setEditingMetaField(null);
                              }}
                            />
                          ) : (
                            <>
                              <span>{value}</span>
                              {editable && field && (
                                <button
                                  className="ann-meta-edit-btn"
                                  onClick={() => {
                                    setEditingMetaField(field);
                                    setEditingMetaValue(
                                      value === '—' ? '' : value,
                                    );
                                  }}
                                >
                                  ✎
                                </button>
                              )}
                            </>
                          )}
                        </dd>
                      </div>
                    ))}
                  </dl>
                </div>

                {/* ── Dataset assignment ── */}
                {!selectedAnnotation.dataset_id ? (
                  <div className="ann-detail-section">
                    <h3 className="ann-detail-section-title">DATASET</h3>
                    <p className="ann-assign-hint">
                      This annotation is not linked to any dataset.
                    </p>
                    <div className="ann-assign-row">
                      <select
                        className="ann-assign-select"
                        value={targetDatasetId}
                        onChange={(e) => {
                          setTargetDatasetId(e.target.value);
                          setAddToDatasetError(null);
                        }}
                        disabled={addingToDataset}
                      >
                        <option value="">Select a dataset...</option>
                        {datasets.map((d) => (
                          <option key={d.dataset_id} value={d.dataset_id}>
                            {d.name} ({d.count} scans)
                          </option>
                        ))}
                      </select>
                      <button
                        className="ann-assign-btn"
                        disabled={!targetDatasetId || addingToDataset}
                        onClick={async () => {
                          setAddingToDataset(true);
                          setAddToDatasetError(null);
                          try {
                            const updated = await addAnnotationToDataset(
                              targetDatasetId,
                              selectedAnnotation.annotation_id,
                            );
                            setSelectedAnnotation(updated);
                            setAnnotRefreshKey((k) => k + 1);
                            // Also refresh dataset list so count updates
                            const data = await getDatasets({
                              species: speciesFilter || undefined,
                              growth_stage:
                                growthStages.length === 1
                                  ? growthStages[0]
                                  : undefined,
                            });
                            setDatasets(data);
                          } catch (err) {
                            setAddToDatasetError(
                              (err as Error).message ||
                                'Failed to add to dataset.',
                            );
                          } finally {
                            setAddingToDataset(false);
                          }
                        }}
                      >
                        {addingToDataset ? (
                          <>
                            <span
                              className="spinner"
                              style={{ width: '10px', height: '10px' }}
                            />{' '}
                            Adding...
                          </>
                        ) : (
                          'Add to Dataset'
                        )}
                      </button>
                    </div>
                    {addToDatasetError && (
                      <p className="ann-assign-error">{addToDatasetError}</p>
                    )}
                  </div>
                ) : (
                  <div className="ann-detail-section">
                    <h3 className="ann-detail-section-title">DATASET</h3>
                    <div className="ann-assign-linked">
                      <svg
                        width="12"
                        height="12"
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="2.5"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                      >
                        <ellipse cx="12" cy="5" rx="9" ry="3" />
                        <path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3" />
                        <path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5" />
                      </svg>
                      <span
                        className="ann-assign-linked-name"
                        onClick={() => {
                          setSelectedAnnotation(null);
                          setSelectedDatasetId(selectedAnnotation.dataset_id!);
                          setActiveTab('datasets');
                        }}
                        title="Go to dataset"
                      >
                        {datasets.find(
                          (d) => d.dataset_id === selectedAnnotation.dataset_id,
                        )?.name ??
                          selectedAnnotation.dataset_id?.slice(0, 8) + '...'}
                      </span>
                    </div>
                  </div>
                )}

                {/* U-Net Detection — admin only */}
                {role === 'admin' && (
                  <div className="ann-detail-section">
                    <h3 className="ann-detail-section-title">
                      U-NET DETECTION
                    </h3>
                    {linkedPredictionLoading ? (
                      <div className="ann-prediction-loading">
                        <span
                          className="spinner"
                          style={{ width: '14px', height: '14px' }}
                        />
                        Loading prediction...
                      </div>
                    ) : linkedPrediction ? (
                      <div className="ann-prediction-data">
                        <div className="ann-prediction-meta">
                          <div>
                            <span className="ann-meta-label">MODEL</span>
                            <span
                              className="ann-meta-value"
                              style={{ fontSize: 'var(--text-xs)' }}
                            >
                              {linkedPrediction.metadata.model_version}
                            </span>
                          </div>
                          <div>
                            <span className="ann-meta-label">LATENCY</span>
                            <span className="ann-meta-value">
                              {(
                                linkedPrediction.metadata.processing_time_ms /
                                1000
                              ).toFixed(3)}
                              s
                            </span>
                          </div>
                        </div>
                        {linkedPrediction.plants.length > 0 ? (
                          <table className="ann-plants-table">
                            <thead>
                              <tr>
                                <th>#</th>
                                <th>Length (mm)</th>
                                <th>Tip</th>
                              </tr>
                            </thead>
                            <tbody>
                              {linkedPrediction.plants.map((p) => (
                                <tr key={p.plant_id}>
                                  <td>{p.plant_id}</td>
                                  <td
                                    style={{
                                      color: 'var(--color-tag-text)',
                                      fontWeight: 600,
                                    }}
                                  >
                                    {p.root_length_mm}mm
                                  </td>
                                  <td
                                    style={{
                                      fontFamily: 'var(--font-mono)',
                                      fontSize: '10px',
                                      color: 'var(--color-text-muted)',
                                    }}
                                  >
                                    ({p.tip_x}, {p.tip_y})
                                  </td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        ) : (
                          <p className="ann-prediction-empty">
                            No plants detected.
                          </p>
                        )}
                      </div>
                    ) : (
                      <p className="ann-prediction-empty">
                        {selectedAnnotation.prediction_id
                          ? 'Prediction data unavailable.'
                          : 'No linked prediction.'}
                      </p>
                    )}
                  </div>
                )}

                {selectedAnnotation.prediction_id && (
                  <div className="ann-detail-section">
                    <h3 className="ann-detail-section-title">FEEDBACK</h3>
                    <div className="rv-feedback-btns">
                      <button
                        className={`rv-fb-btn rv-fb-btn--incorrect ${annotationFeedback === 'incorrect' ? 'rv-fb-btn--selected' : ''}`}
                        disabled={feedbackSaving}
                        onClick={async () => {
                          const next: 'correct' | 'incorrect' = 'incorrect';
                          if (next === annotationFeedback) return;
                          setFeedbackSaving(true);
                          try {
                            await submitAnnotationFeedback(
                              selectedAnnotation.prediction_id!,
                              next,
                            );
                            setAnnotationFeedback(next);
                            setSelectedAnnotation((prev) =>
                              prev
                                ? { ...prev, status: 'review_required' }
                                : null,
                            );
                            setAnnotRefreshKey((k) => k + 1); // triggers list re-fetch
                            setDetailScans((prev) =>
                              prev.map((s) =>
                                s.annotation_id ===
                                selectedAnnotation.annotation_id
                                  ? { ...s, status: 'review_required' }
                                  : s,
                              ),
                            );
                          } finally {
                            setFeedbackSaving(false);
                          }
                        }}
                      >
                        Incorrect
                      </button>
                      <button
                        className={`rv-fb-btn rv-fb-btn--correct ${annotationFeedback === 'correct' ? 'rv-fb-btn--selected' : ''}`}
                        disabled={feedbackSaving}
                        onClick={async () => {
                          const next: 'correct' | 'incorrect' = 'correct';
                          if (next === annotationFeedback) return;
                          setFeedbackSaving(true);
                          try {
                            await submitAnnotationFeedback(
                              selectedAnnotation.prediction_id!,
                              next,
                            );
                            setAnnotationFeedback(next);
                            setSelectedAnnotation((prev) =>
                              prev ? { ...prev, status: 'annotated' } : null,
                            );
                            setAnnotRefreshKey((k) => k + 1); // triggers list re-fetch
                            setDetailScans((prev) =>
                              prev.map((s) =>
                                s.annotation_id ===
                                selectedAnnotation.annotation_id
                                  ? { ...s, status: 'annotated' }
                                  : s,
                              ),
                            );
                          } finally {
                            setFeedbackSaving(false);
                          }
                        }}
                      >
                        Correct
                      </button>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── Lightbox ── */}
      {lightboxSrc && (
        <div className="ann-lightbox" onClick={() => setLightboxSrc(null)}>
          <button
            className="ann-lightbox-close"
            onClick={() => setLightboxSrc(null)}
          >
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
              <line x1="18" y1="6" x2="6" y2="18" />
              <line x1="6" y1="6" x2="18" y2="18" />
            </svg>
          </button>
          <img
            src={lightboxSrc}
            alt="Enlarged view"
            className="ann-lightbox-img"
          />
        </div>
      )}
    </div>
  );
}

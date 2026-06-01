/**
 * pages/Lab.tsx
 *
 * Annotation review and mask correction workspace.
 *
 * Entry modes (via URL query params):
 *
 *   /lab
 *     Empty state — shows a search/filter picker.
 *     User selects a dataset or a single annotation to start working.
 *
 *   /lab?annotation_id=<uuid>
 *     Single-annotation mode. Loads that annotation directly into the
 *     editor. After submitting, returns to the empty state.
 *
 *   /lab?dataset_id=<uuid>
 *     Dataset queue mode. Loads all review_required annotations for
 *     the dataset, presents them one by one. After each submission the
 *     editor auto-advances. When the queue is exhausted, shows a
 *     "all done" screen.
 *
 * The URL is kept in sync with the current annotation so that
 * refreshing or sharing a link restores the correct state.
 *
 * Route: /lab (protected)
 */

import { useEffect, useState, useCallback } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import NavBar from '../components/NavBar';
import { getLabQueue, submitCorrectedMask } from '../api/labApi';
import { getAnnotationDetail } from '../api/libraryApi';
import { getArtifactUrl } from '../api/client';
import type { AnnotationResponse } from '../types/api';
import './Lab.css';
import MaskEditor from '../components/MaskEditor';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function metaRow(key: string, value: string | number | null | undefined) {
  if (value == null || value === '') return null;
  return (
    <div className="lab-meta-row" key={key}>
      <span className="lab-meta-key">{key}</span>
      <span className="lab-meta-value">{String(value)}</span>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export default function Lab() {
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();

  const paramAnnotationId = searchParams.get('annotation_id');
  const paramDatasetId = searchParams.get('dataset_id');

  //  Queue state
  const [queue, setQueue] = useState<AnnotationResponse[]>([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [loading, setLoading] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);

  //  Save state
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);

  //  Derive the active annotation from queue + index
  const currentAnnotation: AnnotationResponse | null =
    queue[currentIndex] ?? null;

  //  Load queue whenever URL params change
  const loadQueue = useCallback(async () => {
    setLoading(true);
    setLoadError(null);
    setSaveError(null);

    try {
      if (paramDatasetId) {
        // Dataset mode — fetch the full queue once.
        const anns = await getLabQueue({ dataset_id: paramDatasetId });
        setQueue(anns);
        if (paramAnnotationId) {
          const idx = anns.findIndex(
            (a) => a.annotation_id === paramAnnotationId,
          );
          setCurrentIndex(idx >= 0 ? idx : 0);
        } else {
          setCurrentIndex(0);
        }
      } else if (paramAnnotationId) {
        // Single-annotation mode — no dataset, just one annotation.
        const ann = await getAnnotationDetail(paramAnnotationId);
        setQueue([ann]);
        setCurrentIndex(0);
      } else {
        setQueue([]);
        setCurrentIndex(0);
      }
    } catch (err) {
      setLoadError((err as Error).message || 'Failed to load annotation.');
    } finally {
      setLoading(false);
    }
  }, [paramDatasetId, paramAnnotationId]);

  useEffect(() => {
    Promise.resolve().then(() => {
      void loadQueue();
    });
  }, [loadQueue]);

  //  Handle mask submission
  async function handleSave(blob: Blob) {
    if (!currentAnnotation) return;
    setSaving(true);
    setSaveError(null);

    try {
      await submitCorrectedMask(currentAnnotation.annotation_id, blob);

      if (paramDatasetId) {
        // Queue mode: advance to the next annotation
        const nextIndex = currentIndex + 1;
        if (nextIndex < queue.length) {
          setCurrentIndex(nextIndex);
          // Keep dataset_id in URL, update annotation_id for shareability
          setSearchParams({
            dataset_id: paramDatasetId,
            annotation_id: queue[nextIndex].annotation_id,
          });
        } else {
          // Queue exhausted — show done state
          setSearchParams({ dataset_id: paramDatasetId, done: '1' });
        }
      } else {
        // Single-annotation mode: clear params after submission
        setSearchParams({});
      }
    } catch (err) {
      setSaveError(
        (err as Error).message || 'Failed to save mask. Please try again.',
      );
    } finally {
      setSaving(false);
    }
  }

  //  Handle skip
  function handleSkip() {
    setSaveError(null);

    if (paramDatasetId) {
      const nextIndex = currentIndex + 1;
      if (nextIndex < queue.length) {
        setCurrentIndex(nextIndex);
        setSearchParams({
          dataset_id: paramDatasetId,
          annotation_id: queue[nextIndex].annotation_id,
        });
      } else {
        setSearchParams({ dataset_id: paramDatasetId, done: '1' });
      }
      return;
    }

    // Single-annotation mode
    setSearchParams({});
  }

  //  "All done" state
  if (searchParams.get('done') === '1') {
    return (
      <div className="lab-state-page">
        <NavBar />
        <div className="lab-done-wrap">
          <div className="lab-done-icon-ring">
            <svg
              width="28"
              height="28"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.8"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
              <polyline points="22 4 12 14.01 9 11.01" />
            </svg>
          </div>

          <h1 className="lab-done-title">Queue complete</h1>
          <p className="lab-done-sub">
            All review-required annotations in this dataset have been processed.
          </p>

          <div className="lab-done-stats">
            <div className="lab-done-stat">
              <span className="lab-done-stat-num">{queue.length}</span>
              <span className="lab-done-stat-label">masks corrected</span>
            </div>
            <div className="lab-done-stat">
              <span className="lab-done-stat-num">0</span>
              <span className="lab-done-stat-label">remaining</span>
            </div>
          </div>

          <div className="lab-done-divider" />

          <div className="lab-done-actions">
            <button
              className="lab-done-btn lab-done-btn--secondary"
              onClick={() => navigate('/library')}
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
              Back to library
            </button>
            <button
              className="lab-done-btn lab-done-btn--primary"
              onClick={() => setSearchParams({})}
            >
              Start new review
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
        </div>
      </div>
    );
  }

  //  Loading state
  if (loading) {
    return (
      <div className="lab-state-page">
        <NavBar />
        <div className="lab-state-body">
          <span className="spinner" aria-hidden="true" />
          <p className="lab-state-sub">Loading annotation…</p>
        </div>
      </div>
    );
  }

  //  Load error state
  if (loadError) {
    return (
      <div className="lab-state-page">
        <NavBar />
        <div className="lab-state-body">
          <div className="lab-state-icon">⚠️</div>
          <h1 className="lab-state-title">Failed to Load</h1>
          <p className="lab-state-sub">{loadError}</p>
          <div className="lab-state-actions">
            <button
              className="lab-btn lab-btn--secondary"
              onClick={() => navigate('/library')}
            >
              Back to Library
            </button>
            <button
              className="lab-btn lab-btn--primary"
              onClick={() => void loadQueue()}
            >
              Retry
            </button>
          </div>
        </div>
      </div>
    );
  }

  //  Empty / browse state (no params in URL)
  if (!paramAnnotationId && !paramDatasetId) {
    return (
      <div className="lab-state-page">
        <NavBar />
        <main className="lab-home">
          <div className="lab-hero">
            <div className="lab-hero-left">
              <p className="lab-eyebrow">Annotation workspace</p>
              <h1 className="lab-home-title">
                Annotation
                <br />
                Lab
              </h1>
              <p className="lab-home-desc">
                Review and correct segmentation masks flagged by the model.
                Corrected masks become ground truth for the next retraining
                cycle.
              </p>
              <button
                className="lab-home-cta"
                onClick={() => navigate('/library')}
              >
                <svg
                  width="15"
                  height="15"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  <path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z" />
                  <path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z" />
                </svg>
                Open Library
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
                  <polyline points="9 18 15 12 9 6" />
                </svg>
              </button>
            </div>

            <div className="lab-flow-diagram">
              <p className="lab-flow-label">Review workflow</p>
              <div className="lab-flow-row">
                {[
                  { icon: 'database', label: 'Dataset', active: false },
                  { icon: 'inference', label: 'Inference', active: false },
                  { icon: 'edit', label: 'Lab review', active: true },
                  { icon: 'retrain', label: 'Retraining', active: false },
                ].map((step, i, arr) => (
                  <div key={step.label} className="lab-flow-step-group">
                    <div
                      className={`lab-flow-step${step.active ? ' lab-flow-step--active' : ''}`}
                    >
                      <div className="lab-flow-icon">
                        {step.icon === 'database' && (
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
                        )}
                        {step.icon === 'inference' && (
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
                            <rect x="2" y="2" width="20" height="20" rx="2" />
                            <path d="M7 12h10M12 7v10" />
                          </svg>
                        )}
                        {step.icon === 'edit' && (
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
                            <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7" />
                            <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z" />
                          </svg>
                        )}
                        {step.icon === 'retrain' && (
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
                            <polyline points="23 4 23 10 17 10" />
                            <path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10" />
                          </svg>
                        )}
                      </div>
                      <span className="lab-flow-name">{step.label}</span>
                    </div>
                    {i < arr.length - 1 && (
                      <svg
                        className="lab-flow-arrow"
                        width="12"
                        height="12"
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="2.5"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                      >
                        <polyline points="9 18 15 12 9 6" />
                      </svg>
                    )}
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div className="lab-info-cards">
            <div className="lab-info-card">
              <svg
                className="lab-info-icon"
                width="18"
                height="18"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.8"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <path d="M9 11l3 3L22 4" />
                <path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11" />
              </svg>
              <p className="lab-info-title">Queue mode</p>
              <p className="lab-info-body">
                Open a dataset from the Library to work through all its flagged
                annotations in sequence.
              </p>
            </div>
            <div className="lab-info-card">
              <svg
                className="lab-info-icon"
                width="18"
                height="18"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.8"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <circle cx="12" cy="12" r="3" />
                <path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7z" />
              </svg>
              <p className="lab-info-title">Single annotation</p>
              <p className="lab-info-body">
                Jump to a specific annotation using the Edit Mask button on its
                detail view.
              </p>
            </div>
            <div className="lab-info-card">
              <svg
                className="lab-info-icon"
                width="18"
                height="18"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.8"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z" />
                <polyline points="17 21 17 13 7 13 7 21" />
                <polyline points="7 3 7 8 15 8" />
              </svg>
              <p className="lab-info-title">Ground truth</p>
              <p className="lab-info-body">
                Every saved correction overwrites the model mask and is queued
                for the next training run.
              </p>
            </div>
          </div>

          <div className="lab-hint-bar">
            <svg
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
              <circle cx="12" cy="12" r="10" />
              <line x1="12" y1="8" x2="12" y2="12" />
              <line x1="12" y1="16" x2="12.01" y2="16" />
            </svg>
            <span>
              Go to <strong>Library</strong>, open a dataset with{' '}
              <strong>review required</strong> annotations, and click{' '}
              <strong>Review in Lab</strong> — or open any annotation and click{' '}
              <strong>Edit Mask</strong>.
            </span>
          </div>
        </main>
      </div>
    );
  }

  //  Queue empty (dataset had no review_required annotations)
  if (!currentAnnotation) {
    return (
      <div className="lab-state-page">
        <NavBar />
        <div className="lab-state-body">
          <div className="lab-state-icon">🎉</div>
          <h1 className="lab-state-title">Nothing to Review</h1>
          <p className="lab-state-sub">
            There are no annotations marked <strong>review required</strong> for
            this dataset.
          </p>
          <div className="lab-state-actions">
            <button
              className="lab-btn lab-btn--primary"
              onClick={() => navigate('/library')}
            >
              Back to Library
            </button>
          </div>
        </div>
      </div>
    );
  }

  //  Editor state
  const totalInQueue = queue.length;
  const positionInQueue = currentIndex + 1;
  const progressPercent =
    totalInQueue <= 1
      ? 100
      : Math.round((currentIndex / (totalInQueue - 1)) * 100);

  const imageUrl = getArtifactUrl(currentAnnotation.image_uri) ?? '';
  const maskUrl = getArtifactUrl(currentAnnotation.mask_url) ?? '';

  return (
    <div className="lab-page">
      <NavBar />

      <main className="lab-main">
        {/*  Sidebar  */}
        <aside className="lab-sidebar">
          {/* Queue position */}
          {paramDatasetId && (
            <div className="lab-sidebar-section">
              <span className="lab-sidebar-label">Review Queue</span>
              <div className="lab-queue-stat">
                {positionInQueue}
                <span
                  style={{
                    fontSize: '1rem',
                    fontWeight: 400,
                    color: 'var(--color-text-muted)',
                  }}
                >
                  {' '}
                  / {totalInQueue}
                </span>
              </div>
              <div className="lab-queue-sub">
                {totalInQueue - positionInQueue === 0
                  ? 'Last annotation'
                  : `${totalInQueue - positionInQueue} remaining`}
              </div>
              <div className="lab-progress-bar">
                <div
                  className="lab-progress-fill"
                  style={{ width: `${progressPercent}%` }}
                />
              </div>
            </div>
          )}

          {/* Annotation metadata */}
          <div className="lab-sidebar-section">
            <span className="lab-sidebar-label">Annotation</span>
            <div className="lab-meta-card">
              {metaRow('ID', currentAnnotation.annotation_id.slice(0, 8) + '…')}
              {metaRow('Species', currentAnnotation.species)}
              {metaRow('Age (days)', currentAnnotation.age_days)}
              {metaRow('Treatment', currentAnnotation.treatment)}
              {metaRow('Growth stage', currentAnnotation.growth_stage)}
              {metaRow('Owner', currentAnnotation.owner_email)}
            </div>
          </div>

          {/* Save error */}
          {saveError && <div className="lab-error-banner">{saveError}</div>}
        </aside>

        {/*  Editor  */}
        <div className="lab-editor-area" style={{ position: 'relative' }}>
          {saving && (
            <div className="lab-saving-overlay">
              <span className="spinner" aria-hidden="true" />
              Saving mask…
            </div>
          )}

          <MaskEditor
            inline
            isOpen
            originalImageUrl={imageUrl}
            maskImageUrl={maskUrl}
            title={
              paramDatasetId && totalInQueue > 1
                ? `Annotation ${positionInQueue} of ${totalInQueue}`
                : 'Edit Segmentation Mask'
            }
            subtitle="Draw to add roots · Erase to remove noise"
            saveLabel="Save & Continue →"
            onSave={(blob) => void handleSave(blob)}
            onClose={handleSkip}
          />
        </div>
      </main>
    </div>
  );
}

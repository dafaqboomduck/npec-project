import { useState } from 'react';
import { getArtifactUrl } from '../api/client';
import MaskEditor from './MaskEditor';
import type { PredictionResponse, Plant } from '../types/api';
import './ResultsView.css';
import { confirmPrediction } from '../api/inferenceApi';
import { submitCorrectedMask } from '../api/labApi';

export type OverlayMode = 'scan' | 'mask' | 'tips';

interface ResultsViewProps {
  result: PredictionResponse;
  file?: File | null; // Local file from upload session
  filePreviewUrl: string | null;
  deployedModelName: string;
  onNewScan?: () => void;
  onDiscard?: () => void;
  onSave?: () => void;
  onFeedbackChange?: (signal: 'correct' | 'incorrect' | null) => void;
  isHistoryView?: boolean;
}

export default function ResultsView({
  result,
  file,
  filePreviewUrl,
  deployedModelName,
  onNewScan,
  onDiscard,
  onSave,
  onFeedbackChange,
  isHistoryView = false,
}: ResultsViewProps) {
  const [overlayMode, setOverlayMode] = useState<OverlayMode>('scan');
  const [feedback, setFeedback] = useState<'correct' | 'incorrect' | null>(
    null,
  );
  const [isEditorOpen, setIsEditorOpen] = useState(false);
  const [maskSaving, setMaskSaving] = useState(false);
  const [maskSaveError, setMaskSaveError] = useState<string | null>(null);
  const meta = result.metadata;

  const overlayModes: OverlayMode[] = ['scan', 'mask', 'tips'];

  // Prefer backend-served URLs, fall back to local object URL from upload
  const inputDisplayUrl =
    getArtifactUrl(result.input_url) || (filePreviewUrl ?? '');

  const displayImageUrl =
    overlayMode === 'mask' && result.segmentation_mask_url
      ? getArtifactUrl(result.segmentation_mask_url)
      : overlayMode === 'tips' && result.tip_photo_url
        ? getArtifactUrl(result.tip_photo_url)
        : inputDisplayUrl;

  return (
    <div className="results-view">
      {/* ── Breadcrumb + heading ─────────────────────────────── */}
      <div className="rv-header">
        <div className="rv-header-left">
          <span className="rv-breadcrumb">
            {isHistoryView ? 'HISTORY / DETAIL' : 'INFERENCE / RESULTS'}
          </span>
          <div className="rv-title-row">
            <h2 className="rv-title">
              {isHistoryView ? 'Past Prediction' : 'Inference complete'}
            </h2>
            <span className="rv-plant-badge">
              {result.plants.length} PLANTS
            </span>
          </div>
          <p className="rv-subtitle">
            {result.input_filename || file?.name || 'scan'} ·{' '}
            {deployedModelName}
          </p>
        </div>
        {!isHistoryView && (
          <div className="rv-header-actions">
            <button
              className="rv-btn rv-btn--outline"
              id="btn-new-scan"
              onClick={onNewScan}
            >
              + New scan
            </button>
            <button
              className="rv-btn rv-btn--danger"
              id="btn-discard-result"
              onClick={onDiscard}
            >
              Discard
            </button>
            <button
              className="rv-btn rv-btn--primary"
              id="btn-save-result"
              onClick={onSave}
            >
              ✓ Save to library
            </button>
          </div>
        )}
      </div>

      {/* ── Stats row ────────────────────────────────────────── */}
      <div className="rv-stats-row">
        <div className="rv-stat-card">
          <span className="rv-stat-label">PLANTS</span>
          <span className="rv-stat-value">{result.plants.length}</span>
        </div>
        <div className="rv-stat-card">
          <span className="rv-stat-label">PLANTS DETECTED</span>
          <span className="rv-stat-value">{result.plants.length}</span>
        </div>
        <div className="rv-stat-card">
          <span className="rv-stat-label">LATENCY</span>
          <span className="rv-stat-value">
            {meta.processing_time_ms}
            <small>ms</small>
          </span>
        </div>
        <div className="rv-stat-card">
          <span className="rv-stat-label">SCALE</span>
          <span className="rv-stat-value">
            {meta.scale_factor_mm_per_px.toFixed(3)}
            <small> mm/px</small>
          </span>
        </div>
      </div>

      {/* ── Body: mask + table side-by-side ───────────────────── */}
      <div className="rv-body">
        {/* Left: Mask + Tip Overlay */}
        <div className="rv-overlay-panel">
          <div className="rv-overlay-header">
            <span className="rv-overlay-title">MASK + TIP OVERLAY</span>
            <div className="rv-overlay-toggles">
              {overlayModes.map((mode) => (
                <button
                  key={mode}
                  className={`rv-toggle-btn ${overlayMode === mode ? 'rv-toggle-btn--active' : ''}`}
                  onClick={() => setOverlayMode(mode)}
                >
                  {mode.toUpperCase()}
                </button>
              ))}
            </div>
          </div>
          <div className="rv-overlay-canvas">
            {displayImageUrl ? (
              <img
                src={displayImageUrl}
                alt={`${overlayMode} view`}
                className="rv-overlay-img"
              />
            ) : (
              <span className="rv-overlay-placeholder">
                {result.input_filename || file?.name || 'scan'} +
                segmentation_mask.png · {result.plants.length} tip markers
              </span>
            )}
            {overlayMode === 'mask' && result.segmentation_mask_url && (
              <button
                className="ann-edit-mask-btn"
                title="Fine-tune segmentation mask"
                onClick={() => setIsEditorOpen(true)}
              >
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none"
                  stroke="currentColor" strokeWidth="2.5"
                  strokeLinecap="round" strokeLinejoin="round">
                  <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7" />
                  <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z" />
                </svg>
                Edit Mask
              </button>
            )}
          </div>
        </div>

        {/* Right: Per-plant table + feedback */}
        <div className="rv-table-panel">
          <h3 className="rv-table-title">PER-PLANT TABLE</h3>
          <div className="rv-table-wrap">
            <table className="rv-table">
              <thead>
                <tr>
                  <th>#</th>
                  <th>TIP (X,Y)</th>
                  <th>LENGTH (mm)</th>
                  <th>LENGTH (px)</th>
                </tr>
              </thead>
              <tbody>
                {result.plants.map((p: Plant) => (
                  <tr key={p.plant_id}>
                    <td className="rv-cell-id">#{p.plant_id}</td>
                    <td>
                      {p.tip_x}, {p.tip_y}
                    </td>
                    <td>{p.root_length_mm.toFixed(1)} mm</td>
                    <td className="rv-cell-px">{Math.round(p.root_length_px)} px</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Feedback widget */}
          <div className="rv-feedback">
            <span className="rv-feedback-label">WAS THIS CORRECT?</span>
            <div className="rv-feedback-btns">
              <button
                className={`rv-fb-btn rv-fb-btn--incorrect ${feedback === 'incorrect' ? 'rv-fb-btn--selected' : ''}`}
                onClick={() => {
                  const next = feedback === 'incorrect' ? null : 'incorrect';
                  setFeedback(next);
                  onFeedbackChange?.(next);
                }}
              >
                Incorrect
              </button>
              <button
                className={`rv-fb-btn rv-fb-btn--correct ${feedback === 'correct' ? 'rv-fb-btn--selected' : ''}`}
                onClick={() => {
                  const next = feedback === 'correct' ? null : 'correct';
                  setFeedback(next);
                  onFeedbackChange?.(next);
                }}
              >
                Correct
              </button>
            </div>
          </div>
        </div>
      </div>

      <MaskEditor
        isOpen={isEditorOpen}
        onClose={() => {
          setIsEditorOpen(false);
          setMaskSaveError(null);
        }}
        onSave={async (blob) => {
          setMaskSaving(true);
          setMaskSaveError(null);
          try {
            // 1. Save the prediction and create the annotation (no metadata needed)
            const confirmed = await confirmPrediction(result.prediction_id);
            // 2. Submit the corrected mask against the new annotation
            await submitCorrectedMask(confirmed.annotation_id, blob);
            // 3. Close everything silently
            setIsEditorOpen(false);
            setFeedback('incorrect');
            onFeedbackChange?.('incorrect');
            onNewScan?.();
          } catch (err) {
            setMaskSaveError((err as Error).message || 'Failed to save mask.');
          } finally {
            setMaskSaving(false);
          }
        }}
        originalImageUrl={getArtifactUrl(result.input_url) || filePreviewUrl || ''}
        maskImageUrl={getArtifactUrl(result.segmentation_mask_url) || ''}
        saveLabel={maskSaving ? 'Saving…' : 'Apply Correction'}
      />
      {maskSaveError && (
        <p style={{
          color: '#b91c1c',
          fontSize: '0.8rem',
          marginTop: '0.5rem',
          textAlign: 'center',
        }}>
          {maskSaveError}
        </p>
      )}
    </div>
  );
}

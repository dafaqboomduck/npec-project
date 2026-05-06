import { useState } from 'react';
import type { PredictionResponse, Plant } from '../types/api';
import './ResultsView.css';

export type OverlayMode = 'scan' | 'mask' | 'tips';

interface ResultsViewProps {
  result: PredictionResponse;
  file?: File | null; // Local file from upload session
  filePreviewUrl: string | null;
  deployedModelName: string;
  onNewScan?: () => void;
  onDiscard?: () => void;
  onSave?: () => void;
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
  isHistoryView = false,
}: ResultsViewProps) {
  const [overlayMode, setOverlayMode] = useState<OverlayMode>('scan');
  const [feedback, setFeedback] = useState<'correct' | 'incorrect' | null>(null);
  const meta = result.metadata;
  const predIdShort = result.prediction_id.slice(0, 6);

  const overlayModes: OverlayMode[] = ['scan', 'mask', 'tips'];

  const displayImageUrl =
    overlayMode === 'mask' && result.segmentation_mask_url
      ? result.segmentation_mask_url
      : filePreviewUrl;

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
            <span className="rv-plant-badge">{result.plants.length} PLANTS</span>
          </div>
          <p className="rv-subtitle">
            prediction p_{predIdShort} · {result.input_filename || file?.name || 'scan'} · {deployedModelName}
          </p>
        </div>
        {!isHistoryView && (
          <div className="rv-header-actions">
            <button className="rv-btn rv-btn--outline" id="btn-new-scan" onClick={onNewScan}>
              + New scan
            </button>
            <button className="rv-btn rv-btn--danger" id="btn-discard-result" onClick={onDiscard}>
              Discard
            </button>
            <button className="rv-btn rv-btn--primary" id="btn-save-result" onClick={onSave}>
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
          <span className="rv-stat-label">AVG CONFIDENCE</span>
          <span className="rv-stat-value">
            {(result.overall_confidence * 100).toFixed(1)}%
          </span>
        </div>
        <div className="rv-stat-card">
          <span className="rv-stat-label">LATENCY</span>
          <span className="rv-stat-value">
            {meta.processing_time_ms}<small>ms</small>
          </span>
        </div>
        <div className="rv-stat-card">
          <span className="rv-stat-label">SCALE</span>
          <span className="rv-stat-value">
            {meta.scale_factor_mm_per_px.toFixed(3)}<small> mm/px</small>
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
                {result.input_filename || file?.name || 'scan'} + segmentation_mask.png · {result.plants.length} tip markers
              </span>
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
                  <th>LENGTH</th>
                  <th>CONF.</th>
                </tr>
              </thead>
              <tbody>
                {result.plants.map((p: Plant) => (
                  <tr key={p.plant_id}>
                    <td className="rv-cell-id">#{p.plant_id}</td>
                    <td>{p.tip_x}, {p.tip_y}</td>
                    <td>{p.root_length_mm.toFixed(1)} mm</td>
                    <td className={p.confidence < 0.8 ? 'rv-cell-low' : ''}>
                      {(p.confidence * 100).toFixed(0)}%
                    </td>
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
                onClick={() => setFeedback('incorrect')}
              >
                Incorrect
              </button>
              <button
                className={`rv-fb-btn rv-fb-btn--correct ${feedback === 'correct' ? 'rv-fb-btn--selected' : ''}`}
                onClick={() => setFeedback('correct')}
              >
                Correct
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

import { useState, useEffect } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import NavBar from '../components/NavBar';
import ResultsView from '../components/ResultsView';
import {
  getPredictions,
  deletePrediction,
  getPrediction,
} from '../api/inferenceApi';
import type { PredictionResponse } from '../types/api';
import './History.css';

export default function History() {
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();
  const [predictions, setPredictions] = useState<PredictionResponse[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedPrediction, setSelectedPrediction] =
    useState<PredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [discarding, setDiscarding] = useState(false);

  // Search & Filter state
  const [searchTerm, setSearchTerm] = useState('');
  const [modelFilter, setModelFilter] = useState('');

  useEffect(() => {
    const predId = searchParams.get('prediction_id');
    if (predId) {
      setLoading(true);
      getPrediction(predId)
        .then((pred) => {
          setSelectedPrediction(pred);
          void getPredictions(1, 50)
            .then(setPredictions)
            .catch(() => undefined);
        })
        .catch((err) => {
          console.error('Failed to load specific prediction:', err);
          fetchHistory();
        })
        .finally(() => setLoading(false));
    } else {
      fetchHistory();
    }
  }, [searchParams]);

  async function fetchHistory() {
    setLoading(true);
    setError(null);
    try {
      const data = await getPredictions(1, 50);
      setPredictions(data);
    } catch (err) {
      setError('Failed to load inference history.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  }

  async function handleDiscard(pred: PredictionResponse, e?: React.MouseEvent) {
    if (e) e.stopPropagation();
    if (
      !window.confirm(
        'Delete this prediction from history? This cannot be undone.',
      )
    )
      return;
    setDiscarding(true);
    try {
      await deletePrediction(pred.prediction_id);
      if (selectedPrediction?.prediction_id === pred.prediction_id) {
        handleBack();
      }
      await fetchHistory();
    } catch (err) {
      setError((err as Error).message || 'Failed to delete prediction.');
    } finally {
      setDiscarding(false);
    }
  }

  function handleBack() {
    setSelectedPrediction(null);
    setSearchParams({}); // Perfectly clears ?prediction_id from the browser URL!
  }

  function formatDate(isoString: string) {
    return new Date(isoString).toLocaleString('en-GB', {
      day: '2-digit',
      month: 'short',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  }

  // Filter predictions
  const filteredPredictions = predictions.filter((pred) => {
    const matchesSearch =
      (pred.input_filename || '')
        .toLowerCase()
        .includes(searchTerm.toLowerCase()) ||
      (pred.metadata.species || '')
        .toLowerCase()
        .includes(searchTerm.toLowerCase()) ||
      (pred.metadata.treatment || '')
        .toLowerCase()
        .includes(searchTerm.toLowerCase());
    const matchesModel =
      modelFilter === '' || pred.metadata.model_version === modelFilter;
    return matchesSearch && matchesModel;
  });

  // Extract unique model versions for filter dropdown
  const uniqueModels = Array.from(
    new Set(predictions.map((p) => p.metadata.model_version)),
  ).filter(Boolean);

  return (
    <div className="history-page">
      <NavBar />

      <main className="history-main">
        <div className="history-container">
          {selectedPrediction ? (
            <div className="history-detail-view">
              <button className="history-back-btn" onClick={handleBack}>
                ← Back to History
              </button>
              <ResultsView
                result={selectedPrediction}
                filePreviewUrl={
                  selectedPrediction.input_url ??
                  selectedPrediction.segmentation_mask_url ??
                  null
                }
                deployedModelName={selectedPrediction.metadata.model_version}
                isHistoryView={true}
                onDiscard={() => void handleDiscard(selectedPrediction)}
              />
              {discarding && (
                <p
                  style={{
                    textAlign: 'center',
                    color: 'var(--color-text-muted)',
                    marginTop: '1rem',
                  }}
                >
                  Deleting...
                </p>
              )}
            </div>
          ) : (
            <div className="history-list-view">
              <header className="history-header">
                <div className="history-header-left">
                  <span className="history-breadcrumb">
                    INFERENCE / HISTORY
                  </span>
                  <h1 className="history-title">Inference History</h1>
                  <p className="history-subtitle">
                    Review past detections and U-Net segmentation results.
                  </p>
                </div>
                <button
                  className="history-refresh-btn"
                  onClick={fetchHistory}
                  disabled={loading}
                >
                  {loading ? 'Refreshing...' : 'Refresh History'}
                </button>
              </header>

              {error && <div className="history-error">{error}</div>}

              {/* Real-time search & filters bar */}
              {predictions.length > 0 && (
                <div className="lib-toolbar" style={{ margin: '1.5rem 0' }}>
                  <div className="lib-toolbar-left" style={{ flex: 1 }}>
                    <input
                      type="text"
                      className="lib-search-input"
                      placeholder="Search filename, species, or treatment..."
                      value={searchTerm}
                      onChange={(e) => setSearchTerm(e.target.value)}
                      style={{ flex: 1, maxWidth: '400px' }}
                    />
                    <div className="lib-filter-select-wrap">
                      <select
                        className="lib-filter-select"
                        value={modelFilter}
                        onChange={(e) => setModelFilter(e.target.value)}
                      >
                        <option value="">All Models</option>
                        {uniqueModels.map((m) => (
                          <option key={m} value={m}>
                            {m}
                          </option>
                        ))}
                      </select>
                    </div>
                  </div>
                </div>
              )}

              {loading && predictions.length === 0 ? (
                <div className="history-loading">
                  <span className="spinner" aria-hidden="true" />
                  <p>Loading your history...</p>
                </div>
              ) : predictions.length === 0 ? (
                <div
                  className="history-empty"
                  style={{ textAlign: 'center', padding: '3rem 1rem' }}
                >
                  <h2>No history found</h2>
                  <p
                    style={{
                      color: 'var(--color-text-muted)',
                      marginBottom: '1.5rem',
                    }}
                  >
                    You haven't run any inference scans yet.
                  </p>
                  <button
                    className="history-refresh-btn"
                    onClick={() => navigate('/inference')}
                  >
                    + Launch Inference Lab
                  </button>
                </div>
              ) : filteredPredictions.length === 0 ? (
                <div style={{ textAlign: 'center', padding: '3rem' }}>
                  <p style={{ color: 'var(--color-text-muted)' }}>
                    No logs match your search criteria.
                  </p>
                </div>
              ) : (
                <div className="history-grid">
                  <table className="history-table">
                    <thead>
                      <tr>
                        <th>Date</th>
                        <th>Filename</th>
                        <th>Model</th>
                        <th>Species</th>
                        <th>Treatment</th>
                        <th>Plants</th>
                        <th style={{ width: '120px' }}>Actions</th>
                      </tr>
                    </thead>
                    <tbody>
                      {filteredPredictions.map((pred) => (
                        <tr
                          key={pred.prediction_id}
                          onClick={() => setSelectedPrediction(pred)}
                          style={{ cursor: 'pointer' }}
                        >
                          <td>
                            {formatDate(pred.metadata.inference_timestamp)}
                          </td>
                          <td className="history-filename">
                            {pred.input_filename || 'Unknown Scan'}
                          </td>
                          <td>{pred.metadata.model_version}</td>
                          <td>{pred.metadata.species || '—'}</td>
                          <td>{pred.metadata.treatment || '—'}</td>
                          <td>
                            <span className="history-plant-count">
                              {pred.plants.length}
                            </span>
                          </td>
                          <td>
                            <div
                              style={{
                                display: 'flex',
                                gap: '8px',
                                alignItems: 'center',
                              }}
                            >
                              <button
                                className="history-view-btn"
                                style={{
                                  padding: '4px 8px',
                                  fontSize: '11px',
                                  margin: 0,
                                }}
                              >
                                View
                              </button>
                              <button
                                className="lib-ds-delete-btn"
                                title="Delete prediction"
                                onClick={(e) => void handleDiscard(pred, e)}
                                style={{
                                  margin: 0,
                                  padding: '4px',
                                  display: 'flex',
                                  alignItems: 'center',
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
                                </svg>
                              </button>
                            </div>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          )}
        </div>
      </main>
    </div>
  );
}

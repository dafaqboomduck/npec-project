import { useState, useEffect } from 'react';
import NavBar from '../components/NavBar';
import ResultsView from '../components/ResultsView';
import { getPredictions } from '../api/inferenceApi';
import type { PredictionResponse } from '../types/api';
import './History.css';

export default function History() {
  const [predictions, setPredictions] = useState<PredictionResponse[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedPrediction, setSelectedPrediction] = useState<PredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchHistory();
  }, []);

  async function fetchHistory() {
    setLoading(true);
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

  function formatDate(isoString: string) {
    return new Date(isoString).toLocaleString('en-GB', {
      day: '2-digit',
      month: 'short',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  }

  return (
    <div className="history-page">
      <NavBar />

      <main className="history-main">
        <div className="history-container">
          {selectedPrediction ? (
            <div className="history-detail-view">
              <button 
                className="history-back-btn"
                onClick={() => setSelectedPrediction(null)}
              >
                ← Back to History
              </button>
              <ResultsView 
                result={selectedPrediction}
                filePreviewUrl={selectedPrediction.segmentation_mask_url || null}
                deployedModelName={selectedPrediction.metadata.model_version}
                isHistoryView={true}
              />
            </div>
          ) : (
            <div className="history-list-view">
              <header className="history-header">
                <div className="history-header-left">
                  <span className="history-breadcrumb">INFERENCE / HISTORY</span>
                  <h1 className="history-title">Inference History</h1>
                  <p className="history-subtitle">
                    Review past detections and segmentation results.
                  </p>
                </div>
                <button className="history-refresh-btn" onClick={fetchHistory} disabled={loading}>
                  {loading ? 'Refreshing...' : 'Refresh'}
                </button>
              </header>

              {error && <div className="history-error">{error}</div>}

              {loading && predictions.length === 0 ? (
                <div className="history-loading">
                  <div className="spinner"></div>
                  <p>Loading your history...</p>
                </div>
              ) : predictions.length === 0 ? (
                <div className="history-empty">
                  <div className="history-empty-icon">📊</div>
                  <h2>No history found</h2>
                  <p>You haven't run any inferences yet. Go to the Inference Lab to get started!</p>
                </div>
              ) : (
                <div className="history-grid">
                  <table className="history-table">
                    <thead>
                      <tr>
                        <th>Date</th>
                        <th>Filename</th>
                        <th>Model</th>
                        <th>Plants</th>
                        <th>Conf.</th>
                        <th></th>
                      </tr>
                    </thead>
                    <tbody>
                      {predictions.map((pred) => (
                        <tr key={pred.prediction_id} onClick={() => setSelectedPrediction(pred)}>
                          <td>{formatDate(pred.metadata.inference_timestamp)}</td>
                          <td className="history-filename">{pred.input_filename || 'Unknown Scan'}</td>
                          <td>{pred.metadata.model_version}</td>
                          <td>
                            <span className="history-plant-count">{pred.plants.length}</span>
                          </td>
                          <td>
                            <span className={`history-conf-badge ${pred.overall_confidence > 0.8 ? 'high' : 'low'}`}>
                              {(pred.overall_confidence * 100).toFixed(1)}%
                            </span>
                          </td>
                          <td>
                            <button className="history-view-btn">View Details</button>
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

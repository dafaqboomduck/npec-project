/**
 * pages/Library.tsx
 *
 * Library page — browse datasets, annotations, and exports.
 * Left sidebar: user info, new dataset button, navigation, filters.
 * Main panel: dataset grid or empty state.
 *
 * Route: /library (protected)
 */

import { useEffect, useState } from 'react';
import NavBar from '../components/NavBar';
import { getUsername, getRole } from '../api/auth';
import {
  getDatasets,
  createDataset,
  deleteDataset,
  getDatasetDetail,
} from '../api/libraryApi';
import type { DatasetSummary, DatasetDetail } from '../api/libraryApi';
import './Library.css';

type LibraryTab = 'datasets' | 'annotations' | 'exports';

export default function Library() {
  const username = getUsername();
  const role = getRole();
  const displayName = username.includes('@')
    ? username
        .split('@')[0]
        .replace(/[._-]/g, ' ')
        .replace(/\b\w/g, (c) => c.toUpperCase())
    : username;

  // ── State ──
  const [activeTab, setActiveTab] = useState<LibraryTab>('datasets');
  const [datasets, setDatasets] = useState<DatasetSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [speciesFilter, setSpeciesFilter] = useState('');
  const [growthStages, setGrowthStages] = useState<string[]>([]);
  const [dbUp, setDbUp] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const [sortOrder, setSortOrder] = useState<'desc' | 'asc'>('desc');

  const [showCreateModal, setShowCreateModal] = useState(false);
  const [createData, setCreateData] = useState({
    name: '',
    description: '',
    species: '',
    growth_stage: '',
  });
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
      } catch {
        setDatasets([]);
        setDbUp(false);
      } finally {
        setLoading(false);
      }
    };

    fetchDatasets();
  }, [speciesFilter, growthStages]);

  useEffect(() => {
    const fetchDetail = async () => {
      if (selectedDatasetId) {
        await Promise.resolve();
        setDetailLoading(true);
        try {
          const detail = await getDatasetDetail(selectedDatasetId);
          setDatasetDetail(detail);
        } catch (error) {
          console.error(error);
        } finally {
          setDetailLoading(false);
        }
      } else {
        setDatasetDetail(null);
      }
    };

    fetchDetail();
  }, [selectedDatasetId]);

  function toggleGrowthStage(stage: string) {
    setLoading(true);
    setGrowthStages((prev) =>
      prev.includes(stage) ? prev.filter((s) => s !== stage) : [...prev, stage],
    );
  }

  async function handleCreateDataset(e: React.FormEvent) {
    e.preventDefault();
    setCreateError(null);
    setCreating(true);
    try {
      await createDataset({
        name: createData.name,
        description: createData.description || undefined,
        species: createData.species || undefined,
        growth_stage: createData.growth_stage || undefined,
      });
      setShowCreateModal(false);
      setCreateData({
        name: '',
        description: '',
        species: '',
        growth_stage: '',
      });
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
  const navItems: { key: LibraryTab; label: string; icon: JSX.Element }[] = [
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
      key: 'exports',
      label: 'Exports',
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
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
          <polyline points="7 10 12 15 17 10" />
          <line x1="12" y1="15" x2="12" y2="3" />
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
                <span className="lib-user-name">
                  {role === 'admin' ? 'Library Admin' : displayName}
                </span>
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

            {/* New Dataset button */}
            {role === 'admin' && (
              <button
                className="lib-btn-new"
                id="btn-new-dataset"
                onClick={() => setShowCreateModal(true)}
              >
                + New Dataset
              </button>
            )}

            {/* Navigation */}
            <div className="lib-nav-section">
              <span className="lib-nav-label">NAVIGATION</span>
              <ul className="lib-nav-list">
                {navItems.map((item) => (
                  <li key={item.key}>
                    <button
                      className={`lib-nav-item ${activeTab === item.key ? 'lib-nav-item--active' : ''}`}
                      onClick={() => setActiveTab(item.key)}
                    >
                      {item.icon}
                      {item.label}
                    </button>
                  </li>
                ))}
              </ul>
            </div>

            {/* Filters */}
            <div className="lib-filter-section">
              <span className="lib-nav-label">FILTERS</span>

              <div className="lib-filter-group">
                <span className="lib-filter-label">GROWTH STAGE</span>
                <div className="lib-filter-chips">
                  {['seedling', 'mature'].map((stage) => (
                    <button
                      key={stage}
                      className={`lib-chip ${growthStages.includes(stage) ? 'lib-chip--active' : ''}`}
                      onClick={() => toggleGrowthStage(stage)}
                    >
                      {stage.charAt(0).toUpperCase() + stage.slice(1)}
                    </button>
                  ))}
                </div>
              </div>
            </div>

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
            {selectedDatasetId ? (
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
                    <h1 className="lib-content-title">{datasetDetail.name}</h1>
                    <div className="lib-ds-list-meta">
                      <span className="lib-ds-list-meta-item">
                        <span className="lib-ds-meta-label">created at:</span>{' '}
                        <span className="lib-ds-meta-val">
                          {new Date(datasetDetail.created_at)
                            .toISOString()
                            .split('.')[0] + 'Z'}
                        </span>
                      </span>
                      <span className="lib-ds-list-meta-item">
                        <span className="lib-ds-meta-label">modified at:</span>{' '}
                        <span className="lib-ds-meta-val">
                          {new Date(datasetDetail.modified_at)
                            .toISOString()
                            .split('.')[0] + 'Z'}
                        </span>
                      </span>
                      <span className="lib-ds-list-meta-item">
                        <span className="lib-ds-meta-label">Species:</span>{' '}
                        <span className="lib-ds-meta-val">
                          {datasetDetail.species ?? 'Arabidopsis'}
                        </span>
                      </span>
                      <span className="lib-ds-list-meta-item">
                        <span className="lib-ds-meta-label">count:</span>{' '}
                        <span className="lib-ds-meta-val">
                          {datasetDetail.count}
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
                        <h3 className="lib-detail-card-title">
                          Scans ({datasetDetail.count.toLocaleString()})
                        </h3>
                        <div className="lib-scans-grid">
                          {Array.from({ length: 15 }).map((_, i) => (
                            <div key={i} className="lib-scan-placeholder" />
                          ))}
                        </div>
                      </div>
                    </div>

                    <div className="lib-detail-side">
                      <div className="lib-detail-card">
                        <h3 className="lib-detail-card-title">STATS</h3>
                        <div className="lib-stats-grid">
                          <div className="lib-stat-box">
                            <span className="lib-stat-label">Scans</span>
                            <span className="lib-stat-val">
                              {datasetDetail.count.toLocaleString()}
                            </span>
                          </div>
                          <div className="lib-stat-box">
                            <span className="lib-stat-label">Annotated</span>
                            <span className="lib-stat-val">
                              {Math.floor(
                                datasetDetail.count * 0.74,
                              ).toLocaleString()}
                            </span>
                          </div>
                          <div className="lib-stat-box">
                            <span className="lib-stat-label">Pending</span>
                            <span className="lib-stat-val">
                              {(
                                datasetDetail.count -
                                Math.floor(datasetDetail.count * 0.74)
                              ).toLocaleString()}
                            </span>
                          </div>
                          <div className="lib-stat-box">
                            <span className="lib-stat-label">Plates</span>
                            <span className="lib-stat-val">86</span>
                          </div>
                        </div>
                      </div>

                      <div className="lib-detail-card">
                        <h3 className="lib-detail-card-title">METADATA</h3>
                        <div className="lib-meta-grid">
                          <div className="lib-meta-row">
                            <span className="lib-meta-k">Species</span>
                            <span className="lib-meta-v">
                              {datasetDetail.species || 'Unknown'}
                            </span>
                          </div>
                          <div className="lib-meta-row">
                            <span className="lib-meta-k">Stage</span>
                            <span className="lib-meta-v">seedling</span>
                          </div>
                          <div className="lib-meta-row">
                            <span className="lib-meta-k">Treatment</span>
                            <span className="lib-meta-v">NaCl 100mM</span>
                          </div>
                          <div className="lib-meta-row">
                            <span className="lib-meta-k">Created by</span>
                            <span className="lib-meta-v">admin@npec.nl</span>
                          </div>
                        </div>
                      </div>
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
                      {loading
                        ? 'Loading…'
                        : datasets.length === 0 &&
                            !speciesFilter &&
                            !searchQuery &&
                            growthStages.length === 0
                          ? 'No datasets created'
                          : 'Datasets'}
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
                        <div className="lib-filter-select-wrap">
                          <select
                            className="lib-filter-select lib-toolbar-select"
                            value={speciesFilter}
                            onChange={(e) => {
                              setLoading(true);
                              setSpeciesFilter(e.target.value);
                            }}
                          >
                            <option value="">All species</option>
                            <option value="Arabidopsis thaliana">
                              Arabidopsis thaliana
                            </option>
                            <option value="Triticum aestivum">
                              Triticum aestivum
                            </option>
                            <option value="Oryza sativa">Oryza sativa</option>
                          </select>
                        </div>
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
                                  modified at:
                                </span>{' '}
                                <span className="lib-ds-meta-val">
                                  {new Date(ds.modified_at)
                                    .toISOString()
                                    .split('.')[0] + 'Z'}
                                </span>
                              </span>
                              <span className="lib-ds-list-meta-item">
                                <span className="lib-ds-meta-label">
                                  Species:
                                </span>{' '}
                                <span className="lib-ds-meta-val">
                                  {ds.species ?? 'Arabidopsis'}
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
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )}

                {activeTab === 'annotations' && (
                  <div className="lib-empty">
                    <p className="lib-empty-text">
                      Annotations view coming soon.
                    </p>
                  </div>
                )}

                {activeTab === 'exports' && (
                  <div className="lib-empty">
                    <p className="lib-empty-text">Exports view coming soon.</p>
                  </div>
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
    </div>
  );
}

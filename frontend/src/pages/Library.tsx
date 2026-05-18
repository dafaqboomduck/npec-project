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
} from '../api/libraryApi';
import type { DatasetSummary, DatasetDetail, DatasetImportResponse } from '../api/libraryApi';
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
  const [importTargetMode, setImportTargetMode] = useState<'new' | 'existing'>('new');
  const [importDatasetId, setImportDatasetId] = useState<string>('');
  const [importNewName, setImportNewName] = useState('');
  const [importNewDesc, setImportNewDesc] = useState('');
  const [importNewSpecies, setImportNewSpecies] = useState('');
  const [importNewGrowth, setImportNewGrowth] = useState('');
  const [importProgress, setImportProgress] = useState(0);
  const [importProcessing, setImportProcessing] = useState(false);
  const [importReport, setImportReport] = useState<DatasetImportResponse | null>(null);
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
      setUploadFiles(prev => [...prev, ...Array.from(e.dataTransfer.files)]);
    }
  }

  function handleFileSelect(e: React.ChangeEvent<HTMLInputElement>) {
    if (e.target.files && e.target.files.length > 0) {
      setUploadFiles(prev => [...prev, ...Array.from(e.target.files!)]);
    }
  }

  async function handleUpload() {
    if (!selectedDatasetId || uploadFiles.length === 0) return;
    setIsUploading(true);
    setUploadError(null);
    try {
      await uploadDatasetImages(selectedDatasetId, uploadFiles);
      setShowUploadModal(false);
      setUploadFiles([]);
      // Refresh detail
      const detail = await getDatasetDetail(selectedDatasetId);
      setDatasetDetail(detail);
      // Refresh list
      const data = await getDatasets({
        species: speciesFilter || undefined,
        growth_stage: growthStages.length === 1 ? growthStages[0] : undefined,
      });
      setDatasets(data);
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
      setImportProgress(prev => {
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
        datasetId: importTargetMode === 'existing' ? importDatasetId : undefined,
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

            {/* New Dataset & Import buttons */}
            {role === 'admin' && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', marginBottom: '1.5rem', width: '100%' }}>
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
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" style={{ marginRight: '6px' }}><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
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
                    <div className="lib-detail-header-top">
                      <h1 className="lib-content-title">{datasetDetail.name}</h1>
                      <button 
                        className="lib-btn-add-scans" 
                        onClick={() => setShowUploadModal(true)}
                      >
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
                        Add Scans
                      </button>
                    </div>
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
                <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
              </div>
              <p className="lib-upload-text">Click or drag images here to upload</p>
              <p className="lib-upload-hint">Supported formats: PNG, JPG, TIFF</p>
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
                    <span className="lib-upload-item-name" title={file.name}>{file.name}</span>
                    <span className={`lib-upload-item-status ${isUploading ? 'uploading' : 'pending'}`}>
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
                {isUploading ? 'Uploading...' : `Upload ${uploadFiles.length > 0 ? uploadFiles.length + ' ' : ''}Images`}
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
              <div className={`wizard-step ${importStep >= 1 ? 'wizard-step--completed' : ''} ${importStep === 1 ? 'wizard-step--active' : ''}`}>1</div>
              <div className={`wizard-step ${importStep >= 2 ? 'wizard-step--completed' : ''} ${importStep === 2 ? 'wizard-step--active' : ''}`}>2</div>
              <div className={`wizard-step ${importStep === 3 ? 'wizard-step--completed' : ''} ${importStep === 3 ? 'wizard-step--active' : ''}`}>3</div>
            </div>

            {/* STEP 1: UPLOAD ZIP */}
            {importStep === 1 && (
              <div>
                <p className="modal-body" style={{ marginBottom: '1.25rem' }}>
                  Upload a single <strong>.zip</strong> archive containing your raw root scan images. 
                  Pre-existing masks matching filenames (e.g. <code>image_mask.png</code>) will be imported as validated annotations.
                </p>
                <div
                  className={`lib-upload-dropzone ${isDragging ? 'lib-upload-dropzone--active' : ''}`}
                  onDragOver={handleZipDragOver}
                  onDragLeave={handleZipDragLeave}
                  onDrop={handleZipDrop}
                  onClick={() => zipInputRef.current?.click()}
                >
                  <div className="lib-upload-icon">
                    <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
                  </div>
                  <p className="lib-upload-text">Click or drag ZIP archive here</p>
                  <p className="lib-upload-hint">Supported archive: ZIP up to 200MB</p>
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
                  Choose where to import the root scans from <strong>{importZipFile?.name}</strong>:
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
                    <span className="target-card-desc">Extract files into a brand new dataset collection.</span>
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
                    <span className="target-card-desc">Append scans directly into a previously created dataset.</span>
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
                      <label htmlFor="import-target-ds">Select Destination Dataset *</label>
                      <select
                        id="import-target-ds"
                        required
                        value={importDatasetId}
                        onChange={(e) => setImportDatasetId(e.target.value)}
                      >
                        <option value="">-- Choose a Dataset --</option>
                        {datasets.map(ds => (
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
                    <p className="modal-body" style={{ fontSize: '0.8rem', color: 'var(--color-text-muted)', marginTop: '0.5rem' }}>
                      Extracting files, executing U-Net models for segmentations, and writing metadata records. Please do not close this modal.
                    </p>
                  </div>
                ) : (
                  <div>
                    <div className="import-success-summary">
                      <div className="import-success-number">{importReport?.scans_imported}</div>
                      <div className="import-success-label">Scans Successfully Imported</div>
                      <p className="modal-body" style={{ marginTop: '0.5rem', fontSize: '0.82rem' }}>
                        All root scans are loaded into the dataset <strong>{importReport?.dataset_name}</strong>.
                      </p>
                    </div>

                    {importReport && importReport.warnings.length > 0 && (
                      <div className="import-report-box">
                        <div className="import-report-title">Import Warnings ({importReport.warnings.length})</div>
                        <ul className="import-report-list">
                          {importReport.warnings.map((warn, idx) => (
                            <li key={idx} className="import-report-item">
                              <span>⚠️</span> {warn}
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}

                    <div className="modal-actions" style={{ marginTop: '1.5rem' }}>
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
    </div>
  );
}

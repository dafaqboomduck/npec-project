/**
 * types/api.ts
 *
 * TypeScript interfaces mirroring every schema defined in api_schema.md.
 * Keep in sync with the FastAPI Pydantic models on the backend.
 */

// ===========================================================================
// Auth
// ===========================================================================

export interface AuthTokens {
  access_token: string;
  token_type: 'bearer';
  expires_in: number;
  refresh_token: string;
  role: 'user' | 'admin';
}

// ===========================================================================
// Health
// ===========================================================================

export interface HealthResponse {
  status: 'ok' | 'degraded' | 'unavailable';
  version: string;
  timestamp: string; // ISO 8601
}

// ===========================================================================
// Overview
// ===========================================================================

export interface OverviewStatusResponse {
  total_experiments: number;
  active_model_count: number;
  data_collected_tb: number;
  quota_percent: number; // 0–100
}

export type ActivityEventType =
  | 'retraining'
  | 'upload'
  | 'annotation'
  | 'deployment';

export interface ActivityItem {
  event_type: ActivityEventType;
  title: string;
  description: string;
  timestamp: string; // ISO 8601
  resource_url?: string;
  /** Tags rendered as badges in the UI (e.g. EPOCH_50, VAL_B) */
  tags?: string[];
}

export interface ActivityFeedResponse {
  items: ActivityItem[];
  total: number;
  page: number;
  page_size: number;
}

export type DriftStatus = 'none' | 'low' | 'medium' | 'high';

export interface ModelHealthResponse {
  f1_score: number; // 0.0–1.0
  latency_ms: number;
  drift_status: DriftStatus;
}

export interface EnvironmentResponse {
  gpu_temp_celsius: number;
  memory_percent: number; // 0–100
  system_status: 'optimal' | 'degraded' | 'critical';
}

// ===========================================================================
// Dashboard
// ===========================================================================

export interface DashboardMetricsResponse {
  f1_score: number;
  throughput_img_per_s: number;
}

export interface F1HistoryPoint {
  epoch: number;
  f1_score: number;
}

export interface F1HistoryResponse {
  model_version: string;
  data_points: F1HistoryPoint[];
}

export interface SpeciesCount {
  species: string;
  count: number;
}

export interface DataDistributionResponse {
  species_counts: SpeciesCount[];
}

// ===========================================================================
// Library — Datasets
// ===========================================================================

export interface DatasetSummary {
  dataset_id: string; // UUID
  name: string;
  created_at: string;
  modified_at: string;
  count: number;
}

export interface DatasetDetail extends DatasetSummary {
  description?: string;
  species?: string;
}

// ===========================================================================
// Library — Models
// ===========================================================================

export type ModelStatus = 'deployed' | 'registered' | 'archived';

export interface ModelSummary {
  model_id: string; // UUID
  name: string;
  status: ModelStatus;
  created_at: string;
}

export interface ModelDetail extends ModelSummary {
  description?: string;
  f1_score?: number;
  precision?: number;
  recall?: number;
  inference_latency_ms?: number;
}

// ===========================================================================
// Inference / Predictions
// ===========================================================================

export interface PredictionMetadata {
  model_version: string;
  inference_timestamp: string;
  processing_time_ms: number;
  input_resolution: [number, number];
  input_channels: number;
  normalisation_applied: boolean;
  scale_factor_mm_per_px: number;
}

export interface Plant {
  plant_id: number;
  tip_x: number;
  tip_y: number;
  confidence: number;
  root_length_px: number;
  root_length_mm: number;
}

export interface PredictionResponse {
  prediction_id: string; // UUID
  status: 'ok' | 'error';
  input_filename?: string;
  metadata: PredictionMetadata;
  segmentation_mask_url?: string;
  plants: Plant[];
  overall_confidence: number;
  warnings: string[];
}

export interface ConfirmPredictionRequest {
  scan_id: string; // UUID
}

export interface ConfirmPredictionResponse {
  message: string;
  prediction_id: string;
}

// ===========================================================================
// Annotations
// ===========================================================================

export type AnnotationStatus = 'pending' | 'annotated' | 'review_required';

export interface AnnotationResponse {
  annotation_id: string; // UUID
  scan_id: string;
  status: AnnotationStatus;
  mask_url?: string;
  species?: string;
  age_days?: number;
  treatment?: string;
  created_at: string;
  updated_at: string;
}

// ===========================================================================
// Training
// ===========================================================================

export type TrainingRunStatus = 'running' | 'completed' | 'failed' | 'stopped';

export interface TrainingStartRequest {
  learning_rate?: number;
  batch_size?: number;
  epochs?: number;
}

export interface TrainingRunSummary {
  run_id: string; // UUID
  status: TrainingRunStatus;
  started_at: string;
  completed_at?: string;
  final_f1?: number;
}

export interface TrainingRunDetail extends TrainingRunSummary {
  hyperparameters?: TrainingStartRequest;
  metrics_history?: F1HistoryPoint[];
}

// ===========================================================================
// Feedback
// ===========================================================================

export interface FeedbackRequest {
  prediction_id: string;
  signal: 'correct' | 'incorrect';
  comment?: string;
}

// ===========================================================================
// Shared
// ===========================================================================

export interface MessageResponse {
  message: string;
}

export interface ErrorResponse {
  status: 'error';
  error_code: string;
  message: string;
  detail?: Record<string, unknown>;
}

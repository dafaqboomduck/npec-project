/**
 * types/api.ts
 *
 * TypeScript interfaces mirroring every schema defined in api_schema.md.
 * Keep in sync with the FastAPI Pydantic models on the backend.
 */

// ===========================================================================
// Auth — AuthTokens lives in api/auth.ts (canonical location)
// ===========================================================================

// ===========================================================================
// Health
// ===========================================================================

export interface HealthResponse {
  status: 'ok' | 'degraded' | 'unavailable';
  version: string;
  timestamp: string; // ISO 8601
  deployment_env: string;
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
  gpu_temp_celsius: number | null;
  gpu_utilisation: number | null;
  vram_percent: number | null;
  cpu_percent: number | null;
  disk_percent: number | null;
  cuda_available: boolean;
  memory_percent: number;
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
  species?: string;
  growth_stage?: string;
  owner_email?: string | null;
}

export interface DatasetDetail extends DatasetSummary {
  description?: string;
  species?: string;
  growth_stage?: string;
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

export interface ModelCreate {
  version: string;
  name: string;
  artifact_uri: string;
  description?: string;
  f1_score?: number;
  precision?: number;
  recall?: number;
  inference_latency_ms?: number;
}

export interface ModelDetail extends ModelSummary {
  version?: string;
  description?: string;
  f1_score?: number;
  precision?: number;
  recall?: number;
  inference_latency_ms?: number;
  artifact_uri?: string; // Admin only
  deployed_at?: string; // Admin only
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
  species?: string;
  treatment?: string;
}

export interface Plant {
  plant_id: number;
  tip_x: number;
  tip_y: number;
  root_length_px: number;
  root_length_mm: number;
}

export interface PredictionResponse {
  prediction_id: string; // UUID
  owner_email?: string | null;
  status: 'ok' | 'error';
  input_filename?: string;
  metadata: PredictionMetadata;
  segmentation_mask_url?: string;
  input_url?: string; // Presigned URL of the original input image
  tip_photo_url?: string; // Presigned URL of the tip visualisation overlay
  plants: Plant[];
  feedback_signal?: 'correct' | 'incorrect' | null;
  warnings: string[];
}

/** Body for POST /predictions/{id}/confirm */
export interface ConfirmPredictionRequest {
  species?: string;
  age_days?: number;
  treatment?: string;
  feedback_signal?: 'correct' | 'incorrect';
}

export interface ConfirmPredictionResponse {
  message: string;
  prediction_id: string;
  annotation_id: string;
}

// ===========================================================================
// Annotations
// ===========================================================================

export type AnnotationStatus = 'pending' | 'annotated' | 'review_required';

export interface AnnotationResponse {
  annotation_id: string; // UUID
  prediction_id: string; // UUID of the linked prediction (renamed from scan_id)
  dataset_id?: string | null; // UUID of the linked dataset
  status: AnnotationStatus;
  image_uri?: string;
  mask_url?: string;
  tip_photo_url?: string;
  species?: string;
  age_days?: number;
  treatment?: string;
  growth_stage?: string | null;
  owner_email?: string | null;
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
  hyperparameters?: Record<string, unknown>;
}

export interface TrainingRunDetail extends TrainingRunSummary {
  hyperparameters?: Record<string, unknown>;
  model_version_id?: string; // UUID of the model version produced by this run
}

export interface TrainingStartResponse {
  run_id: string;
  message: string;
}

// ===========================================================================
// Feedback
// ===========================================================================

export interface FeedbackRequest {
  prediction_id: string;
  feedback_signal: 'correct' | 'incorrect';
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

export interface InferenceQueueStatus {
  queued: number;
  running: number;
  done: number;
  failed: number;
}

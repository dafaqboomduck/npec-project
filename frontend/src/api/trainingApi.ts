/**
 * api/trainingApi.ts
 *
 * Frontend API integrations for the Admin-only Model Training page.
 * Connects to POST /training/start, GET /training/runs, GET /training/runs/{id}, and DELETE /training/runs/{id}.
 */

import { apiClient } from './client';
import type {
  TrainingStartRequest,
  TrainingStartResponse,
  TrainingRunSummary,
  TrainingRunDetail,
  MessageResponse,
} from '../types/api';

/** POST /training/start — Trigger a U-Net retraining background subprocess (admin only) */
export async function startTrainingRun(
  hyperparameters?: TrainingStartRequest,
): Promise<TrainingStartResponse> {
  return apiClient<TrainingStartResponse>('/training/start', {
    method: 'POST',
    body: hyperparameters || {},
  });
}

/** GET /training/runs — List U-Net retraining history runs ordered by started_at desc (admin only) */
export async function listTrainingRuns(): Promise<TrainingRunSummary[]> {
  return apiClient<TrainingRunSummary[]>('/training/runs');
}

/** GET /training/runs/{runId} — Fetch full details of a specific retraining run (admin only) */
export async function getTrainingRunDetail(
  runId: string,
): Promise<TrainingRunDetail> {
  return apiClient<TrainingRunDetail>(`/training/runs/${runId}`);
}

/** DELETE /training/runs/{runId} — Terminate an active training subprocess (admin only) */
export async function stopTrainingRun(runId: string): Promise<MessageResponse> {
  return apiClient<MessageResponse>(`/training/runs/${runId}`, {
    method: 'DELETE',
  });
}

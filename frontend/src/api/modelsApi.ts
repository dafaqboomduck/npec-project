/**
 * api/modelsApi.ts
 *
 * Frontend API integrations for U-Net Model Version management and deployments.
 * Connects to GET /models, POST /models, GET /models/{id}, DELETE /models/{id}, and POST /models/{id}/deploy.
 */

import { apiClient } from './client';
import type {
  ModelSummary,
  ModelDetail,
  ModelCreate,
  MessageResponse,
} from '../types/api';

/** GET /models — List all registered model versions */
export async function listModels(): Promise<ModelSummary[]> {
  return apiClient<ModelSummary[]>('/models');
}

/** POST /models — Register a new model version (admin only) */
export async function registerModel(data: ModelCreate): Promise<ModelDetail> {
  return apiClient<ModelDetail>('/models', {
    method: 'POST',
    body: data,
  });
}

/** GET /models/{model_id} — Fetch full details of a specific model version */
export async function getModelDetail(modelId: string): Promise<ModelDetail> {
  return apiClient<ModelDetail>(`/models/${modelId}`);
}

/** DELETE /models/{model_id} — Delete a registered model version (admin only) */
export async function deleteModel(modelId: string): Promise<MessageResponse> {
  return apiClient<MessageResponse>(`/models/${modelId}`, {
    method: 'DELETE',
  });
}

/** POST /models/{model_id}/deploy — Deploy a model version (admin only) */
export async function deployModel(modelId: string): Promise<MessageResponse> {
  return apiClient<MessageResponse>(`/models/${modelId}/deploy`, {
    method: 'POST',
  });
}

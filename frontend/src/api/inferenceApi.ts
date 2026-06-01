/**
 * api/inferenceApi.ts
 *
 * POST /predict and model helpers for the Inference Lab page.
 */

import { apiClient } from './client';
import type {
  PredictionResponse,
  ModelSummary,
  MessageResponse,
  ConfirmPredictionRequest,
  ConfirmPredictionResponse,
} from '../types/api';

/**
 * POST /predict
 * Submits a root scan image for U-Net inference.
 * Do NOT set Content-Type manually — FormData sets the multipart boundary.
 */
export async function predict(
  file: File,
  signal?: AbortSignal,
): Promise<PredictionResponse> {
  const form = new FormData();
  form.append('file', file);

  return apiClient<PredictionResponse>('/predict', {
    method: 'POST',
    body: form,
    signal,
  }).catch((err) => {
    if (err.name === 'AbortError') throw err;
    throw new Error(
      err.message.includes('API Error') ? `Inference failed` : err.message,
    );
  });
}

/** Fetch all models — used in the sidebar "Inference Model" selector. */
export async function getModelsForInference(): Promise<ModelSummary[]> {
  return apiClient<ModelSummary[]>('/models', { staleTime: 10000 });
}

/** GET /predictions — paginated list of past predictions for this user. */
export async function getPredictions(
  page = 1,
  pageSize = 20,
): Promise<PredictionResponse[]> {
  return apiClient<PredictionResponse[]>(
    `/predictions?page=${page}&page_size=${pageSize}`,
  );
}

/** GET /predictions/:id — single past prediction. */
export async function getPrediction(id: string): Promise<PredictionResponse> {
  return apiClient<PredictionResponse>(`/predictions/${id}`);
}

/** DELETE /predictions/:id — discard an unsaved prediction. */
export async function deletePrediction(id: string): Promise<MessageResponse> {
  return apiClient<MessageResponse>(`/predictions/${id}`, {
    method: 'DELETE',
  });
}

/**
 * GET /predictions?confirmed=false — list predictions that have not yet been
 * confirmed (saved to the library). Newest first, capped at pageSize.
 */
export async function getUnconfirmedPredictions(
  pageSize = 5,
): Promise<PredictionResponse[]> {
  return apiClient<PredictionResponse[]>(
    `/predictions?confirmed=false&page_size=${pageSize}`,
  );
}

/**
 * POST /predictions/:id/confirm — confirm a prediction and create an annotation.
 * Optionally attach biological metadata (species, age_days, treatment).
 */
export async function confirmPrediction(
  predictionId: string,
  body?: ConfirmPredictionRequest,
): Promise<ConfirmPredictionResponse> {
  return apiClient<ConfirmPredictionResponse>(
    `/predictions/${predictionId}/confirm`,
    {
      method: 'POST',
      body: body ?? null,
    },
  );
}

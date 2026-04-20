/**
 * api/inferenceApi.ts
 *
 * POST /predict and model helpers for the Inference Lab page.
 */

import { apiClient } from './client';
import type { PredictionResponse, ModelSummary } from '../types/api';

/**
 * POST /predict
 * Submits a root scan image for U-Net inference.
 * Do NOT set Content-Type manually — FormData sets the multipart boundary.
 */
export async function predict(
  file: File,
  confidenceThreshold: number,
): Promise<PredictionResponse> {
  const form = new FormData();
  form.append('file', file);
  form.append('confidence_threshold', confidenceThreshold.toFixed(2));

  return apiClient<PredictionResponse>('/predict', {
    method: 'POST',
    body: form,
  }).catch((err) => {
    throw new Error(err.message.includes('API Error') ? `Inference failed` : err.message);
  });
}

/** Fetch all models — used in the sidebar "Inference Model" selector. */
export async function getModelsForInference(): Promise<ModelSummary[]> {
  return apiClient<ModelSummary[]>('/models');
}

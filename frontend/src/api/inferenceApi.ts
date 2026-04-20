/**
 * api/inferenceApi.ts
 *
 * POST /predict and model helpers for the Inference Lab page.
 */

import { authHeader } from './auth';
import type { PredictionResponse, ModelSummary } from '../types/api';

const BASE_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:8000/api/v1';

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

  const res = await fetch(`${BASE_URL}/predict`, {
    method: 'POST',
    headers: { ...authHeader() },
    body: form,
  });

  if (!res.ok) {
    const data = (await res.json().catch(() => ({}))) as { message?: string };
    throw new Error(data.message ?? `Inference failed (${res.status})`);
  }

  return res.json() as Promise<PredictionResponse>;
}

/** Fetch all models — used in the sidebar "Inference Model" selector. */
export async function getModelsForInference(): Promise<ModelSummary[]> {
  const res = await fetch(`${BASE_URL}/models`, {
    headers: { ...authHeader() },
  });
  if (!res.ok) throw new Error(`Failed to load models (${res.status})`);
  return res.json() as Promise<ModelSummary[]>;
}

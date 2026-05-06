/**
 * api/libraryApi.ts
 *
 * API calls for the Library page — datasets, annotations.
 * Talks to GET /datasets and related endpoints.
 */

import { apiClient } from './client';

// ---------------------------------------------------------------------------
// Types (matching backend DatasetSummary / DatasetDetail schemas)
// ---------------------------------------------------------------------------

export interface DatasetSummary {
  dataset_id: string;
  name: string;
  created_at: string;
  modified_at: string;
  count: number;
  species?: string;
}

export interface DatasetDetail extends DatasetSummary {
  description?: string;
  species?: string;
}

// ---------------------------------------------------------------------------
// API calls
// ---------------------------------------------------------------------------

/** GET /datasets — list all datasets, optionally filtered. */
export async function getDatasets(params?: {
  species?: string;
  growth_stage?: string;
}): Promise<DatasetSummary[]> {
  const qs = new URLSearchParams();
  if (params?.species) qs.set('species', params.species);
  if (params?.growth_stage) qs.set('growth_stage', params.growth_stage);
  const query = qs.toString() ? `?${qs}` : '';
  return apiClient<DatasetSummary[]>(`/datasets${query}`);
}

/** GET /datasets/:id — single dataset detail. */
export async function getDatasetDetail(id: string): Promise<DatasetDetail> {
  return apiClient<DatasetDetail>(`/datasets/${id}`);
}

/** POST /datasets — create a new dataset (admin only). */
export async function createDataset(data: {
  name: string;
  description?: string;
  species?: string;
  growth_stage?: string;
}): Promise<DatasetDetail> {
  return apiClient<DatasetDetail>('/datasets', {
    method: 'POST',
    body: data,
  });
}

/** DELETE /datasets/:id — delete a dataset (admin only). */
export async function deleteDataset(id: string): Promise<void> {
  return apiClient<void>(`/datasets/${id}`, {
    method: 'DELETE',
  });
}

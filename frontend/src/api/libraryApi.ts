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

/**
 * POST /datasets/:id/images — upload images to a dataset.
 * NOTE: Currently mocked as backend is not implemented yet.
 */
export async function uploadDatasetImages(
  _datasetId: string,
  files: File[],
): Promise<{ success: number; failed: number }> {
  // Simulate network delay
  await new Promise((resolve) => setTimeout(resolve, 1500));

  // Since the backend endpoint doesn't exist yet, we'll mock a successful response.
  // In reality, this would use FormData to send files to the backend:
  /*
  const formData = new FormData();
  files.forEach(file => formData.append('files', file));
  return apiClient<{ success: number; failed: number }>(`/datasets/${datasetId}/images`, {
    method: 'POST',
    body: formData,
    // Note: apiClient automatically removes Content-Type for FormData
  });
  */

  return { success: files.length, failed: 0 };
}

export interface DatasetImportResponse {
  dataset_id: string;
  dataset_name: string;
  scans_imported: number;
  warnings: string[];
  errors: string[];
}

/**
 * POST /datasets/import — import a dataset from a ZIP archive.
 * NOTE: Mocked for now because backend endpoint does not exist.
 */
export async function importDataset(
  _zipFile: File,
  config: {
    targetMode: 'new' | 'existing';
    datasetId?: string;
    name?: string;
    description?: string;
    species?: string;
    growthStage?: string;
  },
): Promise<DatasetImportResponse> {
  // Simulate extraction & processing network delay
  await new Promise((resolve) => setTimeout(resolve, 3000));

  // Log usage to satisfy unused parameter TypeScript compile check
  console.log(
    `Importing dataset archive ${_zipFile.name} with config:`,
    config,
  );

  return apiClient<DatasetImportResponse>('/datasets/import', {
    method: 'POST',
  });
}

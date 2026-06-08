/**
 * api/libraryApi.ts
 *
 * API calls for the Library page — datasets, annotations.
 * Talks to GET /datasets and related endpoints.
 */

import { apiClient } from './client';
import type { AnnotationResponse, MessageResponse } from '../types/api';

// Re-export dataset types from the canonical location so consumers
// that import from libraryApi keep working without changes.
export type { DatasetSummary, DatasetDetail } from '../types/api';
import type { DatasetSummary, DatasetDetail } from '../types/api';

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
  return apiClient<DatasetSummary[]>(`/datasets${query}`, { staleTime: 10000 });
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

/** POST /datasets/:id/annotations/:annotationId */
export async function addAnnotationToDataset(
  datasetId: string,
  annotationId: string,
): Promise<AnnotationResponse> {
  return apiClient<AnnotationResponse>(
    `/datasets/${datasetId}/annotations/${annotationId}`,
    { method: 'POST' },
  );
}

/** DELETE /datasets/:id — delete a dataset (admin only). */
export async function deleteDataset(id: string): Promise<void> {
  return apiClient<void>(`/datasets/${id}`, {
    method: 'DELETE',
  });
}

// ---------------------------------------------------------------------------
// Annotations
// ---------------------------------------------------------------------------

/** GET /annotations — list all annotations, with optional filters and pagination. */
export async function getAnnotations(params?: {
  species?: string;
  growth_stage?: string;
  status?: string;
  dataset_id?: string;
  page?: number;
  page_size?: number;
  order?: 'asc' | 'desc';
}): Promise<AnnotationResponse[]> {
  const qs = new URLSearchParams();
  if (params?.species) qs.set('species', params.species);
  if (params?.growth_stage) qs.set('growth_stage', params.growth_stage);
  if (params?.status) qs.set('status', params.status);
  if (params?.dataset_id) qs.set('dataset_id', params.dataset_id);
  if (params?.page) qs.set('page', String(params.page));
  if (params?.page_size) qs.set('page_size', String(params.page_size));
  if (params?.order) qs.set('order', params.order);
  const query = qs.toString() ? `?${qs}` : '';
  return apiClient<AnnotationResponse[]>(`/annotations${query}`);
}

/** GET /annotations/:id — single annotation detail. */
export async function getAnnotationDetail(
  id: string,
): Promise<AnnotationResponse> {
  return apiClient<AnnotationResponse>(`/annotations/${id}`);
}

/** POST /annotations:id/run-inference - (re)run inference on an annotatino*/
export async function runAnnotationInference(
  annotationId: string,
): Promise<MessageResponse> {
  return apiClient<MessageResponse>(
    `/annotations/${annotationId}/run-inference`,
    {
      method: 'POST',
    },
  );
}

/** POST /annotations/:id/update — update annotation detail. */
export async function updateAnnotationMetadata(
  id: string,
  body: {
    species?: string;
    age_days?: number;
    treatment?: string;
    growth_stage?: string;
  },
): Promise<AnnotationResponse> {
  return apiClient<AnnotationResponse>(`/annotations/${id}/update`, {
    method: 'POST',
    body,
  });
}

/** POST /feedback **/
export async function submitAnnotationFeedback(
  predictionId: string,
  feedback_signal: 'correct' | 'incorrect',
): Promise<void> {
  await apiClient<MessageResponse>('/feedback', {
    method: 'POST',
    body: { prediction_id: predictionId, feedback_signal },
  });
}

/** DELETE /annotations/:id — delete an annotation. */
export async function deleteAnnotation(id: string): Promise<MessageResponse> {
  return apiClient<MessageResponse>(`/annotations/${id}`, {
    method: 'DELETE',
  });
}

/**
 * POST /datasets/:id/upload — upload multiple images to a dataset.
 * Uploads files with a concurrency limit to avoid overwhelming the
 * browser and server with simultaneous requests.
 */
const MAX_CONCURRENT_UPLOADS = 3;

export async function uploadDatasetImages(
  datasetId: string,
  files: File[],
): Promise<{ success: number; failed: number; errors: string[] }> {
  let success = 0;
  let failed = 0;
  const errors: string[] = [];

  async function uploadOne(file: File): Promise<void> {
    const formData = new FormData();
    formData.append('file', file);
    try {
      await apiClient<AnnotationResponse>(`/datasets/${datasetId}/upload`, {
        method: 'POST',
        body: formData,
      });
      success++;
    } catch (err) {
      failed++;
      const message = (err as Error).message || 'Unknown error';
      // Map API error messages to user-friendly ones
      let friendly = `"${file.name}": `;
      if (
        message.includes('413') ||
        message.toLowerCase().includes('too large')
      ) {
        friendly += 'file exceeds the 50MB limit.';
      } else if (
        message.includes('415') ||
        message.toLowerCase().includes('unsupported')
      ) {
        friendly += 'unsupported format. Use PNG, TIFF, or JPEG.';
      } else if (
        message.includes('409') ||
        message.toLowerCase().includes('identical') ||
        message.toLowerCase().includes('duplicate')
      ) {
        friendly += 'already exists in this dataset.';
      } else if (message.includes('404')) {
        friendly += 'dataset not found.';
      } else {
        friendly += message;
      }
      errors.push(friendly);
    }
  }

  // Process files with a concurrency pool of MAX_CONCURRENT_UPLOADS
  const queue = [...files];
  const workers = Array.from(
    { length: Math.min(MAX_CONCURRENT_UPLOADS, queue.length) },
    async () => {
      while (queue.length > 0) {
        const file = queue.shift()!;
        await uploadOne(file);
      }
    },
  );
  await Promise.all(workers);

  return { success, failed, errors };
}

/** POST /datasets/{id}/run-inference — trigger background inference on all pending images */
export async function runDatasetInference(
  datasetId: string,
): Promise<MessageResponse> {
  return apiClient<MessageResponse>(`/datasets/${datasetId}/run-inference`, {
    method: 'POST',
  });
}

export interface DatasetImportResponse {
  dataset_id: string;
  dataset_name: string;
  scans_imported: number;
  warnings: string[];
  errors: string[];
}

/** POST /datasets/import — import a dataset from a ZIP archive. */
export async function importDataset(
  zipFile: File,
  config: {
    targetMode: 'new' | 'existing';
    datasetId?: string;
    name?: string;
    description?: string;
    species?: string;
    growthStage?: string;
  },
): Promise<DatasetImportResponse> {
  const form = new FormData();
  form.append('file', zipFile);
  form.append('target_mode', config.targetMode);
  if (config.datasetId) form.append('dataset_id', config.datasetId);
  if (config.name) form.append('name', config.name);
  if (config.description) form.append('description', config.description);
  if (config.species) form.append('species', config.species);
  if (config.growthStage) form.append('growth_stage', config.growthStage);

  return apiClient<DatasetImportResponse>('/datasets/import', {
    method: 'POST',
    body: form, // do NOT set Content-Type; the browser sets the multipart boundary
  });
}

/**
 * api/labApi.ts
 *
 * API calls for the Lab page — annotation review queue.
 * Wraps existing annotation endpoints with Lab-specific semantics.
 */

import { getAnnotations } from './libraryApi';
import { apiClient } from './client';
import type { AnnotationResponse } from '../types/api';

/**
 * Fetch the review queue — all annotations with status `review_required`,
 * optionally scoped to a single dataset, in stable creation order.
 */
export async function getLabQueue(params?: {
  dataset_id?: string;
  page?: number;
  page_size?: number;
}): Promise<AnnotationResponse[]> {
  return getAnnotations({
    status: 'review_required',
    dataset_id: params?.dataset_id,
    page: params?.page,
    page_size: params?.page_size ?? 50,
    order: 'asc',
  });
}

/**
 * Submit a corrected mask for an annotation.
 * Sends the PNG blob as a multipart/form-data upload.
 * On success the annotation transitions to status `annotated`.
 */
export async function submitCorrectedMask(
  annotationId: string,
  maskBlob: Blob,
): Promise<AnnotationResponse> {
  const form = new FormData();
  form.append('mask', maskBlob, 'corrected_mask.png');

  return apiClient<AnnotationResponse>(`/annotations/${annotationId}/confirm`, {
    method: 'POST',
    body: form,
  });
}

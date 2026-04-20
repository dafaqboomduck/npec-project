/**
 * api/overview.ts
 *
 * Data-fetching functions for the Overview / Home page.
 *
 * Endpoints used:
 *   GET /overview/status       → lab summary stats
 *   GET /overview/activity     → recent activity feed
 *   GET /overview/model-health → active model metrics
 *   GET /overview/environment  → GPU / memory stats (admin)
 *   GET /models                → model list (for active-model card)
 */

import { authHeader } from './auth';
import type {
  OverviewStatusResponse,
  ActivityFeedResponse,
  ModelHealthResponse,
  EnvironmentResponse,
  ModelSummary,
} from '../types/api';

const BASE_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:8000/api/v1';

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    headers: { ...authHeader() },
  });
  if (!res.ok) {
    const body = (await res.json().catch(() => ({}))) as { message?: string };
    throw new Error(body.message ?? `GET ${path} failed (${res.status})`);
  }
  return res.json() as Promise<T>;
}

export const getOverviewStatus = () =>
  get<OverviewStatusResponse>('/overview/status');

export const getOverviewActivity = (page = 1, pageSize = 10) =>
  get<ActivityFeedResponse>(
    `/overview/activity?page=${page}&page_size=${pageSize}`,
  );

export const getModelHealth = () =>
  get<ModelHealthResponse>('/overview/model-health');

export const getEnvironment = () =>
  get<EnvironmentResponse>('/overview/environment');

/** Full model list — filter by status === 'deployed' for the active-model card. */
export const getModels = () => get<ModelSummary[]>('/models');

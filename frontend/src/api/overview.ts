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

import { apiClient } from './client';
import type {
  OverviewStatusResponse,
  ActivityFeedResponse,
  ModelHealthResponse,
  EnvironmentResponse,
  ModelSummary,
} from '../types/api';

export const getOverviewStatus = () =>
  apiClient<OverviewStatusResponse>('/overview/status');

export const getOverviewActivity = (page = 1, pageSize = 10) =>
  apiClient<ActivityFeedResponse>(
    `/overview/activity?page=${page}&page_size=${pageSize}`,
  );

export const getModelHealth = () =>
  apiClient<ModelHealthResponse>('/overview/model-health');

export const getEnvironment = () =>
  apiClient<EnvironmentResponse>('/overview/environment');

/** Full model list — filter by status === 'deployed' for the active-model card. */
export const getModels = () => apiClient<ModelSummary[]>('/models');

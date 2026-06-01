/**
 * api/dashboardApi.ts
 *
 * Frontend API integrations for the Dashboard Analytics page.
 * Connects to /dashboard/metrics, /dashboard/f1-history, and /dashboard/data-distribution.
 */

import { apiClient } from './client';
import type {
  DashboardMetricsResponse,
  F1HistoryResponse,
  DataDistributionResponse,
} from '../types/api';

/** GET /dashboard/metrics — Fetch top-level KPIs (F1 score, Throughput) */
export async function getDashboardMetrics(
  runId?: string,
): Promise<DashboardMetricsResponse> {
  const url = runId
    ? `/dashboard/metrics?run_id=${runId}`
    : '/dashboard/metrics';
  return apiClient<DashboardMetricsResponse>(url, { staleTime: 20000 });
}

/** GET /dashboard/f1-history — Fetch historical validation F1 data points per epoch */
export async function getF1History(runId?: string): Promise<F1HistoryResponse> {
  const url = runId
    ? `/dashboard/f1-history?run_id=${runId}`
    : '/dashboard/f1-history';
  return apiClient<F1HistoryResponse>(url, { staleTime: 30000 });
}

/** GET /dashboard/data-distribution — Fetch annotation counts grouped by plant species */
export async function getSpeciesDistribution(): Promise<DataDistributionResponse> {
  return apiClient<DataDistributionResponse>('/dashboard/data-distribution', {
    staleTime: 20000,
  });
}

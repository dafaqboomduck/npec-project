/* eslint-disable react-refresh/only-export-components */
/**
 * context/InferenceJobsContext.tsx
 *
 * Global context for tracking active inference jobs.
 * Lives at the app root so jobs persist across page navigation.
 * Also polls GET /inference/queue for real-time queue status.
 */

import { createContext, useContext, useState, useEffect } from 'react';
import { getInferenceQueue } from '../api/inferenceApi';
import { isAuthenticated } from '../api/auth';
import { BASE_URL } from '../api/client';
import type { InferenceQueueStatus } from '../types/api';

export type RunningJob = {
  id: string;
  filename: string;
  startedAt: number;
  controller: AbortController;
  /** 'image' = single-file inference; 'dataset' = batch dataset inference */
  type: 'image' | 'dataset';
  /** Set when type === 'dataset' */
  datasetId?: string;
  datasetName?: string;
};

type InferenceJobsContextType = {
  jobs: RunningJob[];
  addJob: (job: RunningJob) => void;
  removeJob: (id: string) => void;
  cancelJob: (id: string) => void;
  /** Returns true if a batch dataset job is currently active for the given dataset ID */
  isDatasetRunning: (datasetId: string) => boolean;
  queueStatus: InferenceQueueStatus;
  refreshQueue: () => Promise<void>;
};

const InferenceJobsContext = createContext<InferenceJobsContextType | null>(
  null,
);

export function InferenceJobsProvider({
  children,
}: {
  children: React.ReactNode;
}) {
  const [jobs, setJobs] = useState<RunningJob[]>(() => {
    try {
      const stored = localStorage.getItem('rootr_active_jobs');
      if (stored) {
        const parsed = JSON.parse(stored) as Omit<RunningJob, 'controller'>[];
        return parsed.map((j) => ({
          ...j,
          controller: new AbortController(),
        }));
      }
    } catch (e) {
      console.error('Failed to load active jobs from localStorage:', e);
    }
    return [];
  });

  const [queueStatus, setQueueStatus] = useState<InferenceQueueStatus>({
    queued: 0,
    running: 0,
    done: 0,
    failed: 0,
  });

  const setJobsAndStore = (
    newJobs: RunningJob[] | ((prev: RunningJob[]) => RunningJob[]),
  ) => {
    setJobs((prev) => {
      const next = typeof newJobs === 'function' ? newJobs(prev) : newJobs;
      try {
        const toStore = next.map((job) => {
          // eslint-disable-next-line @typescript-eslint/no-unused-vars
          const { controller, ...rest } = job;
          return rest;
        });
        localStorage.setItem('rootr_active_jobs', JSON.stringify(toStore));
      } catch (e) {
        console.error('Failed to save active jobs to localStorage:', e);
      }
      return next;
    });
  };

  function addJob(job: RunningJob) {
    setJobsAndStore((prev) => [...prev, job]);
    void refreshQueue();
  }

  function removeJob(id: string) {
    setJobsAndStore((prev) => prev.filter((j) => j.id !== id));
  }

  function cancelJob(id: string) {
    setJobsAndStore((prev) => {
      const job = prev.find((j) => j.id === id);
      if (job) job.controller.abort();
      return prev.filter((j) => j.id !== id);
    });
  }

  function isDatasetRunning(datasetId: string): boolean {
    return jobs.some((j) => j.type === 'dataset' && j.datasetId === datasetId);
  }

  // Synchronise token state with localStorage to allow SSE reconnection on login/logout
  const [token, setToken] = useState<string | null>(() =>
    localStorage.getItem('access_token'),
  );

  useEffect(() => {
    const interval = setInterval(() => {
      const currentToken = localStorage.getItem('access_token');
      if (currentToken !== token) {
        setToken(currentToken);
      }
    }, 1000);
    return () => clearInterval(interval);
  }, [token]);

  async function fetchQueue() {
    if (!isAuthenticated()) {
      setQueueStatus({ queued: 0, running: 0, done: 0, failed: 0 });
      return;
    }
    try {
      const status = await getInferenceQueue();
      setQueueStatus(status);
    } catch (err) {
      console.warn('Failed to fetch inference queue:', err);
    }
  }

  async function refreshQueue() {
    await fetchQueue();
  }

  useEffect(() => {
    if (!token) {
      setTimeout(() => {
        setQueueStatus({ queued: 0, running: 0, done: 0, failed: 0 });
      }, 0);
      return;
    }

    setTimeout(() => {
      void fetchQueue();
    }, 0);

    const path = `/inference/events?token=${encodeURIComponent(token)}`;
    const url = BASE_URL.startsWith('http')
      ? `${BASE_URL}${path}`
      : `${window.location.origin}${BASE_URL}${path}`;

    const eventSource = new EventSource(url);

    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === 'queue_updated') {
          void fetchQueue();
          window.dispatchEvent(new CustomEvent('inference_queue_updated'));
        }
      } catch (e) {
        console.error('Failed to parse SSE event:', e);
      }
    };

    eventSource.onerror = (err) => {
      console.warn('SSE connection status changed:', err);
    };

    return () => {
      eventSource.close();
    };
  }, [token]);

  return (
    <InferenceJobsContext.Provider
      value={{
        jobs,
        addJob,
        removeJob,
        cancelJob,
        isDatasetRunning,
        queueStatus,
        refreshQueue,
      }}
    >
      {children}
    </InferenceJobsContext.Provider>
  );
}

export function useInferenceJobs(): InferenceJobsContextType {
  const ctx = useContext(InferenceJobsContext);
  if (!ctx)
    throw new Error(
      'useInferenceJobs must be used inside InferenceJobsProvider',
    );
  return ctx;
}

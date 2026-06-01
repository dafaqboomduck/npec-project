/* eslint-disable react-refresh/only-export-components */
/**
 * context/InferenceJobsContext.tsx
 *
 * Global context for tracking active inference jobs.
 * Lives at the app root so jobs persist across page navigation.
 */

import { createContext, useContext, useState } from 'react';

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

  return (
    <InferenceJobsContext.Provider
      value={{ jobs, addJob, removeJob, cancelJob, isDatasetRunning }}
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

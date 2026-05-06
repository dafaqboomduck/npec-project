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
};

type InferenceJobsContextType = {
  jobs: RunningJob[];
  addJob: (job: RunningJob) => void;
  removeJob: (id: string) => void;
  cancelJob: (id: string) => void;
};

const InferenceJobsContext = createContext<InferenceJobsContextType | null>(null);

export function InferenceJobsProvider({ children }: { children: React.ReactNode }) {
  const [jobs, setJobs] = useState<RunningJob[]>([]);

  function addJob(job: RunningJob) {
    setJobs((prev) => [...prev, job]);
  }

  function removeJob(id: string) {
    setJobs((prev) => prev.filter((j) => j.id !== id));
  }

  function cancelJob(id: string) {
    setJobs((prev) => {
      const job = prev.find((j) => j.id === id);
      if (job) job.controller.abort();
      return prev.filter((j) => j.id !== id);
    });
  }

  return (
    <InferenceJobsContext.Provider value={{ jobs, addJob, removeJob, cancelJob }}>
      {children}
    </InferenceJobsContext.Provider>
  );
}

export function useInferenceJobs(): InferenceJobsContextType {
  const ctx = useContext(InferenceJobsContext);
  if (!ctx) throw new Error('useInferenceJobs must be used inside InferenceJobsProvider');
  return ctx;
}

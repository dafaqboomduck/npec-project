import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';

// When running inside Docker the container's "localhost" is the container
// itself, not the host machine. Set BACKEND_TARGET to override the proxy
// target (e.g. http://host.docker.internal:8000 for Docker Desktop on
// Windows/Mac, or http://172.17.0.1:8000 on Linux).
const backendTarget = process.env.BACKEND_TARGET ?? 'http://localhost:8000';

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => {
  // Load env file from the project root directory (two levels up from frontend)
  // Vite's loadEnv loads variables starting with VITE_ by default, but passing '' as third arg loads everything.
  const env = loadEnv(mode, '../', '');

  return {
    plugins: [react()],

    define: {
      'import.meta.env.VITE_DEPLOYMENT_ENV': JSON.stringify(
        env.DEPLOYMENT_ENV || process.env.DEPLOYMENT_ENV || 'local',
      ),
    },

    server: {
      port: 3000,
      // Proxy API and WebSocket requests to the Python backend during development.
      // This avoids CORS issues — the browser always talks to Vite (port 5173),
      // and Vite forwards /api/* and /ws/* to FastAPI.
      proxy: {
        '/api': {
          target: backendTarget,
          changeOrigin: true,
        },
        '/ws': {
          target: backendTarget.replace('http', 'ws'),
          ws: true,
        },
      },
    },

    build: {
      outDir: 'dist',
      sourcemap: true,
    },
  };
});

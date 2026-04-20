import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],

  server: {
    port: 5173,
    // Proxy API and WebSocket requests to the Python backend during development.
    // This avoids CORS issues — the browser always talks to Vite (port 5173),
    // and Vite forwards /api/* and /ws/* to FastAPI (port 8000).
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/ws': {
        target: 'ws://localhost:8000',
        ws: true,
      },
    },
  },

  build: {
    outDir: 'dist',
    sourcemap: true,
  },
});

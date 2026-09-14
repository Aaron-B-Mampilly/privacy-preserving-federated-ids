import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'node:path';

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: { '@': path.resolve(__dirname, './src') },
  },
  server: {
    port: 5173,
    strictPort: false,
  },
  build: {
    rollupOptions: {
      output: {
        // Recharts is by far the heaviest dependency and is only needed
        // on chart-bearing screens; splitting it keeps the initial
        // dashboard payload small and lets the browser cache it apart
        // from application code.
        manualChunks: {
          charts: ['recharts'],
          router: ['react-router-dom'],
          vendor: ['react', 'react-dom', 'axios'],
        },
      },
    },
    chunkSizeWarningLimit: 700,
  },
});

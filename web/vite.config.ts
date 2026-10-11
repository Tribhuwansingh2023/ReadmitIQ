import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
const backendTarget = process.env.BACKEND_URL || 'http://127.0.0.1:8000'

export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    port: 5173,
    proxy: {
      '/v1': {
        target: backendTarget,
        changeOrigin: true,
      },
      '/health': {
        target: backendTarget,
        changeOrigin: true,
      },
      '/predict': {
        target: backendTarget,
        changeOrigin: true,
      },
      '/explain': {
        target: backendTarget,
        changeOrigin: true,
      },
      '/similar': {
        target: backendTarget,
        changeOrigin: true,
      },
      '/cohort': {
        target: backendTarget,
        changeOrigin: true,
      },
      '/model-info': {
        target: backendTarget,
        changeOrigin: true,
      },
      '/api': {
        target: backendTarget,
        changeOrigin: true,
      },
    },
  },
})

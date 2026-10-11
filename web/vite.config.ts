import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    port: 5173,
    proxy: {
      '/v1': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
      '/health': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
      '/predict': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
      '/explain': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
      '/similar': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
      '/cohort': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
})

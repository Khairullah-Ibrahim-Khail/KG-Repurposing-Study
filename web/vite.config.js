import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Static SPA. Vitest runs under jsdom with a jest-dom setup file.
export default defineConfig({
  plugins: [react()],
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: ['./src/__tests__/setup.js'],
  },
})

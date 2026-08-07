import { defineConfig } from 'vite';

export default defineConfig({
  server: {
    port: 5173,
    proxy: {
      '/upload-single': 'http://127.0.0.1:8005',
      '/review': 'http://127.0.0.1:8005'
    }
  }
});

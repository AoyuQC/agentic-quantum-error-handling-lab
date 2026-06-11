import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// During `npm run dev` the API is proxied to the FastAPI backend on :8000.
// `npm run build` emits static assets the backend serves from ui/dist.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 3099,
    proxy: {
      "/api": "http://localhost:8000",
    },
  },
  build: {
    outDir: "dist",
  },
});

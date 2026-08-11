import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// The dev server proxies /api to uvicorn so the app and the FastAPI backend
// look like one origin -- no CORS in the browser, same URLs in production.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: { "/api": "http://localhost:8000" },
  },
});

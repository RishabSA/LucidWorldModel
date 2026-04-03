import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  // No SPA fallback, so a missing model is a 404 rather than index.html with status 200
  appType: "mpa",
});

import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  base: "/",
  build: {
    outDir: "../src/terraintrace/static",
    emptyOutDir: true,
  },
  server: {
    host: "127.0.0.1",
    port: 5266,
    strictPort: true,
    proxy: {
      "/api/v1": "http://127.0.0.1:5206",
    },
  },
});

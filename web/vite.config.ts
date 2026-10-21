import { defineConfig } from "vitest/config";
import { VitePWA } from "vite-plugin-pwa";

const base = process.env.ALPHASHO_BASE ?? "/alphasho/";
export default defineConfig({
  base,
  plugins: [
    VitePWA({
      strategies: "injectManifest",
      srcDir: "src",
      filename: "sw.ts",
      registerType: "prompt",
      includeAssets: ["pieces/*.png"],
      manifest: {
        name: "AlphaSho — Browser Shogi",
        short_name: "AlphaSho",
        description: "Play shogi locally, against a CPU player, or with a friend.",
        theme_color: "#17130f",
        background_color: "#f2eadb",
        display: "standalone",
        start_url: base,
        scope: base,
        icons: [
          { src: "pieces/black_king.png", sizes: "any", type: "image/png", purpose: "any" }
        ]
      },
      injectManifest: {
        rollupFormat: "iife",
        globPatterns: ["**/*.{js,css,html,png,woff2}"],
        globIgnores: ["nnue/**"]
      }
    })
  ],
  worker: { format: "es" },
  build: { rollupOptions: { input: { main: "index.html", benchmark: "benchmark.html" } } },
  test: {
    environment: "node",
    include: ["tests/**/*.test.ts"],
    coverage: { reporter: ["text", "html"] }
  }
});

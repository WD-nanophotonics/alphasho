import { defineConfig } from "vitest/config";
import { VitePWA } from "vite-plugin-pwa";

export default defineConfig({
  base: "/alphasho/",
  plugins: [
    VitePWA({
      registerType: "prompt",
      includeAssets: ["pieces/*.png"],
      manifest: {
        name: "AlphaSho — Browser Shogi",
        short_name: "AlphaSho",
        description: "Play shogi locally, against a CPU player, or with a friend.",
        theme_color: "#17130f",
        background_color: "#f2eadb",
        display: "standalone",
        start_url: "/alphasho/",
        scope: "/alphasho/",
        icons: [
          { src: "pieces/black_king.png", sizes: "any", type: "image/png", purpose: "any" }
        ]
      },
      workbox: {
        globPatterns: ["**/*.{js,css,html,png,woff2}"],
        navigateFallback: "index.html"
      }
    })
  ],
  worker: { format: "es" },
  test: {
    environment: "node",
    include: ["tests/**/*.test.ts"],
    coverage: { reporter: ["text", "html"] }
  }
});

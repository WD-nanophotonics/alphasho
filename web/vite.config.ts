import { defineConfig } from "vitest/config";
import { VitePWA } from "vite-plugin-pwa";
import { readFileSync, readdirSync } from 'node:fs';
import { createHash } from 'node:crypto';

const base = process.env.ALPHASHO_BASE ?? "/alphasho/";
const policyHash = createHash('sha256');
for (const name of readdirSync(new URL('./src/', import.meta.url), {recursive:true}).map(p=>String(p).replace(/\\/g,'/')).filter(p=>/\.(ts|css)$/.test(p)).sort()) policyHash.update(name + '\n' + readFileSync(new URL('./src/' + name, import.meta.url), 'utf8').replace(/\r\n/g, '\n'));
export default defineConfig({
  base,
  define: { __NNUE_POLICY_HASH__: JSON.stringify(policyHash.digest('hex')), __NNUE_ENGINE_SHA__: JSON.stringify(JSON.parse(readFileSync(new URL('./public/nnue/engine/build.json', import.meta.url), 'utf8')).files['yaneuraou.wasm']), __NNUE_ENGINE_JS_SHA__: JSON.stringify(JSON.parse(readFileSync(new URL('./public/nnue/engine/build.json', import.meta.url), 'utf8')).files['yaneuraou.js']) },
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

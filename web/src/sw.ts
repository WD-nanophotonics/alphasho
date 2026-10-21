/// <reference lib="webworker" />
export {};
declare const self: ServiceWorkerGlobalScope & { __WB_MANIFEST: Array<{ url: string; revision?: string }> };
declare const __NNUE_ENGINE_SHA__: string;
declare const __NNUE_ENGINE_JS_SHA__: string;
const sw = self;
const manifest = self.__WB_MANIFEST;
const signature = JSON.stringify(manifest); let hash = 2166136261;
for (const c of signature) hash = Math.imul(hash ^ c.charCodeAt(0), 16777619);
const scope = new URL(sw.registration.scope);
const cachePrefix = 'alphasho-shell-nn-v1:' + scope.pathname + ':';
const cacheName = cachePrefix + (hash >>> 0).toString(16);
const nnuePrefix = scope.pathname + 'nnue/';
const assetCacheName = 'alphasho-nnue-assets-v1:' + scope.pathname + ':' + __NNUE_ENGINE_SHA__.slice(0,12) + ':' + __NNUE_ENGINE_JS_SHA__.slice(0,12);
async function validEngine(request: Request, response: Response): Promise<boolean> {
  const pathname = new URL(request.url).pathname;
  const expected = pathname.endsWith('/engine/yaneuraou.wasm') ? __NNUE_ENGINE_SHA__ : pathname.endsWith('/engine/yaneuraou.js') ? __NNUE_ENGINE_JS_SHA__ : null;
  if (!expected) return true;
  const digest = [...new Uint8Array(await crypto.subtle.digest('SHA-256', await response.clone().arrayBuffer()))].map(v => v.toString(16).padStart(2,'0')).join('');
  return digest === expected;
}
function isolated(response: Response): Response {
  if (response.status === 0) return response;
  const headers = new Headers(response.headers);
  headers.set('Cross-Origin-Opener-Policy', 'same-origin');
  headers.set('Cross-Origin-Embedder-Policy', 'require-corp');
  headers.set('Cross-Origin-Resource-Policy', 'same-origin');
  return new Response(response.body, { status: response.status, statusText: response.statusText, headers });
}
sw.addEventListener('install', event => { event.waitUntil(caches.open(cacheName).then(cache => cache.addAll([...new Set(manifest.map(x => new URL(x.url, scope).href))]))); });
sw.addEventListener('activate', event => { event.waitUntil((async () => {
  const assets = await caches.open(assetCacheName);
  for (const key of await caches.keys()) {
    if (key === cacheName || !key.startsWith('alphasho-shell-')) continue;
    const old = await caches.open(key), requests = await old.keys();
    const own = key.startsWith(cachePrefix) || (requests.length > 0 && requests.every(r => { const u = new URL(r.url); return u.origin === scope.origin && u.pathname.startsWith(scope.pathname); }));
    if (!own) continue;
    for (const request of requests) {
      const u = new URL(request.url);
      if (!u.pathname.startsWith(nnuePrefix) || u.pathname.endsWith('/nn.bin')) continue;
      const response = await old.match(request);
      if (response && await validEngine(request, response)) {
        await assets.put(request, response.clone());
        const versioned = new URL(request.url);
        const expected = versioned.pathname.endsWith('/engine/yaneuraou.wasm') ? __NNUE_ENGINE_SHA__ : versioned.pathname.endsWith('/engine/yaneuraou.js') ? __NNUE_ENGINE_JS_SHA__ : null;
        if (expected) { versioned.searchParams.set('sha256', expected); await assets.put(versioned.href, response); }
      }
    }
    await caches.delete(key);
  }
  await sw.clients.claim();
})()); });
sw.addEventListener('message', event => { if (event.data?.type === 'SKIP_WAITING') void sw.skipWaiting(); });
sw.addEventListener('fetch', event => {
  const request = event.request, url = new URL(request.url);
  if (request.method !== 'GET' || url.origin !== scope.origin || !url.pathname.startsWith(scope.pathname)) return;
  event.respondWith((async () => {
    // Models have their own hash-keyed, verified cache; never duplicate the 64 MB file here.
    if (url.pathname.endsWith('/nn.bin')) return isolated(await fetch(request));
    const cache = await caches.open(url.pathname.startsWith(nnuePrefix) ? assetCacheName : cacheName);
    const expected = url.pathname.endsWith('/engine/yaneuraou.wasm') ? __NNUE_ENGINE_SHA__ : url.pathname.endsWith('/engine/yaneuraou.js') ? __NNUE_ENGINE_JS_SHA__ : null;
    const versioned = new URL(url); if (expected) versioned.searchParams.set('sha256', expected);
    try {
      // Also versions Emscripten's nested worker URL, which otherwise drops
      // the importer's query string. A stale HTTP cache cannot replace it.
      const response = expected ? await fetch(versioned.href, {cache:'no-cache'}) : await fetch(request);
      if (expected && response.ok && !await validEngine(request,response)) throw new Error('NNUE engine checksum mismatch');
      if (response.ok) { await cache.put(request, response.clone()); if (expected) await cache.put(versioned.href,response.clone()); }
      return isolated(response);
    } catch {
      const response = await cache.match(request) ?? (expected ? await cache.match(versioned.href) : undefined) ?? (request.mode === 'navigate' ? await cache.match(new URL('index.html', scope)) : undefined);
      if (!response || !await validEngine(request,response)) return new Response('Offline resource not cached', { status: 503 });
      return isolated(response);
    }
  })());
});

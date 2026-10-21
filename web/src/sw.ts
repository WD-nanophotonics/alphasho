/// <reference lib="webworker" />
export {};
declare const self: ServiceWorkerGlobalScope & { __WB_MANIFEST: Array<{ url: string; revision?: string }> };
const sw = self;
const manifest = self.__WB_MANIFEST;
const signature = JSON.stringify(manifest); let hash = 2166136261;
for (const c of signature) hash = Math.imul(hash ^ c.charCodeAt(0), 16777619);
const cacheName = 'alphasho-shell-nn-v1-' + (hash >>> 0).toString(16);
const scope = new URL(sw.registration.scope);
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
  for (const key of await caches.keys()) if (key.startsWith('alphasho-shell-') && key !== cacheName) await caches.delete(key);
  await sw.clients.claim();
})()); });
sw.addEventListener('message', event => { if (event.data?.type === 'SKIP_WAITING') void sw.skipWaiting(); });
sw.addEventListener('fetch', event => {
  const request = event.request, url = new URL(request.url);
  if (request.method !== 'GET' || url.origin !== scope.origin || !url.pathname.startsWith(scope.pathname)) return;
  event.respondWith((async () => {
    // Models have their own hash-keyed, verified cache; never duplicate the 64 MB file here.
    if (url.pathname.endsWith('/nn.bin')) return isolated(await fetch(request));
    const cache = await caches.open(cacheName);
    try {
      const response = await fetch(request);
      if (response.ok) await cache.put(request, response.clone());
      return isolated(response);
    } catch {
      const response = await cache.match(request) ?? (request.mode === 'navigate' ? await cache.match(new URL('index.html', scope)) : undefined);
      if (!response) return new Response('Offline resource not cached', { status: 503 });
      return isolated(response);
    }
  })());
});

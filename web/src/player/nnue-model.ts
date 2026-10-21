import type { NnueManifest } from './nnue-config';
const base = import.meta.env.BASE_URL;
export async function loadModel(signal: AbortSignal, progress: (fraction: number) => void): Promise<{ manifest: NnueManifest; bytes: ArrayBuffer }> {
  const manifest: NnueManifest = await fetch(base + 'nnue/model.json', { signal, cache: 'no-cache' }).then(r => { if (!r.ok) throw new Error('Model manifest unavailable'); return r.json(); });
  if (manifest.schemaVersion !== 1 || manifest.format !== 'yaneuraou-halfkp-256x2-32-32' || !/^[a-f0-9]{64}$/.test(manifest.sha256)) throw new Error('Unsupported NNUE model');
  const cache = await caches.open('alphasho-nnue-models-v1');
  const url = new URL(base + manifest.modelUrl, location.origin);
  const key = url.href + '?sha256=' + manifest.sha256;
  let response = await cache.match(key); const cached = !!response;
  if (!response) response = await fetch(url, { signal });
  if (!response.ok || !response.body) throw new Error('NNUE download failed');
  const reader = response.body.getReader(); const chunks: Uint8Array[] = []; let size = 0;
  try {
    for (;;) { signal.throwIfAborted(); const { value, done } = await reader.read(); if (done) break; size += value.length; if (size > manifest.bytes) throw new Error('NNUE size mismatch'); chunks.push(value); progress(size / manifest.bytes); }
  } finally { reader.releaseLock(); }
  if (size !== manifest.bytes) throw new Error('NNUE download incomplete');
  const bytes = new Uint8Array(size); let offset = 0;
  for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.length; }
  chunks.length = 0;
  const digest = [...new Uint8Array(await crypto.subtle.digest('SHA-256', bytes))].map(v => v.toString(16).padStart(2, '0')).join('');
  if (digest !== manifest.sha256) { await cache.delete(key); throw new Error('NNUE checksum mismatch'); }
  if (!cached) await cache.put(key, new Response(bytes, { headers: { 'Content-Type': 'application/octet-stream' } }));
  progress(1); return { manifest, bytes: bytes.buffer };
}

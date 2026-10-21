import { afterEach, describe, expect, it, vi } from 'vitest';
import { loadModel } from '../src/player/nnue-model';
const bytes=new Uint8Array([1,2,3,4,5,6,7,8]);
async function setup(corrupt=false){
 const sha=[...new Uint8Array(await crypto.subtle.digest('SHA-256',bytes))].map(v=>v.toString(16).padStart(2,'0')).join('');
 const manifest={schemaVersion:1,format:'yaneuraou-halfkp-256x2-32-32',sha256:sha,bytes:bytes.length,modelUrl:'nnue/nn.bin'};
 const store=new Map<string,Response>();let networkModels=0;
 const cache={match:vi.fn(async(key:string)=>store.get(key)?.clone()),put:vi.fn(async(key:string,response:Response)=>{store.set(key,response.clone());}),delete:vi.fn(async(key:string)=>store.delete(key))};
 vi.stubGlobal('location',{origin:'https://example.test'});vi.stubGlobal('caches',{open:async()=>cache});
 vi.stubGlobal('fetch',vi.fn(async(url:unknown)=>{if(String(url).endsWith('model.json'))return new Response(JSON.stringify(manifest));networkModels++;return new Response(corrupt?new Uint8Array([8,7,6,5,4,3,2,1]):bytes);}));
 return {cache,store,modelRequests:()=>networkModels};
}
afterEach(()=>vi.unstubAllGlobals());
describe('verified model cache',()=>{
 it('downloads once, verifies bytes and reuses the hash-keyed cache',async()=>{
  const test=await setup();const progress:number[]=[];const first=await loadModel(new AbortController().signal,p=>progress.push(p));
  expect(new Uint8Array(first.bytes)).toEqual(bytes);expect(progress.at(-1)).toBe(1);
  await loadModel(new AbortController().signal,()=>{});expect(test.modelRequests()).toBe(1);expect(test.cache.put).toHaveBeenCalledOnce();
 });
 it('rejects checksum corruption without caching invalid weights',async()=>{
  const test=await setup(true);await expect(loadModel(new AbortController().signal,()=>{})).rejects.toThrow(/checksum/);expect(test.store.size).toBe(0);expect(test.cache.delete).toHaveBeenCalledOnce();
 });
 it('surfaces storage exhaustion instead of pretending the model is ready',async()=>{
  const test=await setup();test.cache.put.mockRejectedValue(new DOMException('Storage full','QuotaExceededError'));
  await expect(loadModel(new AbortController().signal,()=>{})).rejects.toThrow(/Storage full/);
 });
});

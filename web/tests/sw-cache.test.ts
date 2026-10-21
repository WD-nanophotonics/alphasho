import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { runInNewContext } from 'node:vm';
import ts from 'typescript';
const scope='https://example.test/preview/';
const sha=(text:string)=>createHash('sha256').update(text).digest('hex');
class Cache {
 data=new Map<string,Response>();
 key(r:Request|string){return typeof r==='string'?r:r.url;}
 async put(r:Request|string,response:Response){this.data.set(this.key(r),response.clone());}
 async match(r:Request|string){return this.data.get(this.key(r))?.clone();}
 async keys(){return [...this.data.keys()].map(url=>new Request(url));}
 async addAll(urls:string[]){for(const u of urls)await this.put(u,new Response('shell'));}
}
function storage(){const data=new Map<string,Cache>();return{data,async open(name:string){let cache=data.get(name);if(!cache){cache=new Cache();data.set(name,cache);}return cache;},async keys(){return [...data.keys()];},async delete(name:string){return data.delete(name);}};}
function service(caches:ReturnType<typeof storage>,revision='1'){
 const handlers=new Map<string,(e:any)=>void>();
 const source=ts.transpileModule(readFileSync(new URL('../src/sw.ts',import.meta.url),'utf8'),{compilerOptions:{module:ts.ModuleKind.ES2022,target:ts.ScriptTarget.ES2022}}).outputText.replace(/export\s*\{\s*\};?/g,'');
 runInNewContext(source,{self:{registration:{scope},__WB_MANIFEST:[{url:'index.html',revision}],clients:{claim:async()=>{}},addEventListener:(name:string,fn:(e:any)=>void)=>handlers.set(name,fn)},caches,crypto,URL,Headers,Response,Uint8Array,fetch:async()=>{throw Error('Offline');},__NNUE_ENGINE_SHA__:sha('wasm'),__NNUE_ENGINE_JS_SHA__:sha('javascript')});
 return{async activate(){let done:Promise<void>|undefined;handlers.get('activate')!({waitUntil:(p:Promise<void>)=>{done=p;}});await done;},async fetch(url:string){let response:Promise<Response>|undefined;handlers.get('fetch')!({request:new Request(url),respondWith:(p:Promise<Response>)=>{response=p;}});return await response!;}};
}
describe('scope-safe offline NN cache upgrades',()=>{
 it('migrates verified engine assets while preserving another app cache',async()=>{
  const caches=storage(),legacy=await caches.open('alphasho-shell-old'),other=await caches.open('alphasho-shell-nn-v1:/production/:old');
  await legacy.put(scope+'index.html',new Response('old'));await legacy.put(scope+'nnue/engine/yaneuraou.js',new Response('javascript'));await legacy.put(scope+'nnue/engine/yaneuraou.wasm',new Response('wasm'));await other.put('https://example.test/production/index.html',new Response('production'));
  const sw=service(caches);await sw.activate();expect(caches.data.has('alphasho-shell-old')).toBe(false);expect(caches.data.has('alphasho-shell-nn-v1:/production/:old')).toBe(true);
  const response=await sw.fetch(scope+'nnue/engine/yaneuraou.js');expect(await response.text()).toBe('javascript');expect(response.headers.get('Cross-Origin-Embedder-Policy')).toBe('require-corp');
  expect(await(await sw.fetch(scope+'nnue/engine/yaneuraou.js?sha256='+sha('javascript'))).text()).toBe('javascript');
 });
 it('does not migrate a corrupt engine from a legacy cache',async()=>{
  const caches=storage(),legacy=await caches.open('alphasho-shell-old');await legacy.put(scope+'nnue/engine/yaneuraou.js',new Response('corrupt'));
  const sw=service(caches);await sw.activate();expect((await sw.fetch(scope+'nnue/engine/yaneuraou.js')).status).toBe(503);
 });
 it('preserves the frozen engine across a second shell revision while offline',async()=>{
  const caches=storage(),legacy=await caches.open('alphasho-shell-old');await legacy.put(scope+'nnue/engine/yaneuraou.wasm',new Response('wasm'));
  await service(caches,'1').activate();const next=service(caches,'2');await next.activate();expect(await(await next.fetch(scope+'nnue/engine/yaneuraou.wasm')).text()).toBe('wasm');
 });
});

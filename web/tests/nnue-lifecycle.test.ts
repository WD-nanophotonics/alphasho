import { beforeEach, afterEach, describe, expect, it, vi } from 'vitest';
import { Position } from '../src/engine/position';
vi.mock('../src/player/nnue-model', () => ({ loadModel: async () => ({ manifest: { round:7200 }, bytes:new ArrayBuffer(8) }) }));
import { NnueWorkerPlayer } from '../src/player/nnue-player';
class FakeWorker {
  static instances: FakeWorker[]=[];
  onmessage: ((event:{data:unknown})=>void)|null=null; onerror:((e:{message:string})=>void)|null=null;
  terminated=false; calls:Array<{type:string;id:number}>=[];
  constructor(){FakeWorker.instances.push(this);}
  postMessage(data:{type:string;id:number}){this.calls.push(data);if(data.type==='init')queueMicrotask(()=>this.reply({type:'ready',id:data.id}));}
  reply(data:unknown){this.onmessage?.({data});}
  terminate(){this.terminated=true;}
}
const state=()=>({schemaVersion:1 as const,sfen:new Position().sfen(),history:[],turn:0 as const,outcome:null,lastMove:null});
const flush=async()=>{for(let i=0;i<12;i++)await Promise.resolve();};
describe('NN worker lifecycle',()=>{
 beforeEach(()=>{FakeWorker.instances=[];vi.stubGlobal('Worker',FakeWorker);vi.stubGlobal('crossOriginIsolated',true);vi.stubGlobal('location',{origin:'https://example.test'});});
 afterEach(()=>{vi.unstubAllGlobals();vi.useRealTimers();});
 it('reuses the initialized worker and ignores replies belonging to old requests',async()=>{
  const player=new NnueWorkerPlayer('ibisha'); const abort=new AbortController();
  const first=player.chooseMove(state(),{milliseconds:500,maxDepth:6,guidance:false},abort.signal);await flush();
  const worker=FakeWorker.instances[0]!,id=worker.calls.at(-1)!.id;
  worker.reply({type:'result',id:id-1,result:{move:'bad'}});
  worker.reply({type:'result',id,result:{move:'7g7f',nodes:100,depth:2,elapsedMs:1}});expect((await first).move).toBe('7g7f');
  const second=player.chooseMove(state(),{milliseconds:500,maxDepth:6,guidance:false},abort.signal);await flush();
  expect(FakeWorker.instances).toHaveLength(1);worker.reply({type:'result',id:worker.calls.at(-1)!.id,result:{move:'2g2f',nodes:100,depth:2,elapsedMs:1}});expect((await second).move).toBe('2g2f');player.dispose();
 });
 it('terminates cancelled search and never accepts its late move',async()=>{
  const player=new NnueWorkerPlayer(),abort=new AbortController();const pending=player.chooseMove(state(),{milliseconds:500,maxDepth:6},abort.signal);const rejection=expect(pending).rejects.toThrow(/cancel/i);await flush();
  const worker=FakeWorker.instances[0]!;abort.abort();await rejection;expect(worker.terminated).toBe(true);worker.reply({type:'result',id:worker.calls.at(-1)!.id,result:{move:'7g7f'}});
 });
 it('rejects illegal output and allows a fresh retry',async()=>{
  const player=new NnueWorkerPlayer(),abort=new AbortController();const pending=player.chooseMove(state(),{milliseconds:500,maxDepth:6},abort.signal);const rejection=expect(pending).rejects.toThrow(/illegal/);await flush();
  const worker=FakeWorker.instances[0]!;worker.reply({type:'result',id:worker.calls.at(-1)!.id,result:{move:'7g7a',nodes:0,depth:1,elapsedMs:1}});await rejection;expect(worker.terminated).toBe(true);
  const retry=player.chooseMove(state(),{milliseconds:500,maxDepth:6},abort.signal);const rejectedRetry=expect(retry).rejects.toThrow();await flush();expect(FakeWorker.instances).toHaveLength(2);player.dispose();await rejectedRetry;
 });
 it('terminates a worker that misses the hard response deadline',async()=>{
  vi.useFakeTimers();const player=new NnueWorkerPlayer(),abort=new AbortController();const pending=player.chooseMove(state(),{milliseconds:100,maxDepth:1},abort.signal);const rejection=expect(pending).rejects.toThrow(/timeout/);await flush();await vi.advanceTimersByTimeAsync(400);await rejection;expect(FakeWorker.instances[0]!.terminated).toBe(true);
 });
});

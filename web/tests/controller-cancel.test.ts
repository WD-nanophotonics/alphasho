import { beforeEach, afterEach, describe, expect, it, vi } from 'vitest';
const pending=vi.hoisted(()=>[] as Array<{resolve:(r:unknown)=>void;signal:AbortSignal}>);
vi.mock('../src/player/heuristic-player',()=>({DIFFICULTIES:{normal:{milliseconds:500,maxDepth:6}},HeuristicWorkerPlayer:class{
 dispose(){} chooseMove(_s:unknown,_b:unknown,signal:AbortSignal){return new Promise((resolve,reject)=>{pending.push({resolve,signal});signal.addEventListener('abort',()=>reject(new DOMException('Cancelled','AbortError')),{once:true});});}
}}));
import { GameController } from '../src/game-controller';
const flush=async()=>{for(let i=0;i<8;i++)await Promise.resolve();};
describe('controller cancellation and turn ownership',()=>{
 beforeEach(()=>{pending.length=0;vi.stubGlobal('localStorage',{setItem:vi.fn()});});
 afterEach(()=>vi.unstubAllGlobals());
 it('undoes the human move during CPU thinking and discards the late result',async()=>{
  const c=new GameController();c.start('cpu',1);pending[0]!.resolve({move:'7g7f',elapsedMs:1,nodes:1,depth:1});await flush();
  c.play(c.state.position.parseUsi('3c3d'));expect(c.state.thinking).toBe(true);c.undo();await flush();
  expect(c.state.history).toEqual(['7g7f']);expect(c.canHumanMove()).toBe(true);expect(pending[1]!.signal.aborted).toBe(true);
  pending[1]!.resolve({move:'2h6h'});await flush();expect(c.state.history).toEqual(['7g7f']);c.dispose();
 });
 it('undoes an opening human move without leaving an unplayable CPU turn',async()=>{
  const c=new GameController();c.start('cpu',0);c.play(c.state.position.parseUsi('7g7f'));c.undo();await flush();expect(c.state.history).toEqual([]);expect(c.canHumanMove()).toBe(true);c.dispose();
 });
 it('restarts without allowing a previous game result to enter the new game',async()=>{
  const c=new GameController();c.start('cpu',1);c.start('cpu',0);pending[0]!.resolve({move:'7g7f'});await flush();expect(c.state.history).toEqual([]);expect(c.canHumanMove()).toBe(true);c.dispose();
 });
 it('marks a restored CPU turn paused so explicit resume remains available',()=>{
  const c=new GameController();c.load(c.state.position.sfen(),[],'cpu',1,null);expect(c.paused).toBe(true);expect(c.state.thinking).toBe(false);c.dispose();
 });
});

import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';
import { WasmEngine, webRoot } from './nnue-runtime';
import { Position } from '../src/engine/position';
const fixtures=JSON.parse(fs.readFileSync(path.join(webRoot,'public/nnue/parity.json'),'utf8')) as Array<{sfen:string}>;
const engine=new WasmEngine(),results:unknown[]=[];
try{
 await engine.init();
 for(const piece of ['P','L','N','S','G','B','R']){
  const fixture=fixtures.find(f=>new Position(f.sfen).legalUsiMoves().some(m=>m.startsWith(piece+'*')));
  if(!fixture)throw Error('No drop regression fixture '+piece);
  const drop=new Position(fixture.sfen).legalUsiMoves().find(m=>m.startsWith(piece+'*'))!;
  const result=await engine.search('position sfen '+fixture.sfen,10,[drop],Math.random,{milliseconds:1000,nodes:1000});
  assert.equal(result.move,drop);results.push({piece,sfen:fixture.sfen,drop,...result});
 }
 fs.writeFileSync(path.join(webRoot,'nnue-reports/drop-regression.json'),JSON.stringify({passed:true,results},null,2));console.log('All 7 uppercase drop-piece searchmoves passed');
}finally{engine.close();}

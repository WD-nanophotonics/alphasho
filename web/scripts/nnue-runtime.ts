// Offline test harness for the exact shipped WASM and shared difficulty policy.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { NNUE_LEVELS, parseInfo, selectCandidate, type Candidate } from '../src/player/nnue-config';
export const webRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
export class WasmEngine {
  module: any; consume: ((line: string) => void) | null = null; recent: string[] = [];
  async init() {
    const base = path.join(webRoot, 'public/nnue');
    // Emscripten generated module has no declaration file.
    // @ts-expect-error generated module
    const { default: factory } = await import('../public/nnue/engine/yaneuraou.js');
    const output = (line: string) => { this.recent.push(line); if(this.recent.length>20)this.recent.shift(); this.consume?.(line); };
    this.module = await factory({ print: output, printErr: output, wasmBinary: fs.readFileSync(path.join(base,'engine/yaneuraou.wasm')), locateFile: (f: string) => path.join(base,'engine',f) });
    this.module.addMessageListener(output); this.module.FS.mkdir('/eval'); this.module.FS.writeFile('/eval/nn.bin',fs.readFileSync(path.join(base,'nn.bin')));
    await this.command(['usi'],'usiok');
    await this.command(Object.entries({ USI_Hash:16,Threads:1,MultiPV:1,EvalDir:'/eval',FV_SCALE:16,BookFile:'no_book',USI_Ponder:false,NetworkDelay:0,NetworkDelay2:0,MinimumThinkingTime:1000,ResignValue:99999,PvInterval:0,GenerateAllLegalMoves:true,EnteringKingRule:'NoEnteringKing' }).map(([k,v])=>`setoption name ${k} value ${v}`).concat('isready'),'readyok');
  }
  send(c: string) { this.module.ccall('alphasho_usi_command','number',['string'],[c]); }
  command(commands: string[], prefix: string, timeout=30000, observe?: (s:string)=>void): Promise<string> {
    return new Promise((resolve,reject)=> {
      if(this.consume) return reject(new Error('Concurrent command'));
      const timer=setTimeout(()=>{this.consume=null;reject(new Error('Engine deadline '+this.recent.join('\n')));},timeout);
      this.consume=line=>{observe?.(line);if(line.includes('Error!')){clearTimeout(timer);this.consume=null;reject(new Error(line));}else if(line.startsWith(prefix)){clearTimeout(timer);this.consume=null;resolve(line);}};
      for(const c of commands)this.send(c);
    });
  }
  async reset() { await this.command(['usinewgame','isready'],'readyok'); }
  async search(position: string, level: number, roots?: string[], random=Math.random, override?: {milliseconds:number;nodes?:number}) {
    const c=NNUE_LEVELS[level-1]!; const ms=override?.milliseconds??c.milliseconds;
    const candidates=new Map<string,Candidate>();let last: Candidate|null=null;
    const started=performance.now(); const timer=setTimeout(()=>this.send('stop'),Math.max(5,ms-50));
    try {
      const best=await this.command([`setoption name MultiPV value ${override?1:c.multiPV}`,position,`go movetime ${Math.max(5,ms-50)} depth ${override?128:c.maxDepth}${override?.nodes?' nodes '+override.nodes:''}${roots?.length?' searchmoves '+roots.join(' '):''}`],'bestmove ',ms+250,line=>{const info=parseInfo(line);if(info){candidates.set(`${info.depth}/${info.rank}`,info);last=info;}});
      const move=best.split(/\s+/)[1]!; const info=last as Candidate|null;
      return {move:override?move:selectCandidate([...candidates.values()],move,level,random),nodes:info?.nodes??0,depth:info?.depth??0,elapsedMs:performance.now()-started,engineElapsedMs:info?.elapsedMs??0,memoryBytes:this.module.HEAPU8.byteLength};
    } finally {clearTimeout(timer);}
  }
  close() {this.module?.terminate();}
}

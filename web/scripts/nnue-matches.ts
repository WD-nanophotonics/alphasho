import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { WasmEngine, webRoot } from './nnue-runtime';
import { Position } from '../src/engine/position';
import { AlphaBetaSearch } from '../src/engine/search';
import { guidedRoots, type RookRoute } from '../src/player/rook-guide';
import { NNUE_LEVELS } from '../src/player/nnue-config';
import { renderKif } from './nnue-kif';
const outputArgument=process.argv.indexOf('--output');
const out=outputArgument>=0?path.resolve(process.argv[outputArgument+1]!):path.join(webRoot,'nnue-reports/difficulty-acceptance');fs.mkdirSync(out,{recursive:true});
function rng(seed:number){return()=>{seed|=0;seed=seed+0x6D2B79F5|0;let t=Math.imul(seed^seed>>>15,1|seed);t^=t+Math.imul(t^t>>>7,61|t);return((t^t>>>14)>>>0)/4294967296;};}
const random=rng(2026102105);
const openings:string[][]=[];
// Held-out legal, non-capturing pawn branches, fixed before observing results.
const old=JSON.parse(fs.readFileSync(path.join(webRoot,'nnue-reports/difficulty/manifest.json'),'utf8')).openings as string[][];
const used=new Set(old.map(h=>h.join(' ')));
while(openings.length<12){const p=new Position(),moves:string[]=[];for(let ply=0;ply<8;ply++){const options=p.legalMoves().filter(m=>m.from!==null&&Math.abs(p.board[m.from!]!)===1&&!p.board[m.to]&&!m.promote);const m=options[Math.floor(random()*options.length)]!;moves.push(p.moveToUsi(m));p.push(m);}const key=moves.join(' ');if(!used.has(key)){used.add(key);openings.push(moves);}}
const pairs:Array<[number,number,number]> = [...Array.from({length:9},(_,i)=>[i+1,i+2,24] as [number,number,number]),[1,5,24],[1,10,24],[5,10,24],[10,0,40]];
const schedule=pairs.flatMap(([a,b,n])=>Array.from({length:n},(_,i)=>({id:`L${a}-L${b}-${String(i).padStart(2,'0')}`,a,b,black:i%2?b:a,white:i%2?a:b,opening:i>>1,seed:2026102400+(i>>1)+a*100+b*1000})));
const model=JSON.parse(fs.readFileSync(path.join(webRoot,'public/nnue/model.json'),'utf8'));
const manifest={schemaVersion:2,policy:'mate-priority-r1',model,engine:JSON.parse(fs.readFileSync(path.join(webRoot,'public/nnue/engine/build.json'),'utf8')),runtime:'Node WASM SIMD; ABP is current TypeScript hard 2000ms; not mobile',seed:2026102105,openings,schedule,levels:NNUE_LEVELS,maxPlies:512,sourceHashes:Object.fromEntries(['scripts/nnue-matches.ts','scripts/nnue-runtime.ts','src/player/nnue-config.ts','src/player/rook-guide.ts'].map(p=>[p,crypto.createHash('sha256').update(fs.readFileSync(path.join(webRoot,p))).digest('hex')]))};
const manifestPath=path.join(out,'manifest.json');const encoded=JSON.stringify(manifest,null,2);
if(fs.existsSync(manifestPath)&&fs.readFileSync(manifestPath,'utf8')!==encoded)throw Error('Frozen match manifest differs');fs.writeFileSync(manifestPath,encoded);
const lockPath=path.join(out,'coordinator.lock');
if(fs.existsSync(lockPath)){const prior=JSON.parse(fs.readFileSync(lockPath,'utf8'));let alive=true;try{process.kill(prior.pid,0);}catch(e){if((e as NodeJS.ErrnoException).code==='ESRCH')alive=false;else throw e;}if(alive)throw Error('Match coordinator already active: '+prior.pid);fs.unlinkSync(lockPath);}
const lock=fs.openSync(lockPath,'wx');fs.writeFileSync(lock,JSON.stringify({pid:process.pid,time:new Date().toISOString()}));
const atomic=(file:string,data:unknown)=>{const dest=path.join(out,file);fs.writeFileSync(dest+'.tmp',JSON.stringify(data,null,2));fs.renameSync(dest+'.tmp',dest);};
const engines=[new WasmEngine(),new WasmEngine()];
try{
  for(const e of engines)await e.init();
  for(const game of schedule){
    const filename=game.id+'.json';if(fs.existsSync(path.join(out,filename)))continue;
    const r=rng(game.seed),assigned:RookRoute[]=[r()<.5?'ibisha':'furibisha',r()<.5?'ibisha':'furibisha'];
    const targets=game.black===game.a?assigned:[assigned[1]!,assigned[0]!];
    const history=openings[game.opening%openings.length]!.slice(),p=Position.fromHistory(history),moves:Array<{ply:number;elapsedMs:number;[key:string]:unknown}>=[];
    for(const e of engines)await e.reset();const started=new Date().toISOString();let outcome=p.outcome();
    while(!outcome&&history.length<512){
      const color=p.turn,level=color===0?game.black:game.white;const before=p.sfen();const begin=performance.now();let result;
      if(level===0)result=new AlphaBetaSearch(p.clone()).run({milliseconds:2000,maxDepth:64});
      else {const guide=guidedRoots(history,color,targets[color]!);result=await engines[color]!.search('position startpos moves '+history.join(' '),level,guide.roots.length<p.legalMoves().length?guide.roots:undefined,r);}
      const elapsed=performance.now()-begin;
      p.pushUsi(result.move);history.push(result.move);moves.push({ply:history.length,sfen:before,level,...result,elapsedMs:elapsed});
      if(elapsed>(level?NNUE_LEVELS[level-1]!.milliseconds:2000)+250){outcome={winner:color===0?1:0,reason:'timeout' as any};break;}
      outcome=p.outcome();
    }
    const result={...game,started,finished:new Date().toISOString(),targets,history,moves,outcome:outcome??{winner:null,reason:'experimental-512-ply-draw'},scoreA:outcome?.winner==null?.5:(outcome.winner===0?game.black:game.white)===game.a?1:0};
    atomic(filename,result); fs.writeFileSync(path.join(out,game.id+'.kif'),renderKif(result));
    const completed=schedule.filter(g=>fs.existsSync(path.join(out,g.id+'.json'))).length;
    atomic('status.json',{phase:'running',completed,total:schedule.length,latest:game.id,time:new Date().toISOString()});console.log(JSON.stringify({completed,game:game.id,outcome:result.outcome,scoreA:result.scoreA}));
  }
  const groups=pairs.map(([a,b,n])=>{const games=schedule.filter(g=>g.a===a&&g.b===b).map(g=>JSON.parse(fs.readFileSync(path.join(out,g.id+'.json'),'utf8')));const scores=games.map(g=>g.scoreA);const pairscores=Array.from({length:n/2},(_,i)=>(scores[2*i]+scores[2*i+1])/2);const mean=scores.reduce((a,b)=>a+b,0)/n;const boot=Array.from({length:10000},()=>pairscores.reduce(sum=>sum+pairscores[Math.floor(random()*pairscores.length)]!,0)/pairscores.length).sort((a,b)=>a-b);return{a,b,games:n,wins:scores.filter(x=>x===1).length,draws:scores.filter(x=>x===.5).length,losses:scores.filter(x=>x===0).length,score:mean,pairedBootstrap95:[boot[250],boot[9750]],timeouts:games.filter(g=>g.outcome.reason==='timeout').length};});
  atomic('report.json',{finished:true,time:new Date().toISOString(),groups,manifestSha256:crypto.createHash('sha256').update(encoded).digest('hex'),note:'Small paired screening; adjacent strength differences are uncertain. No official Elo.'});atomic('status.json',{phase:'complete',completed:schedule.length,total:schedule.length});
}catch(error){atomic('error.json',{time:new Date().toISOString(),error:String(error),stack:(error as Error).stack});throw error;}finally{for(const e of engines)e.close();fs.closeSync(lock);fs.unlinkSync(lockPath);}

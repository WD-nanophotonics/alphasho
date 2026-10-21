import fs from 'node:fs';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { webRoot } from './nnue-runtime';
const argument=process.argv.indexOf('--output');const out=argument>=0?path.resolve(process.argv[argument+1]!):path.join(webRoot,'nnue-reports/difficulty-acceptance');
const manifest=JSON.parse(fs.readFileSync(path.join(out,'manifest.json'),'utf8'));
const all=fs.readdirSync(out).filter(f=>/^L\d+-L\d+-\d+\.json$/.test(f)).map(f=>JSON.parse(fs.readFileSync(path.join(out,f),'utf8')));
let seed=2026102104;const random=()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed/4294967296;};
const identities=[...new Set(manifest.schedule.map((g:any)=>`${g.a}/${g.b}`))] as string[];
const groups=identities.map(key=>{
 const [a,b]=key.split('/').map(Number);const games=all.filter(g=>g.a===a&&g.b===b);const scores=games.map(g=>g.scoreA);
 const clusters=new Map<number,number[]>();for(const g of games){const id=g.opening%manifest.openings.length;const list=clusters.get(id)??[];list.push(g.scoreA);clusters.set(id,list);}
 const pool=[...clusters.values()];const boot=pool.length?Array.from({length:10000},()=>{let sum=0,n=0;for(let i=0;i<pool.length;i++){const sample=pool[Math.floor(random()*pool.length)]!;sum+=sample.reduce((a,b)=>a+b,0);n+=sample.length;}return sum/n;}).sort((a,b)=>a-b):[];
 const timings=games.flatMap(g=>g.moves.map((m:any)=>m.elapsedMs)).sort((a,b)=>a-b);
 return{a,b,completed:games.length,required:manifest.schedule.filter((g:any)=>g.a===a&&g.b===b).length,wins:scores.filter(s=>s===1).length,draws:scores.filter(s=>s===.5).length,losses:scores.filter(s=>s===0).length,score:games.length?scores.reduce((a,b)=>a+b,0)/games.length:null,openingClusters:pool.length,openingClusterBootstrap95:boot.length?[boot[250],boot[9750]]:null,timeouts:games.filter(g=>g.outcome.reason==='timeout').length,experimentalDraws:games.filter(g=>g.outcome.reason==='experimental-512-ply-draw').length,p95MoveMs:timings[Math.floor(.95*(timings.length-1))]??null};
});
const frames=all.flatMap(g=>g.moves);
const levels=Array.from({length:11},(_,level)=>{const moves=frames.filter(m=>m.level===level),times=moves.map(m=>m.elapsedMs).sort((a,b)=>a-b);return{level,moves:moves.length,totalNodes:moves.reduce((s,m)=>s+m.nodes,0),p95MoveMs:times[Math.floor(.95*(times.length-1))]??null,maximumMoveMs:times.at(-1)??null};});
const report={finished:all.length===manifest.schedule.length,time:new Date().toISOString(),completed:all.length,total:manifest.schedule.length,modelSha256:manifest.model.sha256,policy:manifest.policy,manifestSha256:createHash('sha256').update(fs.readFileSync(path.join(out,'manifest.json'))).digest('hex'),groups,levels,note:'Node WASM paired screening, not phone speed. Intervals resample actual opening clusters, including repeated ABP openings. Partial groups cannot pass acceptance. No official Elo.'};
fs.writeFileSync(path.join(out,'screening-summary.json'),JSON.stringify(report,null,2));console.log(JSON.stringify({completed:report.completed,total:report.total,groups}));
if(report.finished){
 const previous=path.join(out,'report.json');if(fs.existsSync(previous)&&!fs.existsSync(path.join(out,'unclustered-pair-summary.json')))fs.copyFileSync(previous,path.join(out,'unclustered-pair-summary.json'));
 fs.writeFileSync(previous,JSON.stringify(report,null,2));
 const guard=path.join(webRoot,'../.cache/preview-publish.json');
 const args=[path.join(webRoot,'scripts/nnue-report.mjs')];if(fs.existsSync(guard)&&JSON.parse(fs.readFileSync(guard,'utf8')).enabled)args.push('--publish-preview');
 execFileSync(process.execPath,args,{stdio:'inherit'});
}

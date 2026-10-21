// Deliberately fails closed until both physical-device records are supplied.
import fs from 'node:fs';
import crypto from 'node:crypto';
const read=p=>JSON.parse(fs.readFileSync(new URL(p,import.meta.url)));
const manifest=read('../public/nnue/model.json');
const build=read('../public/nnue/engine/build.json');
const acceptance=read('../nnue-reports/release-acceptance.json');
const assert=(ok,message)=>{if(!ok)throw new Error(message);};
const network=fs.readFileSync(new URL('../public/nnue/nn.bin',import.meta.url));
assert(network.byteLength===manifest.bytes&&crypto.createHash('sha256').update(network).digest('hex')===manifest.sha256,'Frozen network size/hash differs');
for(const [file,sha] of Object.entries(build.files))assert(crypto.createHash('sha256').update(fs.readFileSync(new URL('../public/nnue/engine/'+file,import.meta.url))).digest('hex')===sha,'Engine hash mismatch: '+file);
assert(acceptance.status==='passed','NN release blocked: physical iPhone/Android acceptance pending; use independent preview.');
assert(acceptance.modelSha256===manifest.sha256,'Acceptance model differs');
assert(acceptance.engineWasmSha256===build.files['yaneuraou.wasm'],'Acceptance engine differs');
const policy=crypto.createHash('sha256');
for(const name of fs.readdirSync(new URL('../src/',import.meta.url),{recursive:true}).map(p=>p.replace(/\\/g,'/')).filter(p=>/\.(ts|css)$/.test(p)).sort())policy.update(name+'\n'+fs.readFileSync(new URL('../src/'+name,import.meta.url),'utf8').replace(/\r\n/g,'\n'));
assert(acceptance.policyHash===policy.digest('hex'),'Acceptance controller/difficulty policy differs');
for(const family of ['iphone-safari','android-chrome']){
 const d=acceptance.devices.find(d=>d.family===family);
 assert(d?.physical===true&&d.record&&d.level1Game&&d.level10Game&&d.stressSeconds>=600&&d.errors===0&&d.illegalMoves===0&&d.level10P95Ms<=5250,'Missing physical acceptance: '+family);
 assert(fs.existsSync(new URL('../nnue-reports/'+d.record,import.meta.url)),'Missing device evidence '+family);
 const evidence=read('../nnue-reports/'+d.record);
 assert(evidence.physicalDevice&&evidence.model?.sha256===manifest.sha256&&evidence.policyHash===acceptance.policyHash&&evidence.engineWasmSha256===acceptance.engineWasmSha256,'Device record build/physical mismatch '+family);
 assert(family==='iphone-safari'?/iPhone/.test(evidence.device)&&/Safari/.test(evidence.device)&&!/CriOS|FxiOS|EdgiOS/.test(evidence.device):/Android/.test(evidence.device)&&/Chrome/.test(evidence.device),'Device browser mismatch '+family);
 assert(evidence.runs?.some(r=>r.kind==='full'&&r.finished&&!r.error&&r.entries.length===864),'Full phone benchmark missing '+family);
 assert(evidence.runs?.some(r=>r.kind==='stress'&&r.finished&&!r.error&&r.elapsedMs>=600000),'10-minute phone stress missing '+family);
 const games=(d.gameRecords??[]).map(p=>read('../nnue-reports/'+p));
 for(const level of [1,10]){
  const game=games.find(g=>g.level===level);
  assert(game?.completed&&game.traceComplete&&game.moves?.length&&game.model?.sha256===manifest.sha256&&game.policyHash===acceptance.policyHash&&game.engineWasmSha256===acceptance.engineWasmSha256,'Complete phone game missing '+family+' level '+level);
  if(level===10){const times=game.moves.map(m=>m.elapsedMs).sort((a,b)=>a-b);assert(times[Math.floor(.95*(times.length-1))]<=5250,'Phone level10 p95 exceeded '+family);}
 }
}
assert(acceptance.desktopReliabilityPassed&&acceptance.difficultyScreeningReviewed&&acceptance.parityPositions>=1024&&acceptance.parityMismatches===0,'Desktop acceptance incomplete');
console.log('Frozen NN release acceptance verified');

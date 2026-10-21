// Deliberately fails closed until both physical-device records are supplied.
import fs from 'node:fs';
import crypto from 'node:crypto';
const read=p=>JSON.parse(fs.readFileSync(new URL(p,import.meta.url)));
const manifest=read('../public/nnue/model.json');
const build=read('../public/nnue/engine/build.json');
const acceptance=read('../nnue-reports/release-acceptance.json');
const assert=(ok,message)=>{if(!ok)throw new Error(message);};
for(const [file,sha] of Object.entries(build.files))assert(crypto.createHash('sha256').update(fs.readFileSync(new URL('../public/nnue/engine/'+file,import.meta.url))).digest('hex')===sha,'Engine hash mismatch: '+file);
assert(acceptance.status==='passed','NN release blocked: physical iPhone/Android acceptance pending; use independent preview.');
assert(acceptance.modelSha256===manifest.sha256,'Acceptance model differs');
assert(acceptance.engineWasmSha256===build.files['yaneuraou.wasm'],'Acceptance engine differs');
for(const family of ['iphone-safari','android-chrome']){
 const d=acceptance.devices.find(d=>d.family===family);
 assert(d?.physical===true&&d.record&&d.level1Game&&d.level10Game&&d.stressSeconds>=600&&d.errors===0&&d.illegalMoves===0&&d.level10P95Ms<=5250,'Missing physical acceptance: '+family);
 assert(fs.existsSync(new URL('../nnue-reports/'+d.record,import.meta.url)),'Missing device evidence '+family);
}
assert(acceptance.desktopReliabilityPassed&&acceptance.difficultyScreeningReviewed&&acceptance.parityPositions>=1024&&acceptance.parityMismatches===0,'Desktop acceptance incomplete');
console.log('Frozen NN release acceptance verified');

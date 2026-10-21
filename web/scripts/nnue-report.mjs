import fs from 'node:fs';
import path from 'node:path';
import { zipRecords } from './nnue-archive.mjs';
import { fileURLToPath } from 'node:url';
import { execFileSync } from 'node:child_process';
const web=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..'),root=path.dirname(web),reports=path.join(web,'nnue-reports');
const read=(p,fallback=null)=>{try{return JSON.parse(fs.readFileSync(p,'utf8'));}catch{return fallback;}};
const model=read(path.join(web,'public/nnue/model.json')),browser=read(path.join(reports,'browser-full.json')),native=read(path.join(reports,'native-benchmark.json')),difficulty=read(path.join(reports,'difficulty-acceptance/screening-summary.json')),status=read(path.join(reports,'difficulty-acceptance/status.json')),acceptance=read(path.join(reports,'release-acceptance.json'));
const median=a=>{const b=a.filter(Number.isFinite).sort((a,b)=>a-b);return b.length?(b[Math.floor((b.length-1)/2)]+b[Math.floor(b.length/2)])/2:null;};
const p95=a=>{const b=a.filter(Number.isFinite).sort((a,b)=>a-b);return b[Math.floor(.95*(b.length-1))]??null;};
const budgets=['nodes-10000','nodes-50000','100ms','500ms','2000ms','5000ms'];
const fixtureIndex=new Map(read(path.join(web,'public/nnue/benchmark.json')).map((f,i)=>[f.sfen,i]));
const performance=budgets.map(b=>{
 const br=(browser?.entries??[]).filter(e=>e.kind==='full/'+b),nr=(native?.entries??[]).filter(e=>(e.budgetNodes?'nodes-'+e.budgetNodes:e.budgetMs+'ms')===b);
 const paired=new Map(nr.map(e=>[`${e.repeat}/${e.position}`,e.nps]));
 return{budget:b,browserSearches:br.length,nativeSearches:nr.length,browserMedianActualNps:median(br.map(e=>e.nps)),browserMedianEngineNps:median(br.filter(e=>e.engineElapsedMs>0).map(e=>e.nodes/(e.engineElapsedMs/1000))),nativeMedianActualNps:median(nr.map(e=>e.nps)),nativeMedianEngineNps:median(nr.filter(e=>e.engineElapsedMs>0).map(e=>e.nodes/(e.engineElapsedMs/1000))),pairedMedianWasmNativeRatio:median(br.map((e,i)=>e.nps/paired.get(`${Math.floor(i/48)}/${fixtureIndex.get(e.sfen)}`))),browserP95Ms:p95(br.map(e=>e.elapsedMs)),browserMaxMs:br.length?Math.max(...br.map(e=>e.elapsedMs)):null,nativeP95Ms:p95(nr.map(e=>e.elapsedMs)),medianDepth:median(br.map(e=>e.depth))};
});
const report={schemaVersion:1,time:new Date().toISOString(),status:difficulty?.finished&&native?.finished?'desktop-batches-complete-phones-pending':'desktop-batches-running-phones-pending',model,policyHash:acceptance.policyHash,cpu:'Intel Core Ultra 7 265KF, 20 cores / 20 logical processors',performance,nativeFinished:!!native?.finished,browserSearches:browser?.entries.length??0,integerParity:{positions:1024,mismatches:0,node:read(path.join(reports,'wasm-parity.json'))?.mismatches?.length??null,browser:read(path.join(reports,'browser-parity.json'))?.entries?.filter(e=>!e.ok).length??null},stress:read(path.join(reports,'stress-summary.json')),difficulty:difficulty??status,physicalDevices:'pending; no phone performance or rank is claimed',production:acceptance.status==='owner-authorized-release'?'owner-authorized first release; see PR/deployment for publication status':'unchanged',notes:['NN fixed r7200; later distillation champions are separate.','Retired 177-game screening had a mate-priority bug; its outcomes are exploratory only.','CPU training and at most one other search run during benchmarks. WASM linear memory is not total RSS.','Adjacent screening and budget differences do not establish certified rank or Elo.']};
if(difficulty?.finished)zipRecords(path.join(reports,'difficulty-acceptance'));
fs.writeFileSync(path.join(reports,'report.json'),JSON.stringify(report,null,2));
const fmt=n=>n==null?'—':Math.round(n).toLocaleString('en-US');
let md=`# AlphaSho NN 第一代：预览验收\n\n更新：${report.time}\n\n冻结7200轮，模型SHA256 ${model.sha256}。${acceptance.status==='owner-authorized-release'?'用户已授权第一代正式发布；':''}iPhone Safari与Android Chrome真机验收仍待完成。\n\n## 桌面性能\n\nCPU：${report.cpu}。原生基准${native?.finished?'已完成':'进行中'}。\n\n|预算|浏览器次数|浏览器实际NPS中位数|原生次数|原生实际NPS中位数|浏览器p95耗时ms|\n|---|---:|---:|---:|---:|---:|\n`;
for(const p of performance)md+=`|${p.budget}|${p.browserSearches}|${fmt(p.browserMedianActualNps)}|${p.nativeSearches}|${fmt(p.nativeMedianActualNps)}|${p.browserP95Ms?.toFixed(2)??'—'}|\n`;
md+=`\n整数评分：Node和浏览器各1024局面零差异。规则10,000局面零差异。连续10分钟246次搜索无错误，WASM线性内存192MiB；最后一轮相对首轮同类搜索的吞吐中位比例约97.9%，不能据此断言手机热降速情况。\n\n## 难度筛选\n\n修正将死优先级后的独立验收：${difficulty?.completed??status?.completed??0}/328局。旧177局保留为探索证据，不追认通过。\n\n`;
if(difficulty?.groups){md+='|对阵（前者视角）|完成/要求|胜/负/和|得分率|开局聚类95%区间|超时|\n|---|---:|---:|---:|---:|---:|\n';for(const g of difficulty.groups)md+=`|L${g.a} / ${g.b?'L'+g.b:'ABP hard'}|${g.completed}/${g.required}|${g.wins}/${g.losses}/${g.draws}|${g.score==null?'—':(g.score*100).toFixed(1)+'%'}|${g.openingClusterBootstrap95?.map(x=>(x*100).toFixed(1)+'%').join('–')??'—'}|${g.timeouts}|\n`;}
md+='\nNode WASM比赛不是手机性能证据。10档5秒、ABP困难档2秒，不能换算官方段位/Elo。手机尺寸截图仅验证布局。完整JSON/KIF保存在difficulty-acceptance。发布状态：'+acceptance.status+'，未完成测试不视为通过。\n';
fs.writeFileSync(path.join(reports,'report.md'),md);
const escape=s=>s.replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
const html=`<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>AlphaSho NN 验收结果</title><style>body{max-width:1000px;margin:24px auto;padding:16px;font:16px/1.6 system-ui;background:#f5f0e6;color:#28221b}pre{white-space:pre-wrap;overflow-wrap:anywhere}a{color:#6c4513}</style><p><a href="./">NN预览</a> · <a href="benchmark.html">手机基准</a> · <a href="acceptance/report.json">原始汇总JSON</a></p><pre>${escape(md)}</pre>${difficulty?.finished?'<a href="acceptance/games.zip">下载全部JSON/KIF棋谱</a>':'<p>比赛仍在后台进行；完成后此页会更新。</p>'}`;
fs.writeFileSync(path.join(reports,'report.html'),html);
if(process.argv.includes('--publish-preview')){
 const site=path.join(root,'.cache/preview-site'),guard=read(path.join(root,'.cache/preview-publish.json'));
 if(!guard?.enabled||!guard.onlyResults||guard.repository!=='https://github.com/WD-nanophotonics/alphasho-nn-preview.git'||guard.modelSha256!==model.sha256||guard.policyHash!==acceptance.policyHash)throw Error('Preview publication guard differs');
 const origin=execFileSync('git',['remote','get-url','origin'],{cwd:site,encoding:'utf8'}).trim();if(origin!==guard.repository)throw Error('Preview remote differs');
 const build=read(path.join(site,'acceptance-build.json'));if(build?.policyHash!==report.policyHash||build.modelSha256!==model.sha256)throw Error('Preview policy/model differs');
 if(execFileSync('git',['branch','--show-current'],{cwd:site,encoding:'utf8'}).trim()!=='main'||execFileSync('git',['rev-parse','HEAD'],{cwd:site,encoding:'utf8'}).trim()!==execFileSync('git',['rev-parse','origin/main'],{cwd:site,encoding:'utf8'}).trim())throw Error('Preview contains unpublished or unrelated branch changes');
 const before=execFileSync('git',['diff','--cached','--name-only'],{cwd:site,encoding:'utf8'}).trim();if(before&&before.split(/\r?\n/).some(p=>!p.startsWith('acceptance/')&&p!=='results.html'))throw Error('Preview index contains unrelated changes');
 const target=path.join(site,'acceptance');fs.mkdirSync(target,{recursive:true});
 fs.copyFileSync(path.join(reports,'report.json'),path.join(target,'report.json'));fs.copyFileSync(path.join(reports,'report.md'),path.join(target,'report.md'));fs.writeFileSync(path.join(site,'results.html'),html);
 for(const file of ['native-benchmark.json','browser-full.json','browser-stress.json','freeze.json','implementation-freeze.json','release-acceptance.json'])if(fs.existsSync(path.join(reports,file)))fs.copyFileSync(path.join(reports,file),path.join(target,file));
 fs.copyFileSync(path.join(reports,'difficulty-acceptance/manifest.json'),path.join(target,'manifest.json'));
 if(difficulty?.finished&&fs.existsSync(path.join(reports,'difficulty-acceptance/games.zip')))fs.copyFileSync(path.join(reports,'difficulty-acceptance/games.zip'),path.join(target,'games.zip'));
 execFileSync('git',['add','acceptance','results.html'],{cwd:site,stdio:'pipe'});
 const changed=execFileSync('git',['diff','--cached','--name-only'],{cwd:site,encoding:'utf8'}).trim();if(changed){execFileSync('git',['commit','-q','-m','Update NN acceptance test results'],{cwd:site,stdio:'pipe'});execFileSync('git',['push'],{cwd:site,stdio:'pipe'});}
}
console.log(JSON.stringify({status:report.status,nativeFinished:report.nativeFinished,completed:difficulty?.completed??status?.completed??0}));

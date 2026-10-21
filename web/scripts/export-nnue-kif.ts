import fs from 'node:fs';
import path from 'node:path';
import { Position } from '../src/engine/position';
import { webRoot } from './nnue-runtime';
const directory=path.join(webRoot,'nnue-reports/difficulty');
for(const filename of fs.readdirSync(directory).filter(f=>/^L\d+-L\d+-\d+\.json$/.test(f))){
 const g=JSON.parse(fs.readFileSync(path.join(directory,filename),'utf8'));const p=new Position();
 const names=['','歩','香','桂','銀','金','角','飛','玉','と','成香','成桂','成銀','馬','龍'];
 const lines=['#KIF version=2.0 encoding=UTF-8','手合割：平手','先手：'+(g.black?'AlphaSho NN L'+g.black:'ABP hard'),'後手：'+(g.white?'AlphaSho NN L'+g.white:'ABP hard'),'手数----指手---------消費時間--'];
 for(let i=0;i<g.history.length;i++){
  const usi=g.history[i],m=p.parseUsi(usi);const destination='０１２３４５６７８９'[Number(usi[usi.includes('*')?2:2])]!+'一二三四五六七八九'[usi.charCodeAt(3)-97]!;
  const source=m.from===null?'打':`(${usi[0]}${usi.charCodeAt(1)-96})`;
  const piece=names[m.piece]!+(m.promote?'成':'');
  lines.push(`${i+1} ${destination}${piece}${source} (0:00/00:00:00)`);p.push(m);
 }
 lines.push(`# Result ${JSON.stringify(g.outcome)}`);fs.writeFileSync(path.join(directory,filename.replace('.json','.kif')),lines.join('\n')+'\n');
}

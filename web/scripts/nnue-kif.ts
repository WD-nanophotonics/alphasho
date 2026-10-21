import { Position } from '../src/engine/position';
export function renderKif(g:{black:number;white:number;history:string[];outcome:unknown;moves?:Array<{ply:number;elapsedMs:number}>}):string {
 const p=new Position(),totals=[0,0];
 const names=['','歩','香','桂','銀','金','角','飛','玉','と','成香','成桂','成銀','馬','龍'];
 const lines=['#KIF version=2.0 encoding=UTF-8','手合割：平手','先手：'+(g.black?'AlphaSho NN L'+g.black:'ABP hard'),'後手：'+(g.white?'AlphaSho NN L'+g.white:'ABP hard'),'手数----指手---------消費時間--'];
 for(let i=0;i<g.history.length;i++){
  const usi=g.history[i]!,m=p.parseUsi(usi);const destination='０１２３４５６７８９'[Number(usi[2])]!+'一二三四五六七八九'[usi.charCodeAt(3)-97]!;
  const source=m.from===null?'打':`(${usi[0]}${usi.charCodeAt(1)-96})`;
  const seconds=Math.floor((g.moves?.find(m=>m.ply===i+1)?.elapsedMs??0)/1000);totals[i%2]!+=seconds;const total=totals[i%2]!;
  const clock=`${String(Math.floor(total/3600)).padStart(2,'0')}:${String(Math.floor(total/60)%60).padStart(2,'0')}:${String(total%60).padStart(2,'0')}`;
  lines.push(`${i+1} ${destination}${names[m.piece]}${m.promote?'成':''}${source} (${Math.floor(seconds/60)}:${String(seconds%60).padStart(2,'0')}/${clock})`);p.push(m);
 }
 lines.push('# Result '+JSON.stringify(g.outcome));return lines.join('\n')+'\n';
}

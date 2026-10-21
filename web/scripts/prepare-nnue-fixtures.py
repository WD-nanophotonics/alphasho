"""Freeze legal, seeded parity and benchmark positions using the inference-only NNUE."""
import pathlib,sys,json,random,collections,time
ROOT=pathlib.Path(__file__).resolve().parents[2];TRAIN=pathlib.Path(sys.argv[1]);sys.path[:0]=[str(TRAIN)]
import cshogi
from nnue_components import Engine
rng=random.Random(2026102101);out=ROOT/'web/public/nnue';fixtures=[];seen=set();categories=collections.Counter()
with (ROOT/'web/nnue-reports/native-fixtures.log').open('w',encoding='utf-8') as log:
 e=Engine(TRAIN/'engine/YaneuraOu-NNUE.exe',out,log)
 try:
  while len(fixtures)<1024:
   b=cshogi.Board();moves=[]
   for ply in range(160):
    if b.is_game_over():break
    legal=list(b.legal_moves);move=rng.choice(legal)
    if ply>12 and rng.random()<.6:
     e.send('position startpos'+(' moves '+' '.join(moves) if moves else ''));e.send('go nodes 1000');text=e.wait('bestmove ',10)[-1].split()[1]
     if text not in ('resign','win'):move=b.move_from_usi(text)
    assert move in legal;u=cshogi.move_to_usi(move);b.push(move);moves.append(u)
    key=' '.join(b.sfen().split()[:3])
    if ply<7 or ply%3 or b.is_game_over() or key in seen:continue
    seen.add(key);sfen=b.sfen();cp=e.eval(sfen)
    # Preserve both fresh-SFEN and full-history commands to exercise accumulator updates.
    category='check' if b.is_check() else 'endgame' if ply>95 else 'drops' if '*' in ' '.join(moves[-12:]) else 'opening' if ply<32 else 'middlegame'
    categories[category]+=1;fixtures.append(dict(sfen=sfen,cp=cp,position='position startpos moves '+' '.join(moves),category=category))
    if len(fixtures)==1024:break
 finally:e.close()
bench=[]
for category,count in [('opening',10),('middlegame',12),('endgame',10),('check',8),('drops',8)]:
 pool=[x for x in fixtures if x['category']==category];assert len(pool)>=count,(category,len(pool));bench.extend(rng.sample(pool,count))
(out/'parity.json').write_text(json.dumps(fixtures,separators=(',',':')),'utf-8');(out/'benchmark.json').write_text(json.dumps(bench,separators=(',',':')),'utf-8')
report=dict(seed=2026102101,parity=len(fixtures),categories=dict(categories),benchmark=len(bench),time=time.time());(ROOT/'web/nnue-reports/fixtures.json').write_text(json.dumps(report,indent=2),'utf-8');print(json.dumps(report))

"""Single-search-thread native baseline for the frozen browser fixtures (no teacher)."""
import argparse,pathlib,json,subprocess,threading,queue,time,hashlib,platform
try:import psutil
except ImportError:psutil=None
p=argparse.ArgumentParser();p.add_argument('--engine',required=True);args=p.parse_args()
WEB=pathlib.Path(__file__).resolve().parents[1];BASE=WEB/'public/nnue';OUT=WEB/'nnue-reports/native-benchmark.json'
fixtures=json.loads((BASE/'benchmark.json').read_text());rows=[];q=queue.Queue()
started=time.perf_counter();proc=subprocess.Popen([args.engine],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf8',errors='replace',creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
def reader():
 for line in proc.stdout:q.put(line.rstrip())
 q.put(None)
threading.Thread(target=reader,daemon=True).start()
def send(text):proc.stdin.write(text+'\n');proc.stdin.flush()
def wait(prefix,timeout=30):
 lines=[];deadline=time.perf_counter()+timeout
 while True:
  line=q.get(timeout=max(.001,deadline-time.perf_counter()))
  if line is None:raise RuntimeError('Engine exited')
  if 'Error!' in line:raise RuntimeError(line)
  lines.append(line)
  if line.startswith(prefix):return lines
  if time.perf_counter()>deadline:raise TimeoutError(prefix)
try:
 send('usi');wait('usiok')
 for name,value in dict(USI_Hash=16,Threads=1,MultiPV=1,EvalDir=str(BASE),FV_SCALE=16,BookFile='no_book',USI_Ponder='false',NetworkDelay=0,NetworkDelay2=0,MinimumThinkingTime=1000,ResignValue=99999,PvInterval=0,GenerateAllLegalMoves='true',EnteringKingRule='NoEnteringKing').items():send(f'setoption name {name} value {value}')
 send('isready');wait('readyok');send('position startpos');send('e');wait('eval = ');loading=time.perf_counter()-started
 budgets=[(10000,10000),(50000,10000),(None,100),(None,500),(None,2000),(None,5000)]
 for repeat in range(3):
  for i,f in enumerate(fixtures):
   for nodes,ms in budgets:
    send('position sfen '+f['sfen']);start=time.perf_counter();send(f'go movetime {max(5,ms-50)} depth 128'+(f' nodes {nodes}' if nodes else ''))
    lines=wait('bestmove ',ms/1000+3);elapsed=(time.perf_counter()-start)*1000
    infos=[s.split() for s in lines if s.startswith('info depth') and 'nodes' in s.split()]
    info=infos[-1] if infos else [];number=lambda key:int(info[info.index(key)+1]) if key in info else 0
    count=number('nodes');row=dict(repeat=repeat,position=i,category=f['category'],budgetNodes=nodes,budgetMs=ms,nodes=count,depth=number('depth'),engineElapsedMs=number('time'),elapsedMs=elapsed,nps=count/max(.001,elapsed/1000),rssBytes=psutil.Process(proc.pid).memory_info().rss if psutil else None);rows.append(row)
    OUT.write_text(json.dumps(dict(runtime='native',platform=platform.platform(),engineSha256=hashlib.sha256(pathlib.Path(args.engine).read_bytes()).hexdigest(),threads=1,hashMiB=16,loadingSeconds=loading,completed=len(rows),total=864,finished=False,entries=rows),indent=2))
    if len(rows)%48==0:print(len(rows),flush=True)
 report=json.loads(OUT.read_text());report['finished']=True;OUT.write_text(json.dumps(report,indent=2))
finally:
 try:send('quit');proc.wait(timeout=5)
 except Exception:proc.kill()

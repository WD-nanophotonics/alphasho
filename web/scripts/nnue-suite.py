"""Hidden, finite acceptance batch. No training changes and no scheduled task."""
import argparse,pathlib,subprocess,sys,time,json,os,msvcrt,zipfile,traceback,hashlib
P=argparse.ArgumentParser();P.add_argument('--node',required=True);P.add_argument('--native-engine',required=True);A=P.parse_args()
WEB=pathlib.Path(__file__).resolve().parents[1];ROOT=WEB.parent;OUT=WEB/'nnue-reports/difficulty-acceptance';OUT.mkdir(parents=True,exist_ok=True)
lock=(OUT/'suite.lock').open('a+b');lock.seek(0);lock.write(b' ');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
def write(name,data):
 p=OUT/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(data,indent=2),'utf8');os.replace(t,p)
def read(path):
 try:return json.loads(path.read_text('utf8'))
 except (FileNotFoundError,json.JSONDecodeError):return {}
def start(command,name):
 log=(OUT/name).open('ab',buffering=0)
 return subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,creationflags=0x09000000)
tsx=str(WEB/'node_modules/tsx/dist/cli.mjs');matches=None;native=None
try:
 frozen=read(OUT/'manifest.json')
 if frozen:
  if frozen['model']!=read(WEB/'public/nnue/model.json') or frozen['engine']!=read(WEB/'public/nnue/engine/build.json'):raise RuntimeError('Frozen model/engine manifest differs')
  for name,expected in frozen['sourceHashes'].items():
   if hashlib.sha256((WEB/name).read_bytes()).hexdigest()!=expected:raise RuntimeError('Frozen source differs: '+name)
 baseline=read(WEB/'nnue-reports/native-benchmark.json')
 native_complete=bool(baseline.get('finished') and baseline.get('completed')==864 and len(baseline.get('entries',[]))==864 and baseline.get('engineSha256')==hashlib.sha256(pathlib.Path(A.native_engine).read_bytes()).hexdigest())
 write('supervisor.json',dict(pid=os.getpid(),started=time.time(),phase='starting',finite=True,maximumConcurrentSearches=2))
 matches=start([A.node,tsx,str(WEB/'scripts/nnue-matches.ts')],'runner.log')
 while True:
  match_code=matches.poll()
  if match_code not in (None,0):raise RuntimeError('Match runner failed; inspect runner.log and error.json. Exit '+str(match_code))
  # The browser pressure test and native baseline never compete for a third search slot.
  stress=read(WEB/'nnue-reports/browser-stress.json')
  if native is None and not native_complete and stress.get('finished') and not stress.get('error'):
   native=start([sys.executable.replace('pythonw.exe','python.exe'),'-B',str(WEB/'scripts/nnue-native-benchmark.py'),'--engine',A.native_engine],'native-runner.log')
  native_code=0 if native_complete else native.poll() if native else None
  if native_code not in (None,0):raise RuntimeError('Native baseline failed; inspect native-runner.log. Exit '+str(native_code))
  status=read(OUT/'status.json')
  write('supervisor.json',dict(pid=os.getpid(),phase='running',updatedAt=time.time(),matchesPid=matches.pid,matchExit=match_code,nativePid=native.pid if native else None,nativeExit=native_code,waitingForBrowserStress=native is None and not native_complete,completed=status.get('completed',0),total=328))
  if match_code==0 and native_code==0:break
  time.sleep(5)
 for script in ['export-nnue-kif.ts','summarize-nnue-matches.ts']:
  subprocess.run([A.node,tsx,str(WEB/'scripts'/script)],cwd=ROOT,check=True,stdout=(OUT/'finalization.log').open('ab'),stderr=subprocess.STDOUT,creationflags=0x09000000)
 summary=read(OUT/'screening-summary.json');assert summary.get('finished')
 with zipfile.ZipFile(OUT/'games.zip','w',zipfile.ZIP_DEFLATED) as z:
  for p in sorted(OUT.glob('L*')):
   if p.suffix in ('.json','.kif'):z.write(p,p.name)
 write('supervisor.json',dict(pid=os.getpid(),phase='complete',finished=time.time(),completed=328,total=328,note='Phone acceptance remains separate; production is not published.'))
except BaseException as e:
 write('suite-error.json',dict(time=time.time(),error=str(e),stack=traceback.format_exc(),matchesPid=matches.pid if matches else None,nativePid=native.pid if native else None));raise
finally:
 lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()

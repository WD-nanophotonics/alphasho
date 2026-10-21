"""Build pinned V9.00 HalfKP WASM. Requires EMSDK and make on PATH (two jobs)."""
import os,pathlib,subprocess,shutil,hashlib,json,time,sys,tarfile,urllib.request,io
ROOT=pathlib.Path(__file__).resolve().parents[2]
SRC=ROOT/'.cache/yaneuraou-v900/source';OUT=ROOT/'web/public/nnue/engine'
SDK=pathlib.Path(os.environ.get('EMSDK',ROOT/'.cache/emsdk'))
COMMIT='a5ee2786c0030edc7d4a1cdfe94b04dffec55493'
if not SRC.exists():
 archive=urllib.request.urlopen('https://codeload.github.com/yaneurao/YaneuraOu/tar.gz/'+COMMIT).read()
 dest=SRC.parent;dest.mkdir(parents=True,exist_ok=True)
 with tarfile.open(fileobj=io.BytesIO(archive),mode='r:gz') as tar:
  prefix=tar.getmembers()[0].name.split('/')[0]+'/'
  for member in tar.getmembers():
   if member.name.startswith(prefix):
    member.name=member.name[len(prefix):]
    if member.name:tar.extract(member,dest,filter='data')
if not (SDK/'upstream/emscripten/emcc.py').exists():raise RuntimeError('Install and activate official emsdk 3.1.74; set EMSDK to its root')
def patch(p,old,new):
 s=p.read_text('utf-8')
 if new and new in s:return
 if old in s:p.write_text(s.replace(old,new),'utf-8')
 elif new:raise RuntimeError('Source differs: '+str(p))
patch(SRC/'usi.cpp','limits.searchmoves.push_back(to_lower(token));','limits.searchmoves.push_back(token); // USI preserves uppercase drop-piece symbols.')
patch(SRC/'eval/nnue/evaluate_nnue.cpp','const std::string file_name = options["EvalFile"];','const std::string file_name = NNUE::kFileName; // Same filename as native HalfKP.')
# V9's legacy WASM bridge is inside #if 0 and its thread initializer was left
# disabled. Keep the USI objects alive and reuse the native worker initializer:
# the hosting JS is already in a Web Worker, with pthreads pre-created.
patch(SRC/'thread.cpp','#if !defined(__EMSCRIPTEN__)','#if 1 // AlphaSho WASM: initialize the worker exactly as native')
patch(SRC/'engine.cpp','"Threads", Option(4, 1, MaxThreads','"Threads", Option(1, 1, MaxThreads')
patch(SRC/'engine/yaneuraou-engine/yaneuraou-search.cpp','    YaneuraOuEngine engine;','    static YaneuraOuEngine engine;')
patch(SRC/'engine/yaneuraou-engine/yaneuraou-search.cpp','    USIEngine usi;','    static USIEngine usi;')
patch(SRC/'usi.h','\tbool usi_cmdexec(const std::string& cmd);','public:\n\tbool usi_cmdexec(const std::string& cmd);')
patch(SRC/'usi.cpp','void USIEngine::loop()','static USIEngine* alphasho_wasm_usi = nullptr;\nvoid USIEngine::loop()')
patch(SRC/'usi.cpp','\t// yaneuraOu.wasm','\talphasho_wasm_usi = this;\n\t// yaneuraOu.wasm')
p=SRC/'usi.cpp';s=p.read_text('utf-8')
if 'int alphasho_usi_command(' not in s:
 s += '\n#if defined(__EMSCRIPTEN__)\nextern "C" EMSCRIPTEN_KEEPALIVE int alphasho_usi_command(const char* cmd) { YaneuraOu::alphasho_wasm_usi->usi_cmdexec(cmd); return 0; }\n#endif\n';p.write_text(s,'utf-8')
patch(SRC/'wasm_pre.js','"usi_command"','"alphasho_usi_command"')
patch(SRC/'Makefile','-s STRICT=1','')
# WASM's affine SIMD kernel consumes row-major weights. SSE4.2 implies SSSE3,
# whose native kernel scrambles them; do not apply that native-only permutation.
patch(SRC/'eval/nnue/layers/affine_transform.h',
 'static constexpr IndexType GetWeightIndex(IndexType i) {\n#if defined(USE_SSSE3) || defined(USE_NEON_DOTPROD)',
 'static constexpr IndexType GetWeightIndex(IndexType i) {\n#if !defined(USE_WASM_SIMD) && (defined(USE_SSSE3) || defined(USE_NEON_DOTPROD))')
# Use the native, parity-tested SSE integer kernels translated to WASM SIMD128.
# V9's separate experimental wasm affine kernel does not match native NNUE.
p=SRC/'Makefile';s=p.read_text('utf-8');p.write_text(s.replace('-DUSE_WASM_SIMD -msimd128','-msimd128'),'utf-8')
for old,new in [('PTHREAD_POOL_SIZE=32','PTHREAD_POOL_SIZE=2'),('STACK_SIZE=67108864','STACK_SIZE=8388608'),('MAXIMUM_MEMORY=4294967296','MAXIMUM_MEMORY=536870912'),('INITIAL_MEMORY=$(EM_INITIAL_MEMORY_SIZE)','INITIAL_MEMORY=201326592'),('-s --closure 1','-s EXPORT_ES6=1')]:patch(SRC/'Makefile',old,new)
env=os.environ.copy();env['EM_CONFIG']=str(SDK/'.emscripten');env['EMSDK']=str(SDK)
env['EMSDK_PYTHON']=sys.executable
env['PATH']=str(SDK/'upstream/emscripten')+os.pathsep+env['PATH']
compiler=(SDK/'upstream/emscripten'/('em++.bat' if os.name=='nt' else 'em++')).as_posix()
make=os.environ.get('NNUE_MAKE','make');OUT.mkdir(parents=True,exist_ok=True)
with (ROOT/'web/nnue-reports/wasm-build.log').open('w',encoding='utf-8') as log:
 subprocess.run([make,'normal','-j2','OBJDIR=../obj-wasm-sse','COMPILER='+compiler,'TARGET_CPU=WASM','YANEURAOU_EDITION=YANEURAOU_ENGINE_NNUE'],cwd=SRC,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
files={}
for p in SRC.glob('yaneuraou.*'):
 if p.suffix in ('.js','.wasm','.mjs'):
  shutil.copy2(p,OUT/p.name);files[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
shutil.copy2(SRC.parent/'Copying.txt',OUT/'COPYING.txt')
report=dict(source='https://github.com/yaneurao/YaneuraOu',commit=COMMIT,emsdk='3.1.74',files=files,time=time.time(),threads=1,pthreadPool=2,hashMiB=16,initialMemoryMiB=192,maxMemoryMiB=512,stackMiB=8)
(OUT/'build.json').write_text(json.dumps(report,indent=2),'utf-8');print(json.dumps(report))

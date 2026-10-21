"""Ship the full corresponding patched engine source beside the binary."""
import pathlib,tarfile,hashlib,json
ROOT=pathlib.Path(__file__).resolve().parents[2];SRC=ROOT/'.cache/yaneuraou-v900';OUT=ROOT/'web/public/nnue/engine'
with tarfile.open(OUT/'source.tar.gz','w:gz') as tar:
 for p in sorted(SRC.rglob('*')):
  if p.is_file() and not any(x.startswith('obj-') for x in p.parts) and p.suffix not in ('.o','.d','.js','.wasm','.mjs','.exe','.a'):
   # wasm_pre.js is required for rebuilding.
   tar.add(p,arcname='YaneuraOu-V9.00/'+p.relative_to(SRC).as_posix())
 for p in (SRC/'source').glob('wasm*.js'):tar.add(p,arcname='YaneuraOu-V9.00/source/'+p.name)
 tar.add(ROOT/'web/scripts/build-nnue.py',arcname='build-nnue.py')
report=json.loads((OUT/'build.json').read_text());report['correspondingSource']='source.tar.gz';report['sourceArchiveSha256']=hashlib.sha256((OUT/'source.tar.gz').read_bytes()).hexdigest()
(OUT/'build.json').write_text(json.dumps(report,indent=2),'utf8');print(json.dumps({'sourceBytes':(OUT/'source.tar.gz').stat().st_size,'sourceSha256':report['sourceArchiveSha256']}))

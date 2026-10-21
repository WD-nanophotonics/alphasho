import fs from 'node:fs';
import path from 'node:path';
import { webRoot } from './nnue-runtime';
import { renderKif } from './nnue-kif';
const argument=process.argv.indexOf('--output');const directory=argument>=0?path.resolve(process.argv[argument+1]!):path.join(webRoot,'nnue-reports/difficulty-acceptance');
for(const filename of fs.readdirSync(directory).filter(f=>/^L\d+-L\d+-\d+\.json$/.test(f))){
 const g=JSON.parse(fs.readFileSync(path.join(directory,filename),'utf8'));
 fs.writeFileSync(path.join(directory,filename.replace('.json','.kif')),renderKif(g));
}

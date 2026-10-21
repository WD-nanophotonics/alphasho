import fs from 'node:fs';
import path from 'node:path';
import { deflateRawSync } from 'node:zlib';
import { fileURLToPath } from 'node:url';
export function zipRecords(directory,destination=path.join(directory,'games.zip')) {
 const table=Array.from({length:256},(_,n)=>{for(let k=0;k<8;k++)n=n&1?0xedb88320^(n>>>1):n>>>1;return n>>>0;});
 const crc=bytes=>{let c=0xffffffff;for(const b of bytes)c=table[(c^b)&255]^(c>>>8);return(c^0xffffffff)>>>0;};
 const local=[],central=[];let offset=0,count=0;
 for(const name of fs.readdirSync(directory).filter(n=>/^L\d+-L\d+-\d+\.(json|kif)$/.test(n)).sort()){
  const raw=fs.readFileSync(path.join(directory,name)),compressed=deflateRawSync(raw),filename=Buffer.from(name),checksum=crc(raw),header=Buffer.alloc(30),entry=Buffer.alloc(46);
  header.writeUInt32LE(0x04034b50,0);header.writeUInt16LE(20,4);header.writeUInt16LE(0x800,6);header.writeUInt16LE(8,8);header.writeUInt16LE(33,12);header.writeUInt32LE(checksum,14);header.writeUInt32LE(compressed.length,18);header.writeUInt32LE(raw.length,22);header.writeUInt16LE(filename.length,26);
  entry.writeUInt32LE(0x02014b50,0);entry.writeUInt16LE(20,4);entry.writeUInt16LE(20,6);entry.writeUInt16LE(0x800,8);entry.writeUInt16LE(8,10);entry.writeUInt16LE(33,14);entry.writeUInt32LE(checksum,16);entry.writeUInt32LE(compressed.length,20);entry.writeUInt32LE(raw.length,24);entry.writeUInt16LE(filename.length,28);entry.writeUInt32LE(offset,42);
  local.push(header,filename,compressed);central.push(entry,filename);offset+=header.length+filename.length+compressed.length;count++;
 }
 const directoryBytes=Buffer.concat(central),end=Buffer.alloc(22);end.writeUInt32LE(0x06054b50);end.writeUInt16LE(count,8);end.writeUInt16LE(count,10);end.writeUInt32LE(directoryBytes.length,12);end.writeUInt32LE(offset,16);fs.writeFileSync(destination,Buffer.concat([...local,directoryBytes,end]));
}
if(process.argv[1]===fileURLToPath(import.meta.url))zipRecords(process.argv[2],process.argv[3]);

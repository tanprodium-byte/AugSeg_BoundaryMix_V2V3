"""Explicit private transfer archive outside Git; never connects/installs inputs or loads tensors."""
import argparse, csv, hashlib, json, os, tarfile, time
from pathlib import Path, PurePosixPath
from portable import C, load, sha_file, verify_package

def safe_member(name):
 p=PurePosixPath(name)
 return not p.is_absolute() and '..' not in p.parts and len(p.parts)>1 and p.parts[0] in ('augseg_inputs','augseg_diagnostic_inputs')

def create(destination):
 verify_package();destination=Path(destination).resolve()
 # Output must be new and outside every Git checkout (walk ancestors without inspecting secrets).
 assert not destination.exists(),'TRANSFER_DESTINATION_EXISTS'
 assert not any((q/'.git').exists() for q in [destination,*destination.parents]),'TRANSFER_MUST_BE_OUTSIDE_GIT'
 inp=load('inputs.json');files=[]
 for row in csv.DictReader(open(C/'input_payload_hashes.csv')):
  files.append((Path(inp['dataset_source_root'])/row['path'],'augseg_inputs/VOC2012/'+row['path'],int(row['size_bytes']),row['sha256']))
 for row in inp['splits']:
  rel='augseg_inputs/splits/pascal/'+('val.txt' if row['key']=='val' else row['key']+'/labeled.txt')
  files.append((Path(row['source_path']),rel,row['bytes'],row['sha256']))
 cp=inp['checkpoint'];files.append((Path(cp['source_path']),'augseg_diagnostic_inputs/s2_donor_pool/ckpt_best.pth',cp['bytes'],cp['sha256']))
 assert len({x[1] for x in files})==len(files)
 for source,rel,size,sha in files:
  assert safe_member(rel) and source.is_file()
  assert source.stat().st_size==size and sha_file(source)==sha,('SOURCE_CHECKSUM_MISMATCH',rel)
 total=sum(x[2] for x in files)
 import shutil
 assert shutil.disk_usage(destination.parent).free>=total+1024**3,'INSUFFICIENT_TRANSFER_DISK'
 destination.mkdir(parents=True,exist_ok=False);archive=destination/'s2_donor_pool_inputs.tar'
 manifest={rel:{'bytes':size,'sha256':sha} for source,rel,size,sha in files}
 with tarfile.open(archive,'w',dereference=True) as tf:
  for source,rel,size,sha in files:tf.add(source,arcname=rel,recursive=False)
 # Stream-check every archived payload, not just tar headers; no tensor materialization.
 with tarfile.open(archive,'r') as tf:
  members=tf.getmembers();assert len(members)==len(manifest)
  for m in members:
   assert m.isfile() and safe_member(m.name) and m.name in manifest
   h=hashlib.sha256()
   with tf.extractfile(m) as f:
    for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
   assert m.size==manifest[m.name]['bytes'] and h.hexdigest()==manifest[m.name]['sha256']
 (destination/'payload_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 receipt={'archive':str(archive),'bytes':archive.stat().st_size,'sha256':sha_file(archive),'files':len(manifest),'archive_payloads_stream_verified':True,'created_unix':time.time(),'scope':'diagnostic transfer bytes; not historical training input attestation','transferred_to_A6000':False,'no_credentials':True}
 (destination/'archive_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
 (destination/'SHA256SUMS').write_text(''.join(sha_file(destination/n)+'  '+n+'\n' for n in ['s2_donor_pool_inputs.tar','payload_manifest.json','archive_receipt.json']))
 return receipt
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);a=parser.parse_args();print(json.dumps(create(a.output),indent=2))

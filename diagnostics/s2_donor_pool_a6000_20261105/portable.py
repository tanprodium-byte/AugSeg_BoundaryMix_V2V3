"""Portable preparation/checks only. No credentials, remote calls or model imports."""
import argparse, ast, csv, hashlib, importlib.metadata, json, os, platform, re
import shutil, socket, subprocess, sys, tempfile, time
from pathlib import Path
C=Path(__file__).resolve().parent

def sha_file(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()

def load(name,base=C):return json.loads((base/name).read_text())
def output():return Path(os.environ.get('DIAGNOSTIC_OUTPUT',load('target.json')['output'])).resolve()
def verify_package(base=C):
 base=Path(base);manifest=load('package_manifest.json',base)
 actual={str(f.relative_to(base)) for f in base.rglob('*') if f.is_file() and f.name!='package_manifest.json'}
 assert actual==set(manifest),('PACKAGE_FILE_SET_MISMATCH',sorted(actual-set(manifest)),sorted(set(manifest)-actual))
 for rel,h in manifest.items():
  f=base/rel;assert not f.is_symlink() and sha_file(f)==h,('PACKAGE_CHECKSUM_MISMATCH',rel)
 plan=load('plan.json',base);old=load('historical/plan.json',base)
 allowed={'checkpoint','dataset_root','validation_list','source_training_manifest','source_training_config','input_payload_manifest_sha256'}
 assert {k for k in set(plan)|set(old) if plan.get(k)!=old.get(k)}<=allowed,'UNEXPLAINED_SCIENCE_CHANGE'
 for name,key in [('source_manifest.json','source_manifest_sha256'),('receiver_plan.csv','receiver_plan_sha256'),('donor_plan.csv','donor_plan_sha256'),('input_payload_hashes.csv','input_payload_manifest_sha256')]:assert sha_file(base/name)==plan[key],name
 assert sha_file(base/'augmented_input_hashes.csv')==load('augmentation_pin.json',base)['sha256']
 for rel,h in load('source_manifest.json',base).items():assert sha_file(base/'source'/rel)==h,rel
 assert load('scientific_equivalence.json',base)['science_changes']==[]
 inputs=load('inputs.json',base)
 assert plan['checkpoint']==inputs['checkpoint']['target_path'] and plan['dataset_root']==inputs['dataset_root']
 assert plan['validation_list']==next(x['target_path'] for x in inputs['splits'] if x['key']=='val')
 assert plan['batch']==16 and plan['diagnostic_batches']==64 and plan['crop']==[321,321]
 assert plan['resource_policy']['min_gpu_free_mib']==16384 and plan['resource_policy']['min_disk_free_bytes']==34359738368
 return {'package_files':len(manifest),'source_files':len(load('source_manifest.json',base)),'science_changes':0,'package_manifest_sha256':sha_file(base/'package_manifest.json')}

def validate_lists(base,inputs):
 lists={}
 for row in inputs['splits']:
  fp=Path(row['target_path']);assert fp.is_file(),('MISSING_SPLIT',row['key'],str(fp))
  assert fp.stat().st_size==row['bytes'] and sha_file(fp)==row['sha256'],('SPLIT_CHECKSUM_MISMATCH',row['key'])
  ids=fp.read_text().splitlines();assert len(ids)==row['lines'] and len(set(ids))==len(ids),('SPLIT_COUNT_OR_DUPLICATE',row['key'])
  assert all(re.fullmatch(r'[A-Za-z0-9_-]+',x) for x in ids),'UNSAFE_IMAGE_ID'
  lists[row['key']]=set(ids)
 assert lists['662']<=lists['1323']<=lists['2646'],'DONOR_LISTS_NOT_NESTED'
 assert not lists['val']&lists['2646'],'DONOR_VALIDATION_OVERLAP'
 rx=list(csv.DictReader(open(base/'receiver_plan.csv')));donors=list(csv.DictReader(open(base/'donor_plan.csv')))
 assert len(rx)==1024 and len({x['id'] for x in rx})==1024
 assert {x['id'] for x in rx}<=lists['val']
 assert len(donors)==3072
 for split in ['662','1323','2646']:
  ds=[x for x in donors if x['split']==split];assert len(ds)==1024 and {x['id'] for x in ds}<=lists[split]
  assert len({x['id'] for x in ds})==load('plan.json',base)['unique_donors'][split]
 for rows in [rx]+[[x for x in donors if x['split']==s] for s in ['662','1323','2646']]:
  assert all(sum(int(x['batch'])==b for x in rows)==16 for b in range(64))
 return lists

def check_inputs(base=C):
 base=Path(base);inputs=load('inputs.json',base);lists=validate_lists(base,inputs);root=Path(inputs['dataset_root'])
 entries=list(csv.DictReader(open(base/'input_payload_hashes.csv')));found=set();total=0
 for z in entries:
  rel=Path(z['path']);assert not rel.is_absolute() and '..' not in rel.parts,'UNSAFE_PAYLOAD_PATH'
  expected=('JPEGImages/'+z['id']+'.jpg') if z['kind']=='image' else ('SegmentationClassAug/'+z['id']+'.png')
  assert str(rel)==expected and z['kind'] in ('image','mask'),'LOADER_PATH_MISMATCH'
  fp=root/rel;assert fp.is_file(),('MISSING_PAYLOAD',str(rel))
  assert fp.stat().st_size==int(z['size_bytes']) and sha_file(fp)==z['sha256'],('PAYLOAD_CHECKSUM_MISMATCH',str(rel))
  assert (z['id'],z['kind']) not in found,'DUPLICATE_PAYLOAD_ENTRY'
  found.add((z['id'],z['kind']));total+=fp.stat().st_size
 # Full validation (1449), plus every scheduled donor; no need to transfer unused train images.
 donors={x['id'] for x in csv.DictReader(open(base/'donor_plan.csv'))};required=lists['val']|donors
 assert found=={(id_,k) for id_ in required for k in ['image','mask']},'INCOMPLETE_REQUIRED_IMAGE_MASK_MANIFEST'
 cp=inputs['checkpoint'];fp=Path(cp['target_path']);assert fp.is_file(),'MISSING_DESIGNATED_CHECKPOINT'
 assert fp.stat().st_size==cp['bytes'] and sha_file(fp)==cp['sha256'],'CHECKPOINT_CHECKSUM_MISMATCH'
 assert cp['sha256']==load('checkpoint_pin.json',base)['checkpoint_sha256']
 return {'input_sha256_checks_passed':True,'required_images':len(required),'required_masks':len(required),'dataset_bytes':total,'checkpoint_bytes':cp['bytes'],'checkpoint_tensor_load':False,'historical_training_input_attestation':'UNKNOWN; current transfer bytes only'}

def dependencies(base=C):
 expected=load('augmentation_pin.json',base);actual={}
 for name in ['torch','torchvision','Pillow','numpy','scipy','scikit-image','opencv-python','PyYAML']:
  try:actual[name]=importlib.metadata.version(name)
  except importlib.metadata.PackageNotFoundError:
   if name=='opencv-python':actual[name]=importlib.metadata.version('opencv-python-headless')
   else:raise
 for key in ['torch','torchvision','Pillow','numpy']:
  assert actual[key]==expected[key],('AUGMENTATION_RUNTIME_MISMATCH',key,actual[key],expected[key])
 return {'python':sys.version,'packages_current_metadata':actual,'not_historical_loaded_libraries':True,'CUDA_context_created':False}

def require_target():
 t=load('target.json');import getpass
 assert socket.gethostname()==t['host'] and getpass.getuser()==t['user'],'WRONG_HOST: real-model actions only on islab-server2 / islabworker2'
 assert os.environ.get('CUDA_VISIBLE_DEVICES') in (None,'','0'),'GPU_ROUTING_MISMATCH: do not remap GPU0'
 # Empty CUDA visibility disables GPUs; fail clearly rather than override caller environment.
 assert os.environ.get('CUDA_VISIBLE_DEVICES')!='','CUDA_VISIBLE_DEVICES disables GPU0'

def require_prepared(base,path):
 expected=sha_file(base/'package_manifest.json');receipt=load('preparation.json',path)
 assert receipt['package_manifest_sha256']==expected,'PREPARATION_PACKAGE_MISMATCH'
 assert path.resolve()!=base.resolve() and not path.resolve().is_relative_to(base.resolve()),'OUTPUT_MUST_BE_OUTSIDE_CODE'

def prepare():
 require_target();checks=verify_package();p=output();assert p==Path(load('target.json')['output']),'OUTPUT_OVERRIDE_NOT_ALLOWED_FOR_TARGET_PREPARATION'
 assert not p.exists(),'OUTPUT_EXISTS: inspect; never overwrite or create automatic attempt'
 p.mkdir(parents=True,exist_ok=False)
 (p/'preparation.json').write_text(json.dumps(checks|{'state':'PREPARED_NOT_RUN','created_unix':time.time(),'code_path':str(C)},indent=2)+'\n')
 return {'prepared_output':str(p),'model_run':False}

def cpu_tests():
 checks=verify_package()
 # Every test writes only a fresh temporary directory, never production inputs/state.
 with tempfile.TemporaryDirectory(prefix='augseg-donor-cpu-') as td:
  env=dict(os.environ,DIAGNOSTIC_OUTPUT=td,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2')
  for name in ['test_cpu.py','test_portable.py']:
   subprocess.run([sys.executable,'-B',str(C/name)],env=env,check=True,cwd=td)
  result=load('cpu_tests.json',Path(td));extra=load('portable_tests.json',Path(td))
  result.update(portable_tests=extra,package_manifest_sha256=checks['package_manifest_sha256'],total_tests=result['tests_run']+extra['tests_run'])
 # Only explicit A6000 prepare + target-side tests can write the prerequisite runtime receipt.
 p=output()
 if socket.gethostname()==load('target.json')['host'] and p.exists():
  require_target();require_prepared(C,p)
  assert not any((p/n).exists() for n in ['status.json','batch_counts.csv','validation.json']),'EXECUTION_ALREADY_STARTED: do not overwrite test attestation'
  (p/'cpu_tests.json').write_text(json.dumps(result,indent=2)+'\n')
 return result

def main():
 parser=argparse.ArgumentParser();parser.add_argument('action',choices=['check-package','check-inputs','check-deps','test','prepare']);args=parser.parse_args()
 funcs={'check-package':verify_package,'check-inputs':lambda:(verify_package(),check_inputs())[1],'check-deps':dependencies,'test':cpu_tests,'prepare':prepare}
 print(json.dumps(funcs[args.action](),indent=2))
if __name__=='__main__':main()

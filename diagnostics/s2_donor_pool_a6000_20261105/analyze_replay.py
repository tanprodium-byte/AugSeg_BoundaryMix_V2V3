"""Read diagnostic outputs, aggregate/paired-bootstrap, report+ZIP; NEVER calls runner/model."""
import csv,datetime,hashlib,json,math,os,zipfile
from collections import defaultdict
from pathlib import Path
import numpy as np
C=Path(__file__).resolve().parent
P=Path(os.environ.get('DIAGNOSTIC_OUTPUT',json.loads((C/'target.json').read_text())['output'])).resolve()
NAMES=['background','aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']
def put(n,x):(P/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def table(n,rows):
 if not rows:return
 keys=list(dict.fromkeys(k for r in rows for k in r))
 with (P/n).open('w',newline='') as out:w=csv.DictWriter(out,fieldnames=keys);w.writeheader();w.writerows(rows)
def mdtable(rows):
 keys=list(rows[0]);return '| '+' | '.join(keys)+' |\n|'+'|'.join('---' for k in keys)+'|\n'+'\n'.join('| '+' | '.join(str(r.get(k,'')).replace('|','/') for k in keys)+' |' for r in rows)
def ratio(a,b):return a/b if b else 'NA'
def read(name):return list(csv.DictReader(open(P/name)))
def hfile(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
from portable import verify_package
verify_package(C)
plan=json.loads((C/'plan.json').read_text());status=json.loads((P/'status.json').read_text()) if (P/'status.json').exists() else {'state':'NOT_RUN'};binding=json.loads((P/'checkpoint_binding.json').read_text()) if (P/'checkpoint_binding.json').exists() else json.loads((C/'historical/checkpoint_binding.json').read_text());tests=json.loads((P/'cpu_tests.json').read_text());pre=json.loads((P/'preflight.json').read_text()) if (P/'preflight.json').exists() else {};ready=status['state']=='COMPLETED'
profiles=[{'donor_pool':s,'recipe':r,'planned_receiver_images':1024,'planned_batches':64,'planned_batch':16,'planned_donor_slots':1024,'planned_unique_donors':plan['unique_donors'][str(s)],'observed_completed_batches':64 if ready else 'NOT_RUN_OR_INCOMPLETE','status':status['state']} for s in [662,1323,2646] for r in ['baseline','S2']];table('six_conditions.csv',profiles)
results_text='';pairrows=[];poolrows=[]
if ready:
 val=json.loads((P/'validation.json').read_text());assert val['passed'];assert json.loads((P/'teacher_state_before.json').read_text())==json.loads((P/'teacher_state_after.json').read_text())
 batch=read('batch_counts.csv');cl=read('class_batch_counts.csv');assert len(batch)==384 and len(cl)==8064
 counts=['reference_valid','gt_valid','correct_pseudo_active','wrong_pseudo_active','valid_filtered','gt_ignore','reference_ignore','pseudo_reference_ignore_active','pseudo_reference_ignore_filtered','active_loss','original_receiver_wrong_active_removed','original_receiver_correct_active_removed']
 ag=[];classag=[];grid={};mat={};fields=['gt_valid','correct_pseudo_active','wrong_pseudo_active','valid_filtered','active_supervision_target_class']
 for split in [662,1323,2646]:
  for recipe in ['baseline','S2']:
   key=(split,recipe);br=sorted([x for x in batch if int(x['split'])==split and x['recipe']==recipe],key=lambda r:int(r['batch']));assert [int(x['batch']) for x in br]==list(range(64));n=sum(int(x['grid_pixels']) for x in br);grid[key]=n
   z={'split':split,'recipe':recipe,'batches':64,'grid_pixels':n,**{k:sum(int(x[k]) for x in br) for k in counts}}
   for k in ['gt_valid','gt_ignore','reference_ignore','active_loss','correct_pseudo_active','wrong_pseudo_active','valid_filtered']:z[k+'_over_grid']=ratio(z[k],n)
   z['active_pseudo_error_rate']=ratio(z['wrong_pseudo_active'],z['correct_pseudo_active']+z['wrong_pseudo_active']);ag.append(z)
   arr=np.zeros((64,21,len(fields)),dtype=np.int64)
   for cid in range(21):
    rows=sorted([x for x in cl if int(x['split'])==split and x['recipe']==recipe and int(x['class_id'])==cid],key=lambda r:int(r['batch']));assert len(rows)==64
    totalfields=['reference_valid','gt_valid','correct_pseudo_active','wrong_pseudo_active','valid_filtered','pseudo_reference_valid','active_supervision_target_class','unknown_reference_active_target_class','original_receiver_wrong_active_removed','original_receiver_correct_active_removed']
    cz={'split':split,'recipe':recipe,'class_id':cid,'class_name':NAMES[cid],'grid_pixels':n,**{k:sum(int(x[k]) for x in rows) for k in totalfields}}
    for k in fields:cz[k+'_over_grid']=ratio(cz[k],n)
    cz['gt_coverage_in_reference_class']=ratio(cz['gt_valid'],cz['reference_valid']);cz['active_pseudo_coverage']=ratio(cz['correct_pseudo_active']+cz['wrong_pseudo_active'],cz['pseudo_reference_valid']);cz['active_pseudo_error_rate']=ratio(cz['wrong_pseudo_active'],cz['correct_pseudo_active']+cz['wrong_pseudo_active']);classag.append(cz)
    arr[:,cid,:]=np.array([[int(x[k]) for k in fields] for x in rows])
   mat[key]=arr
   for k in ['reference_valid','gt_valid','correct_pseudo_active','wrong_pseudo_active','valid_filtered']:assert sum(x[k] for x in classag if x['split']==split and x['recipe']==recipe)==z[k]
 table('condition_totals.csv',ag);table('class_totals.csv',classag)
 gen=np.random.default_rng(20261106);idx=gen.integers(0,64,size=(2000,64));effects={}
 for split in [662,1323,2646]:
  dif=mat[(split,'S2')]-mat[(split,'baseline')];effects[split]=dif
  denom=grid[(split,'S2')];boot=(dif[idx].sum(1)/denom*100);ci=np.quantile(boot,[.025,.975],axis=0)
  for cid in range(21):
   z={'split':split,'class_id':cid,'class_name':NAMES[cid]}
   for j,f in enumerate(fields):z['delta_'+f+'_count']=int(dif[:,cid,j].sum());z['delta_'+f+'_mass_pp']=float(dif[:,cid,j].sum()/denom*100);z[f+'_batch_bootstrap_low_pp']=float(ci[0,cid,j]);z[f+'_batch_bootstrap_high_pp']=float(ci[1,cid,j])
   a=next(x for x in classag if x['split']==split and x['recipe']=='S2' and x['class_id']==cid);b=next(x for x in classag if x['split']==split and x['recipe']=='baseline' and x['class_id']==cid)
   for f in ['gt_coverage_in_reference_class','active_pseudo_coverage','active_pseudo_error_rate']:z['delta_'+f+'_pp']='NA' if 'NA' in [a[f],b[f]] else (a[f]-b[f])*100
   pairrows.append(z)
 for a,b in [(1323,662),(2646,662),(2646,1323)]:
  d=effects[a]-effects[b];denom=grid[(a,'S2')];boot=d[idx].sum(1)/denom*100;ci=np.quantile(boot,[.025,.975],axis=0)
  for cid in range(21):
   z={'pool_difference':f'{a}-{b}','class_id':cid,'class_name':NAMES[cid]}
   for j,f in enumerate(fields):z['difference_of_recipe_deltas_'+f+'_pp']=float(d[:,cid,j].sum()/denom*100);z[f+'_batch_bootstrap_low_pp']=float(ci[0,cid,j]);z[f+'_batch_bootstrap_high_pp']=float(ci[1,cid,j])
   poolrows.append(z)
 table('paired_recipe_deltas.csv',pairrows);table('paired_pool_differences.csv',poolrows)
 checks=read('input_condition_checks.csv');assert len(checks)==384
 for bi in range(64):
  rows=[r for r in checks if int(r['batch'])==bi]
  for key in ['receiver_weak_sha256','receiver_strong_sha256','receiver_GT_reference_sha256','cached_q_sha256','cached_pseudo_sha256']:assert len({r[key] for r in rows})==1,(bi,key)
  for sp in [662,1323,2646]:
   rows2=[r for r in rows if int(r['split'])==sp]
   for key in ['donor_RGB_sha256','donor_GT_sha256']:assert len({r[key] for r in rows2})==1
 put('analysis_checks.json',{'all6cells64batches':True,'teacher_hash_unchanged':True,'common_inputs_match':True,'count_conservation':True,'batch_bootstrap_replicates':2000,'bootstrap_seed':20261106,'uncertainty_scope':'paired batches; not pixels or training seeds','real_model_diagnostic_completed':True})
 priority=[1,4,5,11,8,12,17,16]
 rows=[]
 for cid in priority:
  z={'class':NAMES[cid]}
  for s in [662,1323,2646]:
   t=next(x for x in pairrows if x['split']==s and x['class_id']==cid);z[str(s)+' ΔGT / Δcorrectpseudo / Δwrongpseudo pp']=' / '.join(f'{t["delta_"+f+"_mass_pp"]:+.5f}' for f in ['gt_valid','correct_pseudo_active','wrong_pseudo_active'])
  rows.append(z)
 results_text='## Kết quả đo thật\n\n'+mdtable(rows)+'\n\nĐủ21class nằm ở class_totals.csv/paired_recipe_deltas.csv; difference-of-deltas giữa pools ởpaired_pool_differences.csv. Không chỉ đọc wrong-pseudo reduction: phải đối chiếu correct-pseudo loss và GT mass cùng hàng. Conditional error rates thay đổi cả composition, không tự là ranking quality.\n'
else:
 results_text='''## Kết quả đo thật: CHƯA CÓ

Không có per-class/per-batch replay observations để kết luận donor pool làm thay đổi supervision của class nào. Không tạo zero-filled metrics. Không có bằng chứng mới để xác nhận hoặc bác bỏ lời giải regression nhẹ hơn ởVOC2646. CPU synthetic examples là kiểm chứng runner, không phải kết quả trên frozen baseline teacher.
'''
now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=7))).isoformat();resource=status.get('resources',pre.get('resources',{}));testn=tests['tests_run']
report=f'''# S2 donor-pool controlled replay — {status['state']}

Report time (Vietnam): {now}. Target binding recorded here: {(P/'checkpoint_binding.json').exists()}.
Historical CPU binding is evidence of checkpoint identity, not target validation or replay.
Package manifest SHA256: {hfile(C/'package_manifest.json')}.
Source commit: {plan['source_commit']}; teacher: {plan['teacher_execution_id']}, epoch70.
Checkpoint SHA256: {binding['checkpoint_sha256']}.

## Design and status
FP32, no AMP/TF32, C321, B16,64 shared receiver batches,1024 fixed held-out images.
Three nested donor pools, uniform with replacement; baseline two-stage versus S2 one-stage
exact-mask relocated saliency×area selection. No optimizer/EMA, training entrypoint, network,
W&B run or HF upload. Teacher parameters/buffers must match before/after.
Actual validation must match75.20909874915435% within0.0001pp before replay; never relax automatically.
Resource readings are timestamped observations, not a future capacity guarantee: {resource}.
{mdtable(profiles)}

{results_text}
## Definitions and interpretation
Raw per-class counts use reference/content class, not mean class ID.
N=BHW. Mass=count/N; conditional ratios use their explicit denominators; denominator0 is NA.
Valid reference=valid GT-origin+correct active pseudo+wrong active pseudo+valid filtered.
Ignored reference=GT-ignore+pseudo reference-ignore active+pseudo reference-ignore filtered.
Active CE=valid GT-origin+correct active pseudo+wrong active pseudo+pseudo reference-ignore active.
Receiver GT never determines predictions/gate/selection. Baseline carries provenance through both stages.
Removed original-receiver supervision is not a corrected prediction of the same content.
Read all21 classes, including losses of correct pseudo supervision and gains of GT supervision.
Paired recipe deltas and difference-of-deltas use the same64 receiver batches; bootstrap units
are batches, not pixels or training seeds. These are allocation measurements, not historical
training masks, donor repetition effects, learning utility or proof of the mIoU mechanism.

## Commands (no automatic restart)
```bash
bash {C}/run.sh preflight
bash {C}/run.sh run
bash {C}/run.sh analyze
```
A completed replay is not rerun. Partial observations block automatic retry.
Analyze is independent and never invokes model/replay/training. Report/CSV/JSON ZIP excludes inputs.
'''
(P/'REPORT.md').write_text(report)
put('artifact_manifest.json',{'status':status['state'],'checkpoint_sha256':binding['checkpoint_sha256'],'plan_sha256':hfile(C/'plan.json'),'source_manifest_sha256':hfile(C/'source_manifest.json'),'real_model_diagnostic_completed':ready,'report_timestamp_vietnam':now,'production_changed':False,'no_optimizer_or_EMA':True})
# Package only report/metadata/CSVs, never checkpoint/cache/tensors/images/secrets.
import shutil
for name in ['plan.json','recipe.json','source_manifest.json','source_origins.json','checkpoint_pin.json','inputs.json','target.json','scientific_equivalence.json','package_manifest.json']:
 shutil.copyfile(C/name,P/name)
allow=[]
for f in sorted(P.iterdir()):
 if f.is_file() and f.suffix in ['.md','.csv','.json','.py','.sh'] and f.name not in ['zip_receipt.json','SHA256.json']:allow.append(f)
sha={f.name:hfile(f) for f in allow};put('SHA256.json',sha);allow.append(P/'SHA256.json')
zipname=P.parent/(P.name+'.zip')
with zipfile.ZipFile(zipname,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for f in allow:z.write(f,arcname=P.name+'/'+f.name)
with zipfile.ZipFile(zipname) as z:
 assert z.testzip() is None
 for name in z.namelist():assert not name.startswith('/') and '..' not in Path(name).parts
 for f in allow:assert hashlib.sha256(z.read(P.name+'/'+f.name)).hexdigest()==hfile(f)
receipt={'path':str(zipname),'bytes':zipname.stat().st_size,'sha256':hfile(zipname),'entries':len(allow),'CRC_and_all_payload_hashes_verified':True,'contains_checkpoint_or_images_or_credentials':False};put('zip_receipt.json',receipt);print(json.dumps(receipt,indent=2))

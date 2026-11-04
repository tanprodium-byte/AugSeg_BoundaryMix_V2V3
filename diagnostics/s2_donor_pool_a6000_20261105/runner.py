"""Frozen-teacher validation + paired donor-pool replay; no training entrypoint."""
import argparse,copy,csv,fcntl,gc,json,os,sys,time,traceback
from pathlib import Path
import numpy as np
from PIL import Image
from common import *
from checkpoint_binding import verify as verify_checkpoint

def resource_gate(resources):
 if not resources['ready']:raise RuntimeError('BLOCKED_RESOURCE: GPU free or disk guard not met')

def load_image(id_,root):
 with Image.open(root/'JPEGImages'/(id_+'.jpg')) as im:rgb=im.convert('RGB').copy()
 with Image.open(root/'SegmentationClassAug'/(id_+'.png')) as im:gt=im.copy()
 return rgb,gt

def validate_teacher(model,plan,cfg,device):
 # Historical val recipe: native-size RGB, no resize/crop/flip, batch1,
 # same normalization, no auxiliary head,21-class macro IoU and ignore255.
 from augseg.dataset.pascal_voc import build_basic_transfrom
 from augseg.dataset.augs_TIBA import ToTensorAndNormalize
 from augseg.utils.utils import intersectionAndUnion
 d=copy.deepcopy(cfg['dataset']);d.update(d['val']);assert 'crop' not in d and 'rand_resize' not in d
 transform=build_basic_transfrom(d,split='val',mean=d['mean']);normalize=ToTensorAndNormalize(d['mean'],d['std']);root=Path(plan['dataset_root']);ids=Path(plan['validation_list']).read_text().splitlines();assert len(ids)==1449
 inter=np.zeros(21,dtype=np.int64);union=np.zeros(21,dtype=np.int64);start=time.time()
 with torch.no_grad():
  for i,id_ in enumerate(ids):
   rgb,gt=load_image(id_,root);rgb,gt=transform(rgb,gt);x,y=normalize(rgb,gt);pred,_=model(x[None].to(device,dtype=torch.float32));label=pred.argmax(1)[0].cpu().numpy();ii,uu,_=intersectionAndUnion(label,y.numpy(),21,255);inter+=ii;union+=uu
   if (i+1)%100==0:print('validation',i+1,'/1449',round(time.time()-start,1),'s',flush=True)
 iou=inter/(union+1e-10);miou=float(iou.mean()*100);diff=miou-plan['expected_teacher_miou_percent'];result={'passed':abs(diff)<=plan['validation_abs_tolerance_pp'],'recipe':'historical native validation, batch1; crop321 applies only replay','images':len(ids),'teacher_miou_percent':miou,'expected_percent':plan['expected_teacher_miou_percent'],'difference_pp':diff,'tolerance_pp_predeclared':plan['validation_abs_tolerance_pp'],'class_iou_percent':(iou*100).tolist(),'class_intersections':inter.tolist(),'class_unions':union.tolist(),'seconds':time.time()-start}
 put('validation.json',result)
 if not result['passed']:raise RuntimeError('VALIDATION_MISMATCH: stop; never switch checkpoint or recipe')
 return result

def augment_batch(planrows,cfg,root,is_receiver):
 from augseg.dataset.pascal_voc import build_basic_transfrom,build_additional_strong_transform
 from augseg.dataset.augs_TIBA import ToTensorAndNormalize
 d=copy.deepcopy(cfg['dataset']);d.update(d['train']);weaktr=build_basic_transfrom(d,split='train',mean=d['mean']);strongtr=build_additional_strong_transform(d);normalize=ToTensorAndNormalize(d['mean'],d['std']);xs=[];strong=[];ys=[]
 for row in planrows:
  with preserve_cpu_rng(int(row['augmentation_seed'])):
   rgb,gt=load_image(row['id'],root);rgb,gt=weaktr(rgb,gt)
   if is_receiver:strongrgb,_,_=strongtr(rgb.copy(),return_info=True)
   x,y=normalize(rgb,gt);xs.append(x);ys.append(y)
   if is_receiver:strong.append(normalize(strongrgb,gt)[0])
 return torch.stack(xs),torch.stack(strong) if is_receiver else None,torch.stack(ys)

def run():
 plan=json.loads((C/'plan.json').read_text());cfg=json.loads((C/'recipe.json').read_text())
 if (P/'status.json').exists() and json.loads((P/'status.json').read_text()).get('state')=='COMPLETED':
  print('Already completed; no replay',flush=True);return 0
 if any((P/f).exists() for f in ['batch_counts.csv','class_batch_counts.csv','pseudo_confusion.csv']):raise RuntimeError('EXISTING_DIAGNOSTIC_OBSERVATIONS: no automatic replay/retry')
 put('status.json',{'state':'CHECKING','started_unix':time.time()})
 check_pins(check_inputs=True);r=resource_check();put('resources.json',r)
 if not r['ready']:
  put('status.json',{'state':'BLOCKED_RESOURCE','GPU_model_diagnostic_started':False,'resources':r});return 2
 resource_gate(r)
 for f in ['batch_counts.csv','class_batch_counts.csv','pseudo_confusion.csv']:
  if (P/f).exists():raise RuntimeError('EXISTING_DIAGNOSTIC_OBSERVATIONS: no automatic replay/retry')
 if not (P/'cpu_tests.json').is_file() or not json.loads((P/'cpu_tests.json').read_text())['passed']:raise RuntimeError('CPU_TESTS_REQUIRED')
 assert json.loads((P/'cpu_tests.json').read_text())['package_manifest_sha256']==sha_file(C/'package_manifest.json'),'CPU_TEST_PACKAGE_MISMATCH'
 binding=verify_checkpoint();expected_sha=binding['checkpoint_sha256'];assert sha_file(plan['checkpoint'])==expected_sha
 # No CUDA initialization/model import before the resource gate.
 torch.cuda.set_device(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.set_num_threads(2);seed_cpu(2)
 from augseg.models.model_helper import ModelBuilder
 net=copy.deepcopy(cfg['net']);net['encoder']['pretrain']=None;net['encoder']['kwargs']['pretrained']=False
 model=ModelBuilder(net).float();ckpt=torch.load(plan['checkpoint'],map_location='cpu',weights_only=True);model.load_state_dict(ckpt['teacher_state'],strict=True,assign=True);del ckpt;gc.collect();model.eval()
 for par in model.parameters():par.requires_grad_(False)
 device=torch.device('cuda:0');model.to(device);before=model_hashes(model)
 put('teacher_state_before.json',before);put('runtime.json',{'torch':torch.__version__,'cuda_runtime':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),'device':torch.cuda.get_device_name(0),'FP32':True,'AMP':False,'TF32_matmul':False,'TF32_cudnn':False,'deterministic_cudnn':True,'skip_pretrain_constructor_only':'pretrained=False; strict designated teacher_state overwrites all state','optimizer_created':False,'EMA_update':False})
 try:
  put('status.json',{'state':'VALIDATING','checkpoint_sha256':expected_sha});validate_teacher(model,plan,cfg,device);assert model_hashes(model)==before
  expected_aug={(int(r['batch']),r['role'],r['pool']):r for r in csv.DictReader(open(C/'augmented_input_hashes.csv'))}
  rx=list(csv.DictReader(open(C/'receiver_plan.csv')));donors=list(csv.DictReader(open(C/'donor_plan.csv')));root=Path(plan['dataset_root']);put('status.json',{'state':'REPLAYING','next_batch':0})
  for bi in range(64):
   # Before every batch, preserve existing GPU/disk policy; never kill another job.
   rr=resource_check()
   # During our own model allocation free VRAM differs; use current free+our
   # reserved memory solely to account for this diagnostic, not other processes.
   rr['own_reserved_mib']=torch.cuda.memory_reserved(0)//1024**2
   rr['gpu_pass_accounting_own_allocation']=rr['gpu_free_mib']+rr['own_reserved_mib']>=plan['resource_policy']['min_gpu_free_mib']
   if not rr['disk_pass'] or not rr['gpu_pass_accounting_own_allocation']:raise RuntimeError('RESOURCE_CHANGED: hold diagnostic; no batch downgrade')
   rows=[x for x in rx if int(x['batch'])==bi];weak,strong,reference=augment_batch(rows,cfg,root,True);assert weak.shape==(16,3,321,321)
   receiver_hash={'weak':tensor_sha(weak),'strong':tensor_sha(strong),'reference':tensor_sha(reference)}
   expected=expected_aug[bi,'receiver','common'];assert receiver_hash=={'weak':expected['weak_RGB_sha256'],'strong':expected['strong_RGB_sha256'],'reference':expected['GT_sha256']},'Receiver preprocessing drift'
   weak=weak.to(device);strong=strong.to(device);reference=reference.to(device)
   with torch.no_grad():
    logits,_=model(weak);prob=torch.softmax(logits,1);q,pseudo=prob.max(1);h=-(prob*torch.log(prob+1e-10)).sum(1)/np.log(21);gate=(q*(1-h)).mean((1,2)).cpu().tolist();assert bool(torch.isfinite(q).all()) and all(0<=v<=1 for v in gate),'nonfinite/invalid teacher score';del logits,prob,h
   # One weak teacher forward per shared receiver batch, cached for all6 cells.
   cached_hash={'q':tensor_sha(q),'pseudo':tensor_sha(pseudo)}
   for split in [662,1323,2646]:
    ds=[x for x in donors if int(x['split'])==split and int(x['batch'])==bi];lx,_,ly=augment_batch(ds,cfg,root,False);donor_hash={'RGB':tensor_sha(lx),'GT':tensor_sha(ly)};expected=expected_aug[bi,'donor',str(split)];assert donor_hash=={'RGB':expected['weak_RGB_sha256'],'GT':expected['GT_sha256']},'Donor preprocessing drift';lx=lx.to(device);ly=ly.to(device)
    for recipe in ['baseline','S2']:
     with preserve_cpu_rng(plan['mix_seed_base']+bi):
      masks=None;stats={};candidate=[];selected=[]
      if recipe=='S2':
       try:masks,stats,candidate,selected=select_masks(model,lx,ly)
       except Exception as exc:
        put('selector_exception.json',{'split':split,'batch':bi,'exception_type':type(exc).__name__,'message':str(exc),'no_component_fallback_is_separate':True,'action':'abort controlled comparison; do not aggregate legacy two-stage exception fallback as S2'});raise
      mixed=mix(recipe,strong,pseudo,q,lx,ly,gate,reference,masks=masks,batch_index=bi)
     glob,cls,conf=counts(mixed,pseudo,q,reference);prefix={'split':split,'recipe':recipe,'batch':bi};glob=prefix|glob
     if recipe=='S2':glob.update(selector_exception_count=0,no_component_fallback_images=sum(bool(x['no_component_fallback']) for x in selected),eligible_components=sum(x['candidate_count'] for x in selected))
     else:glob.update(selector_exception_count=0,no_component_fallback_images='NA',eligible_components='NA')
     append_csv('mix_trace.csv',[prefix|z for z in mixed['trace']]);append_csv('batch_counts.csv',[glob]);append_csv('class_batch_counts.csv',[prefix|x for x in cls]);append_csv('pseudo_confusion.csv',[prefix|{'reference_class':i,'assigned_class':j,'count':int(conf[i][j])} for i in range(21) for j in range(21) if conf[i][j]])
     if candidate:append_csv('component_candidates.csv',[prefix|x for x in candidate])
     if selected:append_csv('component_selected.csv',[prefix|x for x in selected])
     append_csv('input_condition_checks.csv',[prefix|{'receiver_weak_sha256':receiver_hash['weak'],'receiver_strong_sha256':receiver_hash['strong'],'receiver_GT_reference_sha256':receiver_hash['reference'],'donor_RGB_sha256':donor_hash['RGB'],'donor_GT_sha256':donor_hash['GT'],'cached_q_sha256':cached_hash['q'],'cached_pseudo_sha256':cached_hash['pseudo'],'mix_seed':plan['mix_seed_base']+bi}])
     del mixed,masks
    del lx,ly
   assert tensor_sha(q)==cached_hash['q'] and tensor_sha(pseudo)==cached_hash['pseudo'];assert tensor_sha(weak)==receiver_hash['weak'] and tensor_sha(strong)==receiver_hash['strong'] and tensor_sha(reference)==receiver_hash['reference']
   assert all(not m.training for m in model.modules()) and all(not p.requires_grad and p.grad is None for p in model.parameters())
   del weak,strong,reference,q,pseudo;torch.cuda.empty_cache();put('status.json',{'state':'REPLAYING','completed_batches':bi+1,'conditions_per_batch':6});print('replay',bi+1,'/64 ×6',flush=True)
  after=model_hashes(model);put('teacher_state_after.json',after);assert after==before
  put('status.json',{'state':'COMPLETED','receiver_batches':64,'conditions':6,'weak_teacher_forwards':64,'saliency_batch_probes':192,'teacher_parameters_buffers_unchanged':True,'no_optimizer_no_EMA':True,'completed_unix':time.time()});return 0
 finally:
  after=model_hashes(model);put('teacher_state_exit.json',after);assert after==before,'Teacher parameter/buffer mutation detected'

def main():
 from portable import require_target, require_prepared, dependencies
 require_target();require_prepared(C,P);dependencies()
 parser=argparse.ArgumentParser();parser.add_argument('action',choices=['preflight','run']);args=parser.parse_args()
 if args.action=='preflight':
  pins=check_pins(check_inputs=True);r=resource_check();put('preflight.json',{'pins':pins,'resources':r,'model_started':False});print(json.dumps({'pins':pins,'resources':r},indent=2));return 0 if r['ready'] else 2
 with (P/'diagnostic.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  try:return run()
  except Exception as exc:put('status.json',{'state':'FAILED_DIAGNOSTIC','error_type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc(),'automatic_retry':False});raise
if __name__=='__main__':raise SystemExit(main())

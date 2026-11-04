"""CPU-only verification of the exact designated checkpoint, never select alternatives."""
import gc,json
from pathlib import Path
from common import C,P,sha_file,state_sha,put,torch

def verify(create_pin=False):
 plan=json.loads((C/'plan.json').read_text());cp=Path(plan['checkpoint']);sha=sha_file(cp)
 if (P/'checkpoint_binding.json').exists():assert json.loads((P/'checkpoint_binding.json').read_text())['checkpoint_sha256']==sha,'designated checkpoint bytes changed'
 ckpt=torch.load(cp,map_location='cpu',weights_only=True)
 assert ckpt['run_id']==plan['teacher_execution_id'],'checkpoint execution mismatch'
 assert ckpt['epoch']==plan['expected_checkpoint_next_epoch'],'checkpoint epoch is not designated best'
 assert 'teacher_state' in ckpt and ckpt['teacher_state'],'teacher_state missing; never substitute student'
 assert abs(float(ckpt['best_miou'])*100-plan['expected_teacher_miou_percent'])<1e-10
 manifest=ckpt['extra']['manifest'];assert manifest['run_id']==plan['teacher_execution_id'] and manifest['epoch']==plan['expected_best_epoch_zero_based']
 assert manifest['best_epoch']==plan['expected_best_epoch_zero_based'] and manifest['world_size']==1
 assert plan['source_commit'].startswith(manifest['git_commit'])
 assert abs(manifest['miou_tea']*100-plan['expected_teacher_miou_percent'])<1e-10
 c=ckpt['cfg'];assert c['dataset']['n_sup']==2646 and c['dataset']['train']['batch_size']==16 and c['dataset']['train']['crop']['size']==[321,321]
 assert c['trainer']['epochs']==80 and c['trainer']['unsupervised']['threshold']==.95 and c['trainer']['unsupervised']['loss_weight']==1
 assert not c.get('saliency_cutmix',{}).get('enabled',False)
 assert ckpt['args']['seed']==2
 recipe=json.loads((C/'recipe.json').read_text());diff=[]
 def flat(d,p=''):
  if isinstance(d,dict):return {kk:vv for k,x in d.items() for kk,vv in flat(x,p+'.'+k if p else k).items()}
  return {p:d}
 a=flat({k:c[k] for k in recipe});b=flat(recipe)
 for k in sorted(set(a)|set(b)):
  if a.get(k)!=b.get(k):diff.append({'field':k,'checkpoint':a.get(k),'resolved_log':b.get(k)})
 allowed={'net.encoder.kwargs.sync_bn':True,'net.decoder.kwargs.sync_bn':True,'net.decoder.kwargs.num_classes':21,'net.decoder.kwargs.in_planes':2048}
 assert all(d['field'] in allowed and d['checkpoint']==allowed[d['field']] and d['resolved_log'] is None for d in diff),('unexplained recipe difference',diff)
 from augseg.models.model_helper import ModelBuilder
 net=json.loads((C/'recipe.json').read_text())['net'];net['encoder']['pretrain']=None;net['encoder']['kwargs']['pretrained']=False
 model=ModelBuilder(net).float().eval();model.load_state_dict(ckpt['teacher_state'],strict=True,assign=True)
 assert state_sha(model.state_dict().items())==state_sha(ckpt['teacher_state'].items())
 assert not torch.cuda.is_initialized()
 del model
 result={'PASS':True,'checkpoint_path':str(cp),'checkpoint_bytes':cp.stat().st_size,'checkpoint_sha256':sha,'run_id':ckpt['run_id'],'saved_next_epoch':ckpt['epoch'],'embedded_manifest_epoch':manifest['epoch'],'teacher_best_epoch':manifest['best_epoch'],'teacher_miou_percent':manifest['miou_tea']*100,'teacher_state_sha256':state_sha(ckpt['teacher_state'].items()),'student_state_sha256_for_distinction':state_sha(ckpt['model_state'].items()),'teacher_tensor_keys':len(ckpt['teacher_state']),'source_commit':plan['source_commit'],'source_config_constructor_defaults':diff,'strict_CPU_model_state_load_passed':True,'state_restore_assign':True,'preserves_checkpoint_buffer_dtype':'115 EMA num_batches_tracked buffers are float32 in checkpoint; default copy would cast to int64; assign preserves exact bytes','teacher_loaded_for_inference':False,'validation_status':'NOT_RUN','optimizer_object_created':False,'optimizer_state_loaded_into_optimizer':False}
 assert result['teacher_state_sha256']!=result['student_state_sha256_for_distinction']
 pin={'checkpoint_sha256':sha,'teacher_state_sha256':result['teacher_state_sha256'],'run_id':result['run_id'],'saved_next_epoch':result['saved_next_epoch'],'source_commit':plan['source_commit']}
 if (C/'checkpoint_pin.json').exists():assert json.loads((C/'checkpoint_pin.json').read_text())==pin,'locked checkpoint pin mismatch'
 else:
  assert create_pin,'CPU checkpoint pin must be prepared before run'
  with (C/'checkpoint_pin.json').open('x') as stream:json.dump(pin,stream,indent=2)
 put('checkpoint_binding.json',result);del ckpt;gc.collect();return result
if __name__=='__main__':
 from portable import require_target,verify_package,require_prepared
 require_target();verify_package();require_prepared(C,P)
 r=verify(create_pin=False);print(json.dumps({k:r[k] for k in ['PASS','run_id','saved_next_epoch','embedded_manifest_epoch','teacher_miou_percent','checkpoint_sha256','teacher_tensor_keys','validation_status']},indent=2))

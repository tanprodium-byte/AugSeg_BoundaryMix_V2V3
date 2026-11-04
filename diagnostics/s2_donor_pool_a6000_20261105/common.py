"""Isolated replay utilities: no training entrypoint, optimizer, EMA or remote calls."""
import contextlib,csv,hashlib,json,os,random,shutil,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
C=Path(__file__).resolve().parent
P=Path(os.environ.get('DIAGNOSTIC_OUTPUT',json.loads((C/'target.json').read_text())['output'])).resolve()
sys.path.insert(0,str(C/'source'))
from util.boundary_mix import cut_mix_label_adaptive_with_mask, _rand_bbox
from util.saliency_cutmix import get_saliency_component_guided_masks
NAMES=['background','aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']
def sha_file(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def put(name,data):
 path=P/name;tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False,default=str)+'\n');os.replace(tmp,path)
def append_csv(name,rows):
 if not rows:return
 path=P/name;exists=path.exists()
 with path.open('a',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));
  if not exists:w.writeheader()
  w.writerows(rows);f.flush()
def seed_cpu(seed):random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
@contextlib.contextmanager
def preserve_cpu_rng(seed):
 py=random.getstate();npstate=np.random.get_state();ts=torch.get_rng_state();cs=torch.cuda.get_rng_state(0) if torch.cuda.is_initialized() else None;seed_cpu(seed)
 try:yield
 finally:
  random.setstate(py);np.random.set_state(npstate);torch.set_rng_state(ts)
  if cs is not None:torch.cuda.set_rng_state(cs,0)
def tensor_sha(t):
 t=t.detach().cpu().contiguous();h=hashlib.sha256();h.update(str(t.dtype).encode());h.update(str(tuple(t.shape)).encode());h.update(t.numpy().tobytes());return h.hexdigest()
def state_sha(items):
 h=hashlib.sha256()
 for k,t in sorted(items):h.update(k.encode());h.update(tensor_sha(t).encode())
 return h.hexdigest()
def model_hashes(model):return {'parameters':state_sha(model.named_parameters()),'buffers':state_sha(model.named_buffers())}
def resource_check():
 plan=json.loads((C/'plan.json').read_text());pol=plan['resource_policy']
 q=subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=10).strip().split(', ')
 procs=subprocess.check_output(['nvidia-smi','--id=0','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True,timeout=10).strip().splitlines()
 disk=shutil.disk_usage(P).free
 r={'checked_unix':time.time(),'gpu_index':int(q[0]),'uuid':q[1],'gpu_name':q[2],'gpu_total_mib':int(q[3]),'gpu_used_mib':int(q[4]),'gpu_free_mib':int(q[5]),'gpu_utilization':int(q[6]),'compute_processes_informational':procs,'disk_free_bytes':disk,'disk_free_GiB':disk/1024**3,'policy':pol,'gpu_pass':int(q[5])>=pol['min_gpu_free_mib'],'disk_pass':disk>=pol['min_disk_free_bytes']};r['ready']=r['gpu_pass'] and r['disk_pass'];return r

def check_pins(check_inputs=True):
 from portable import verify_package, check_inputs as verify_inputs
 verify_package(C)
 if check_inputs:verify_inputs(C)
 return {'source_pins':True,'runner_pins':True,'plan_inputs':check_inputs,'payload_bytes_rechecked':check_inputs}

def shadow_reference(ref):
 b,h,w=ref.shape;ix=ref.clone().long();ix[ix==255]=21
 one=torch.nn.functional.one_hot(ix,num_classes=22).permute(0,3,1,2).float()
 identity=(torch.arange(b,device=ref.device).float()+1)[:,None,None,None].expand(b,1,h,w)
 return torch.cat((one,identity),dim=1)

def mix(recipe,rgb,pseudo,q,labeled_rgb,labeled_gt,gate,ref,*,masks=None,batch_index=0):
 # Optional probability transport is used ONLY as an observational side channel.
 # Gate/pseudo/selector never read held-out GT. Production source remains byte-identical.
 args=dict(unlabeled_probs=shadow_reference(ref))
 if recipe=='S2':
  assert masks is not None and masks.shape==pseudo.shape
  args.update(labeled_masks=masks,direct_labeled_mix=True,direct_paste_policy='component_mask_random_valid_destination',destination_context={'base_seed':2,'rank':0,'epoch':0,'iteration':batch_index},destination_diagnostics={})
 elif recipe!='baseline':raise ValueError(recipe)
 capture={};translations=[];draws=[];previous=sys.getprofile();old_random=np.random.random
 def observer(frame,event,arg):
  if event!='return':return
  if frame.f_code is cut_mix_label_adaptive_with_mask.__code__:
   v=frame.f_locals;capture['permutation']=v['u_rand_index'].cpu().tolist()
   if recipe=='baseline':
    capture['lboxes']=list(zip(*[v[k].tolist() for k in ['l_bbx1','l_bby1','l_bbx2','l_bby2']]))
    capture['uboxes']=list(zip(*[v[k].tolist() for k in ['u_bbx1','u_bby1','u_bbx2','u_bby2']]))
  elif frame.f_code.co_name=='_translate_mask_to_random_valid_destination' and isinstance(arg,dict):
   parent=frame.f_back.f_locals
   translations.append({'recipient_index':int(parent['i']),'donor_index':int(parent['src']),'source_extent':list(arg['source_extent']),'destination_extent':list(arg['destination_extent']),'delta_r':arg['delta_r'],'delta_c':arg['delta_c'],'valid_translation_count':arg['num_valid_translations']})
 def logged_random(*a,**kw):
  value=old_random(*a,**kw)
  if np.ndim(value)==0:draws.append(float(value))
  return value
 sys.setprofile(observer);np.random.random=logged_random
 try:out=cut_mix_label_adaptive_with_mask(rgb.clone(),pseudo.clone(),q.clone(),labeled_rgb,labeled_gt,gate,**args)
 finally:sys.setprofile(previous);np.random.random=old_random
 assert len(draws)==(len(rgb) if recipe=='baseline' else 0),'Unexpected sample-gate RNG routing'
 trace=[]
 for i,donor in enumerate(capture['permutation']):
  rec={'recipient_index':i,'labeled_donor_index':donor,'mixing_stages':2 if recipe=='baseline' else 1,'sample_score':gate[i] if recipe=='baseline' else 'NA','gate_draw':draws[i] if recipe=='baseline' else 'NA','gate_pass':bool(draws[i]>gate[i]) if recipe=='baseline' else 'NA','stage2_donor_intermediate_index':donor if recipe=='baseline' else 'NA','source_box':str(capture['lboxes'][i]) if recipe=='baseline' else 'NA','stage2_box':str(capture['uboxes'][i]) if recipe=='baseline' else 'NA','destination_box':'NA','delta_r':'NA','delta_c':'NA','valid_translation_count':'NA'}
  if recipe=='S2':
   t=next((z for z in translations if z['recipient_index']==i),None)
   if t:rec.update(source_box=str(t['source_extent']),destination_box=str(t['destination_extent']),delta_r=t['delta_r'],delta_c=t['delta_c'],valid_translation_count=t['valid_translation_count'])
  trace.append(rec)
 rgb2,target,conf,origin,shadow=out
 is_gt=origin>.5;ref2=shadow[:,:22].argmax(1).long();ref2[ref2==21]=255;ref2=torch.where(is_gt,target.long(),ref2)
 assert torch.allclose(shadow[:,:22].sum(1)[~is_gt],torch.ones_like(conf[~is_gt]))
 assert bool(torch.all(conf[is_gt]==1));assert bool(torch.all(target[is_gt]==ref2[is_gt]))
 content=shadow[:,22].round().long()-1
 return {'rgb':rgb2,'target':target,'q':conf,'origin':is_gt,'reference':ref2,'receiver_content_index':content,'relocation':args.get('destination_diagnostics',{}),'trace':trace}

def select_masks(teacher,images,gt):
 # Profile the historical helper's return locals; no code patch or random draw.
 captured={};probabilities={};code=get_saliency_component_guided_masks.__code__;previous=sys.getprofile()
 def observer(frame,event,arg):
  if event=='return' and frame.f_code.co_name=='sample_box_by_softmax' and frame.f_back.f_code is code and isinstance(arg,tuple):
   probabilities[int(frame.f_back.f_locals['b'])]=arg[1].detach().cpu()[0].tolist()
  if event=='return' and frame.f_code is code:
   captured['component_records']=frame.f_locals['component_records'];captured['selected_records']=frame.f_locals['selected_records']
 sys.setprofile(observer)
 try:
  masks,stats=get_saliency_component_guided_masks(teacher,images,gt,_rand_bbox,temperature=.2,ignore_index=255,eps=1e-6,lam_sampler=lambda:np.random.beta(8,2),connectivity=8,foreground_only=True,min_component_area=64,max_component_area=20000)
 finally:sys.setprofile(previous)
 cand=[];selected=[]
 for i,(records,pick) in enumerate(zip(captured['component_records'],captured['selected_records'])):
  selected_index=next((j for j,r in enumerate(records) if pick is not None and r['mask'] is pick['mask']),None)
  for j,r in enumerate(records):cand.append({'donor_index':i,'candidate_index':j,'class_id':int(r['class_id']),'area':int(r['area']),'saliency_mean':r['saliency_score'],'area_fraction':r['area_score'],'score':r['score'],'selected':j==selected_index,'sampling_probability':probabilities[i][j]})
  selected.append({'donor_index':i,'candidate_count':len(records),'no_component_fallback':pick is None,'class_id':int(pick['class_id']) if pick else 'NA','component_area':int(pick['area']) if pick else 'NA','score':pick['score'] if pick else 'NA','prob_selected':pick['prob_selected'] if pick else 'NA','prob_max':pick['prob_max'] if pick else 'NA','entropy_normalized':pick['entropy'] if pick else 'NA','selector_exception':False,'actual_selected_mask_area':int(masks[i].sum().item()),'source_mask_valid_GT':int((masks[i]&gt[i].ne(255)).sum().item()),'source_mask_ignore_GT':int((masks[i]&gt[i].eq(255)).sum().item())})
 assert len(selected)==len(images)
 return masks,stats,cand,selected

def ratio(a,b):return a/b if b else 'NA'
def counts(mixed,original_pseudo,original_q,original_ref):
 y=mixed['target'].long();ref=mixed['reference'].long();g=mixed['origin'];q=mixed['q'];valid=ref!=255;active=(q>=.95)&(y!=255);p=~g
 gtvalid=g&valid;gtignore=g&~valid;correct=p&valid&active&(y==ref);wrong=p&valid&active&(y!=ref);filtered=valid&~active;unknownactive=p&~valid&active;unknownfiltered=p&~valid&~active
 n=y.numel();masks={'reference_valid':valid,'gt_valid':gtvalid,'correct_pseudo_active':correct,'wrong_pseudo_active':wrong,'valid_filtered':filtered}
 glob={name:int(v.sum()) for name,v in masks.items()};glob.update(grid_pixels=n,gt_ignore=int(gtignore.sum()),reference_ignore=int((~valid).sum()),pseudo_reference_ignore_active=int(unknownactive.sum()),pseudo_reference_ignore_filtered=int(unknownfiltered.sum()),active_loss=int(active.sum()))
 assert glob['gt_valid']+glob['correct_pseudo_active']+glob['wrong_pseudo_active']+glob['valid_filtered']==glob['reference_valid']
 assert glob['gt_ignore']+glob['pseudo_reference_ignore_active']+glob['pseudo_reference_ignore_filtered']==glob['reference_ignore']
 assert glob['reference_valid']+glob['reference_ignore']==n
 assert glob['active_loss']==glob['gt_valid']+glob['correct_pseudo_active']+glob['wrong_pseudo_active']+glob['pseudo_reference_ignore_active']
 assert not bool((g&valid&~active).any())
 def hist(mask,labels):return torch.bincount(labels[mask],minlength=21).cpu().tolist()[:21]
 hs={name:hist(mask,ref) for name,mask in masks.items()};target_hist=hist(active,y);unknown_target=hist(unknownactive,y)
 # Removed ORIGINAL receiver supervision; not corrected labels of the same content.
 b=y.shape[0];identity=torch.arange(b,device=y.device)[:,None,None];removed=g|(mixed['receiver_content_index']!=identity);oldvalid=original_ref!=255;oldactive=(original_q>=.95)&oldvalid
 removedwrong=removed&oldactive&(original_pseudo!=original_ref);removedcorrect=removed&oldactive&(original_pseudo==original_ref)
 hw=hist(removedwrong,original_ref);hc=hist(removedcorrect,original_ref);glob['original_receiver_wrong_active_removed']=int(removedwrong.sum());glob['original_receiver_correct_active_removed']=int(removedcorrect.sum())
 classes=[]
 for c in range(21):
  z={'class_id':c,'class_name':NAMES[c],**{name:hs[name][c] for name in hs},'active_supervision_target_class':target_hist[c],'unknown_reference_active_target_class':unknown_target[c],'original_receiver_wrong_active_removed':hw[c],'original_receiver_correct_active_removed':hc[c]}
  pv=z['correct_pseudo_active']+z['wrong_pseudo_active']+z['valid_filtered'];z['pseudo_reference_valid']=pv
  for field in ['gt_valid','correct_pseudo_active','wrong_pseudo_active','valid_filtered','reference_valid','active_supervision_target_class']:z[field+'_over_grid']=ratio(z[field],n)
  z['gt_coverage_in_reference_class']=ratio(z['gt_valid'],z['reference_valid']);z['active_pseudo_coverage_in_pseudo_reference_class']=ratio(z['correct_pseudo_active']+z['wrong_pseudo_active'],pv);z['active_pseudo_error_rate']=ratio(z['wrong_pseudo_active'],z['correct_pseudo_active']+z['wrong_pseudo_active']);classes.append(z)
 confmask=p&valid&active;confusion=torch.bincount((ref[confmask]*21+y[confmask]),minlength=441).reshape(21,21).cpu().tolist()
 for field in ['gt_valid','correct_pseudo_active','wrong_pseudo_active','valid_filtered','reference_valid','active_loss','gt_ignore','reference_ignore']:glob[field+'_over_grid']=ratio(glob[field],n)
 glob['active_pseudo_error_rate']=ratio(glob['wrong_pseudo_active'],glob['correct_pseudo_active']+glob['wrong_pseudo_active'])
 return glob,classes,confusion

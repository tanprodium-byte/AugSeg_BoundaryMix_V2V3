"""Synthetic CPU-only checks of historical helpers and observational instrumentation."""
import ast,copy,json,math,random,unittest,tempfile,contextlib,io
from unittest.mock import patch
from common import *
from common import _rand_bbox
from util.saliency_cutmix import compute_labeled_teacher_saliency
from augseg.utils.loss_helper import compute_unsupervised_loss_by_threshold
from runner import resource_gate

def rng_digest():return (repr(random.getstate()),repr(np.random.get_state()),tensor_sha(torch.get_rng_state()))
def toy(b=4,h=8,w=8):
 ref=(torch.arange(b*h*w).reshape(b,h,w)%21).long();ref[:,0,0]=255
 y=(ref.clamp_max(20)+3)%21;q=torch.full((b,h,w),.97);q[:,0,1]=.5
 rgb=torch.stack((ref.float(),torch.arange(h*w).reshape(1,h,w).expand(b,h,w).float(),torch.arange(b).reshape(b,1,1).expand(b,h,w).float()+1),1)
 lg=(ref+4)%21;lg[:,2,2]=255;lx=torch.stack((lg.float(),rgb[:,1],rgb[:,2]+100),1)
 return rgb,y,q,lx,lg,[.5]*b,ref
class Probe(torch.nn.Module):
 def __init__(self):
  super().__init__();self.conv=torch.nn.Conv2d(3,21,1);self.bn=torch.nn.BatchNorm2d(21)
 def forward(self,x):return self.bn(self.conv(x)),None
class ReplayChecks(unittest.TestCase):
 def test_01_baseline_shadow_no_choice_or_rng_change(self):
  x,y,q,l,g,gate,ref=toy();seed_cpu(91);base=cut_mix_label_adaptive_with_mask(x.clone(),y.clone(),q.clone(),l,g,gate);rb=rng_digest()
  seed_cpu(91);obs=mix('baseline',x,y,q,l,g,gate,ref);ro=rng_digest()
  for a,b in zip(base,[obs['rgb'],obs['target'],obs['q'],obs['origin'].float()]):self.assertTrue(torch.equal(a,b))
  self.assertEqual(rb,ro)
 def test_02_receiver_reference_cannot_change_baseline_choices(self):
  args=toy();seed_cpu(77);a=mix('baseline',*args);seed_cpu(77);other=list(args);other[-1]=torch.zeros_like(other[-1]);b=mix('baseline',*other)
  for k in ['rgb','target','q','origin']:self.assertTrue(torch.equal(a[k],b[k]))
 def test_03_two_stages_and_permutation_alignment(self):
  x,y,q,l,g,gate,ref=toy();idx=torch.tensor([1,2,3,0]);boxes=[(1,1,5,5),(0,0,6,6)]
  def bbox(size,lam):return tuple(np.full(size[0],v,dtype=np.int64) for v in boxes.pop(0))
  with patch('util.boundary_mix._rand_bbox',side_effect=bbox),patch('torch.randperm',return_value=idx):
   o=mix('baseline',x,y,q,l,g,[0.]*4,ref)
  self.assertEqual(len(boxes),0);self.assertTrue(torch.equal(o['rgb'][:,0].long(),o['reference']))
  self.assertTrue(o['origin'][:,1:5,1:5].all());self.assertFalse(o['origin'][:,6:,:].any())
  for i in range(4):self.assertEqual(float(o['rgb'][i,2,2,2]),101+int(idx[idx[i]]))
  z,_,_=counts(o,y,q,ref);self.assertGreater(z['gt_ignore'],0)
 def test_04_s2_one_stage_exact_relocation_alignment(self):
  x,y,q,l,g,gate,ref=toy();m=torch.zeros_like(g,dtype=torch.bool);m[:,1:5,1:5]=True;seed_cpu(54)
  o=mix('S2',x,y,q,l,g,gate,ref,masks=m,batch_index=3)
  self.assertEqual(int(o['origin'].sum()),4*16);self.assertTrue(torch.equal(o['rgb'][:,0].long(),o['reference']))
  self.assertTrue(torch.equal(o['target'][~o['origin']],y[~o['origin']]))
  identity=torch.arange(4)[:,None,None].expand_as(y);self.assertTrue(torch.equal(o['receiver_content_index'][~o['origin']],identity[~o['origin']]))
  self.assertTrue(all(z['mixing_stages']==1 for z in o['trace']))
 def test_05_s2_shadow_rng_equivalence_and_reference_independence(self):
  x,y,q,l,g,gate,ref=toy();m=torch.zeros_like(g,dtype=torch.bool);m[:,1:5,1:5]=True
  args=dict(labeled_masks=m,direct_labeled_mix=True,direct_paste_policy='component_mask_random_valid_destination',destination_context={'base_seed':2,'rank':0,'epoch':0,'iteration':0})
  seed_cpu(42);a=cut_mix_label_adaptive_with_mask(x.clone(),y.clone(),q.clone(),l,g,gate,**args);r1=rng_digest()
  seed_cpu(42);b=mix('S2',x,y,q,l,g,gate,ref,masks=m);r2=rng_digest()
  for a1,b1 in zip(a,[b['rgb'],b['target'],b['q'],b['origin'].float()]):self.assertTrue(torch.equal(a1,b1))
  self.assertEqual(r1,r2)
  seed_cpu(42);c=mix('S2',x,y,q,l,g,gate,torch.zeros_like(ref),masks=m)
  for k in ['rgb','target','q','origin']:self.assertTrue(torch.equal(b[k],c[k]))
 def test_06_ignore_threshold_BHW_and_all_count_categories(self):
  # Seven mutually exclusive groups, including unknown-reference active pseudo.
  y=torch.tensor([[[1,255,2,3,4,5,6,7]]]);ref=torch.tensor([[[1,255,2,2,4,255,255,7]]]);q=torch.tensor([[[1.,1.,.95,.99,.4,.96,.4,.95]]]);gt=torch.tensor([[[1,1,0,0,0,0,0,0]]],dtype=torch.bool)
  mixd={'target':y,'reference':ref,'q':q,'origin':gt,'receiver_content_index':torch.zeros_like(y)}
  total,cl,_=counts(mixd,y,q,ref);self.assertEqual(total['grid_pixels'],8);self.assertEqual(total['gt_ignore'],1);self.assertEqual(total['pseudo_reference_ignore_active'],1);self.assertEqual(total['pseudo_reference_ignore_filtered'],1);self.assertEqual(total['wrong_pseudo_active'],1);self.assertEqual(total['valid_filtered'],1)
  logits=torch.zeros(1,21,1,8);loss,active=compute_unsupervised_loss_by_threshold(logits,y.clone(),q,thresh=.95)
  self.assertAlmostEqual(float(active),total['active_loss']/8);self.assertAlmostEqual(float(loss),math.log(21)*total['active_loss']/8,places=6)
  self.assertEqual(cl[20]['active_pseudo_error_rate'],'NA')
 def test_07_teacher_probe_state_and_parameters_preserved(self):
  model=Probe().eval()
  for p in model.parameters():p.requires_grad_(False)
  x=torch.rand(2,3,12,12);gt=torch.ones(2,12,12,dtype=torch.long);gt[:,0,0]=255;h=model_hashes(model)
  s=compute_labeled_teacher_saliency(model,x,gt);self.assertEqual(s.shape,(2,12,12));self.assertFalse(s.requires_grad);self.assertEqual(model_hashes(model),h);self.assertFalse(model.training);self.assertTrue(all(not p.requires_grad and p.grad is None for p in model.parameters()))
 def test_08_component_observer_has_no_rng_or_selection_effect(self):
  model=Probe().eval()
  for p in model.parameters():p.requires_grad_(False)
  x=torch.rand(2,3,12,12);gt=torch.ones(2,12,12,dtype=torch.long);gt[1]=0;gt[:,0,0]=255
  seed_cpu(33);a,stats=get_saliency_component_guided_masks(model,x,gt,_rand_bbox,lam_sampler=lambda:np.random.beta(8,2));r1=rng_digest()
  seed_cpu(33);b,stats2,candidates,selected=select_masks(model,x,gt);r2=rng_digest()
  self.assertTrue(torch.equal(a,b));self.assertEqual(stats,stats2);self.assertEqual(r1,r2);self.assertEqual(selected[0]['candidate_count'],1);self.assertTrue(selected[1]['no_component_fallback']);self.assertEqual(len(candidates),1)
 def test_09_resource_failure_cannot_pass(self):
  with self.assertRaisesRegex(RuntimeError,'BLOCKED_RESOURCE'):resource_gate({'ready':False})
 def test_10_locked_plan_and_disjoint_nested_lists(self):
  p=json.loads((C/'plan.json').read_text());r=list(csv.DictReader(open(C/'receiver_plan.csv')));d=list(csv.DictReader(open(C/'donor_plan.csv')))
  self.assertEqual(len(r),1024);self.assertEqual(len({z['id'] for z in r}),1024);self.assertEqual(len(d),3072)
  self.assertEqual({z['split'] for z in d},{'662','1323','2646'}) # full nested lists checked separately against external checksummed splits
  self.assertFalse(set(z['id'] for z in r)&set(z['id'] for z in d));self.assertEqual(p['batch'],16)
 def test_11_no_optimizer_or_training_entrypoint_calls(self):
  code=(C/'runner.py').read_text();tree=ast.parse(code);calls=[ast.unparse(x.func) for x in ast.walk(tree) if isinstance(x,ast.Call)]
  self.assertFalse(any(x.startswith('torch.optim.') or x.endswith('optimizer.step') for x in calls));self.assertNotIn('import train_semi',code);self.assertNotIn('wandb.init',code)
 def test_12a_assign_restores_float_EMA_BN_counter_exactly(self):
  m=Probe().eval();state=m.state_dict();state['bn.num_batches_tracked']=torch.tensor(123.42578125,dtype=torch.float32);m.load_state_dict(state,strict=True,assign=True)
  self.assertEqual(state_sha(m.state_dict().items()),state_sha(state.items()));self.assertEqual(m.bn.num_batches_tracked.dtype,torch.float32);h=model_hashes(m)
  for p in m.parameters():p.requires_grad_(False)
  compute_labeled_teacher_saliency(m,torch.rand(2,3,12,12),torch.ones(2,12,12,dtype=torch.long));self.assertEqual(model_hashes(m),h)
 def test_12b_receiver_GT_cannot_change_augmentation_RGB(self):
  from runner import augment_batch
  from PIL import Image
  rgb=Image.fromarray(np.arange(48*64*3,dtype=np.uint8).reshape(48,64,3));a=Image.fromarray(np.zeros((48,64),dtype=np.uint8));b=Image.fromarray(np.full((48,64),255,dtype=np.uint8));rows=[{'id':'synthetic','augmentation_seed':6100000}];cfg=json.loads((C/'recipe.json').read_text())
  with patch('runner.load_image',return_value=(rgb,a)):x,sa,_=augment_batch(rows,cfg,P,True)
  with patch('runner.load_image',return_value=(rgb,b)):y,sb,_=augment_batch(rows,cfg,P,True)
  self.assertTrue(torch.equal(x,y));self.assertTrue(torch.equal(sa,sb))
 def test_12c_analysis_cannot_restart_model_or_training(self):
  text=(C/'analyze_replay.py').read_text();tree=ast.parse(text);names=[ast.unparse(x.func) for x in ast.walk(tree) if isinstance(x,ast.Call)];self.assertFalse(any(x in ('run','runner.run') for x in names));self.assertNotIn('import runner',text);self.assertNotIn('import torch',text)
 def test_12_pins_intact_and_no_CUDA_context(self):
  self.assertTrue(check_pins(check_inputs=False)['source_pins']);self.assertFalse(torch.cuda.is_initialized())
 def test_13_validation_mismatch_must_fail(self):
  from runner import validate_teacher
  from PIL import Image
  class Wrong(torch.nn.Module):
   def forward(self,x):return torch.zeros(x.shape[0],21,*x.shape[-2:]),None
  with tempfile.TemporaryDirectory(dir=P) as td:
   fp=Path(td)/'val.txt';fp.write_text('synthetic\n'*1449);plan=json.loads((C/'plan.json').read_text());plan['validation_list']=str(fp);plan['expected_teacher_miou_percent']=1.;cfg=json.loads((C/'recipe.json').read_text());rgb=Image.fromarray(np.zeros((4,5,3),dtype=np.uint8));gt=Image.fromarray(np.ones((4,5),dtype=np.uint8));captured={}
   with patch('runner.load_image',return_value=(rgb,gt)),patch('runner.put',side_effect=lambda name,data:captured.update({name:data})),contextlib.redirect_stdout(io.StringIO()):
    with self.assertRaisesRegex(RuntimeError,'VALIDATION_MISMATCH'):validate_teacher(Wrong().eval(),plan,cfg,torch.device('cpu'))
   self.assertFalse(captured['validation.json']['passed'])
 def test_14_recovery_never_replays_completed_or_partial(self):
  from runner import run
  with tempfile.TemporaryDirectory(dir=P) as td:
   root=Path(td)
   for name in ['plan.json','recipe.json']:(root/name).write_bytes((C/name).read_bytes())
   (root/'status.json').write_text(json.dumps({'state':'COMPLETED'}))
   with patch('runner.P',root),patch('runner.verify_checkpoint',side_effect=AssertionError('must not reload')),contextlib.redirect_stdout(io.StringIO()):self.assertEqual(run(),0)
   (root/'status.json').write_text(json.dumps({'state':'FAILED_DIAGNOSTIC'}));(root/'batch_counts.csv').write_text('partial')
   with patch('runner.P',root),patch('runner.verify_checkpoint',side_effect=AssertionError('must not reload')):
    with self.assertRaisesRegex(RuntimeError,'EXISTING_DIAGNOSTIC_OBSERVATIONS'):run()
if __name__=='__main__':
 suite=unittest.defaultTestLoader.loadTestsFromTestCase(ReplayChecks);result=unittest.TextTestRunner(verbosity=2).run(suite);put('cpu_tests.json',{'passed':result.wasSuccessful(),'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'CUDA_context_initialized':torch.cuda.is_initialized(),'scope':'synthetic CPU/mocks only; no real teacher forward/validation/replay'});raise SystemExit(0 if result.wasSuccessful() else 1)

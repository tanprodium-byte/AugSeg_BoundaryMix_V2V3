"""Historical ScientificContract tests; portable harness, CPU only."""
import ast,copy,csv,importlib.util,io,json,sys,tempfile,unittest
from pathlib import Path
from types import ModuleType,SimpleNamespace
from unittest import mock
import numpy as np
import torch
import torch.nn.functional as F
import yaml
PKG=Path(__file__).resolve().parents[1]
ROOT=PKG/'sources'/'s2'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(PKG))
from util import saliency_cutmix as sc
from util import boundary_mix as bm
from augseg.dataset.augs_ALIA import cut_mix_label_adaptive,rand_bbox
from augseg.utils.loss_helper import compute_unsupervised_loss_by_threshold
import controller
origins=json.loads((PKG/'origins.json').read_text())
def git(*args):
    assert args[0]=='show'
    rev,rel=args[1].split(':',1)
    matches=[p for p in (PKG/'historical/git_exports').iterdir() if p.name.startswith(rev)]
    assert len(matches)==1,(rev,rel)
    return (matches[0]/rel).read_bytes()
ops=SimpleNamespace(BASE=origins['sources']['s2']['parent_commit'],git=git,profiles=lambda:[r for r in controller.profiles() if r['source']=='s2'],expected_config=controller.expected_config)
torch.set_num_threads(1)

class ScientificContract(unittest.TestCase):
    def fixtures(self):
        x=torch.zeros(2,3,32,32);y=torch.zeros(2,32,32,dtype=torch.long)
        y[:,1:9,1:9]=1;y[:,16:24,1:17]=2
        maps=torch.zeros(2,32,32);maps[:,1:9,1:9]=1;maps[:,16:24,1:17]=.1
        return x,y,maps

    def historical(self):
        module=ModuleType('historical_s2')
        exec(ops.git('show',ops.BASE+':util/saliency_cutmix.py'),module.__dict__)
        return module

    def test_teacher_input_gradient_no_parameter_bn_updates(self):
        teacher=torch.nn.Sequential(torch.nn.Conv2d(3,21,1),torch.nn.BatchNorm2d(21));teacher.train()
        before={k:v.clone() for k,v in teacher.state_dict().items()}
        x=torch.randn(2,3,4,4);y=torch.ones(2,4,4,dtype=torch.long);y[:,0]=255
        sal=sc.compute_labeled_teacher_saliency(teacher,x,y)
        self.assertTrue(teacher.training);self.assertFalse(sal.requires_grad);self.assertIsNone(x.grad)
        self.assertTrue(torch.isfinite(sal).all());self.assertTrue(((sal>=0)&(sal<=1)).all())
        for k,v in teacher.state_dict().items():self.assertTrue(torch.equal(v,before[k]))
        for p in teacher.parameters():self.assertIsNone(p.grad);self.assertTrue(p.requires_grad)

    def test_unchanged_eligibility_saliency_sampling_and_relocation_source(self):
        current=ast.parse((ROOT/'util/saliency_cutmix.py').read_text());old=ast.parse(ops.git('show',ops.BASE+':util/saliency_cutmix.py').decode())
        for name in ('normalize_saliency_per_image','compute_labeled_teacher_saliency','connected_components_from_gt','sample_box_by_softmax','get_saliency_component_guided_boxes','boxes_to_masks'):
            get=lambda t:next(n for n in t.body if isinstance(n,ast.FunctionDef)and n.name==name)
            self.assertEqual(ast.dump(get(current)),ast.dump(get(old)))
        self.assertEqual((ROOT/'util/boundary_mix.py').read_bytes(),ops.git('show',ops.BASE+':util/boundary_mix.py'))
        self.assertEqual((ROOT/'augseg/utils/loss_helper.py').read_bytes(),ops.git('show',ops.BASE+':augseg/utils/loss_helper.py'))

    def test_default_policy_matches_control_outputs_stats_and_rng(self):
        x,y,maps=self.fixtures();values=[]
        for module in (self.historical(),sc):
            np.random.seed(11);torch.manual_seed(13)
            with mock.patch.object(module,'compute_labeled_teacher_saliency',return_value=maps):
                result=module.get_saliency_component_guided_masks(None,x,y,bm._rand_bbox,lam_sampler=lambda:np.random.beta(8,2))
            values.append((result,torch.get_rng_state(),np.random.get_state()))
        self.assertTrue(torch.equal(values[0][0][0],values[1][0][0]));self.assertEqual(values[0][0][1],values[1][0][1])
        self.assertTrue(torch.equal(values[0][1],values[1][1]));self.assertTrue(np.array_equal(values[0][2][1],values[1][2][1]));self.assertEqual(values[0][2][2:],values[1][2][2:])

    def test_area_only_scores_softmax_sampling_and_identical_rng_call_counts(self):
        x,y,maps=self.fixtures();scores=[];states=[];probs=[]
        original=sc.sample_box_by_softmax;draw=torch.multinomial
        for mode in ('mean_saliency_area','area_only'):
            def capture(v,**kw):
                scores.append(v.clone());return original(v,**kw)
            np.random.seed(17);torch.manual_seed(19)
            with mock.patch.object(sc,'compute_labeled_teacher_saliency',return_value=maps) as sal,mock.patch.object(sc,'sample_box_by_softmax',side_effect=capture),mock.patch.object(sc.torch,'multinomial',wraps=draw) as multi,mock.patch.object(bm,'_rand_bbox',wraps=bm._rand_bbox) as box,mock.patch.object(np.random,'beta',wraps=np.random.beta) as beta:
                sc.get_saliency_component_guided_masks(None,x,y,box,lam_sampler=lambda:np.random.beta(8,2),component_score=mode)
                self.assertEqual(sal.call_count,1);self.assertEqual(multi.call_count,2);self.assertEqual(box.call_count,1);beta.assert_called_once_with(8,2)
                probs.append(multi.call_args_list[0].args[0].clone())
            states.append((torch.get_rng_state(),np.random.get_state()))
        torch.testing.assert_close(scores[0],torch.tensor([[64/1024,.1*128/1024]]))
        torch.testing.assert_close(scores[2],torch.tensor([[64/1024,128/1024]]))
        torch.testing.assert_close(probs[1],torch.softmax(scores[2]/.2,dim=1).clamp_min(1e-6))
        self.assertTrue(torch.equal(states[0][0],states[1][0]));self.assertTrue(np.array_equal(states[0][1][1],states[1][1][1]));self.assertEqual(states[0][1][2:],states[1][1][2:])

    def test_component_eligibility_foreground_ignore_area_and_connectivity(self):
        y=torch.zeros(24,24,dtype=torch.long);y[1:9,1:9]=1;y[12:15,12:15]=2;y[19:,19:]=255
        components=sc.connected_components_from_gt(y,min_component_area=64,max_component_area=20000)
        self.assertEqual([(c['class_id'],c['area']) for c in components],[(1,64)])
        self.assertEqual(sc.connected_components_from_gt(y,min_component_area=65),[])

    def test_no_component_random_rectangle_fallback_preserved(self):
        x=torch.zeros(2,3,8,8);y=torch.full((2,8,8),255,dtype=torch.long);coords=tuple(np.array([v,v])for v in (1,2,4,6))
        results=[]
        for module,kwargs in ((self.historical(),{}),(sc,{'component_score':'area_only'})):
            with mock.patch.object(module,'compute_labeled_teacher_saliency',return_value=torch.zeros(2,8,8)),mock.patch.object(module,'sample_box_by_softmax',side_effect=AssertionError('no component must not sample')):
                mask,stats=module.get_saliency_component_guided_masks(None,x,y,lambda *a,**k:coords,lam_sampler=lambda:.8,**kwargs)
            results.append(mask);self.assertEqual(stats['saliency/fallback_ratio'],1.)
        self.assertTrue(torch.equal(*results));self.assertEqual(results[0].sum().item(),24)

    def test_one_stage_donor_alignment_transport_relocation_and_no_gate(self):
        u=torch.zeros(2,3,8,8);l=torch.arange(2*3*8*8).reshape(2,3,8,8).float()+1000
        gt=torch.stack([torch.full((8,8),3),torch.full((8,8),4)]);gt[0,1,2]=255
        y=torch.zeros_like(gt);q=torch.full((2,8,8),.6);masks=torch.zeros_like(gt,dtype=torch.bool);masks[0,1:3,2:5]=True;masks[1,4:7,1:3]=True
        observations=[];original=bm._translate_mask_to_random_valid_destination
        def translate(mask,generator):
            value=original(mask,generator);observations.append(value);return value
        with mock.patch.object(bm.torch,'randperm',return_value=torch.tensor([1,0])) as perm,mock.patch.object(bm.np.random,'random',side_effect=AssertionError('sample gate forbidden')),mock.patch.object(bm,'_rand_bbox',side_effect=AssertionError('stage2/fallback sampling forbidden')),mock.patch.object(bm,'_translate_mask_to_random_valid_destination',side_effect=translate):
            rgb,target,conf,origin=bm.cut_mix_label_adaptive_with_mask(u,y,q,l,gt,[1.,1.],labeled_masks=masks,direct_labeled_mix=True,direct_paste_policy='component_mask_random_valid_destination',destination_context={'base_seed':2,'rank':0,'epoch':0,'iteration':0},destination_diagnostics={})
        self.assertEqual(perm.call_count,1);self.assertEqual(len(observations),2)
        for i,t in enumerate(observations):
            src=1-i;sh,sw,eh,ew=t['source_extent'];dh,dw,deh,dew=t['destination_extent'];mask=t['local_mask']
            self.assertTrue(torch.equal(rgb[i,:,dh:deh,dw:dew][:,mask],l[src,:,sh:eh,sw:ew][:,mask]))
            self.assertTrue(torch.equal(target[i,dh:deh,dw:dew][mask],gt[src,sh:eh,sw:ew][mask]));self.assertTrue(conf[i,origin[i].bool()].eq(1).all())
        self.assertEqual(int(origin.sum()),int(masks.sum()));self.assertEqual(int((target==255).sum()),1)
        before=torch.get_rng_state().clone();stats=sc.actual_gt_paste_stats(origin,target,.5)
        self.assertTrue(torch.equal(before,torch.get_rng_state()));self.assertEqual(stats['s2/gt_origin_pixels'],12);self.assertEqual(stats['s2/ignore_gt_origin_pixels'],1);self.assertEqual(stats['s2/valid_gt_origin_pixels'],11)
        self.assertEqual(stats['s2/gt_origin_fraction'],12/128);self.assertEqual(stats['s2/component_fallback_sample_fraction'],.5)

    def test_production_caller_routes_area_only_and_keeps_single_stage(self):
        import builtins
        tree=ast.parse((ROOT/'train_semi.py').read_text());selector=next(n for n in ast.walk(tree)if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=='get_saliency_component_guided_masks')
        mixer=next(n for n in ast.walk(tree)if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=='cut_mix_label_adaptive_with_mask' and any(k.arg=='direct_labeled_mix'for k in n.keywords) and not any(k.arg=='return_target_metadata'for k in n.keywords))
        cfg=ops.expected_config(ops.profiles()[0]);x,y,maps=self.fixtures();names={n.id for call in (selector,mixer)for n in ast.walk(call)if isinstance(n,ast.Name)}
        scope={n:None for n in names if not hasattr(builtins,n)};scope.update(cfg=cfg,np=np,torch=torch,model_teacher=None,image_l=x+10,label_l=y,image_u_aug=x,label_u_aug=torch.zeros_like(y),logits_u_aug=torch.full(y.shape,.6),confidence=[1.,1.],saliency_cutmix_cfg=cfg['saliency_cutmix'],fixed_size_csl_destination_cfg={},destination_kwargs={'destination_context':{'base_seed':2,'rank':0,'epoch':0,'iteration':0},'destination_diagnostics':{}},get_saliency_component_guided_masks=sc.get_saliency_component_guided_masks,cut_mix_label_adaptive_with_mask=bm.cut_mix_label_adaptive_with_mask,_rand_bbox=bm._rand_bbox)
        with mock.patch.object(sc,'compute_labeled_teacher_saliency',return_value=maps),mock.patch.object(bm.np.random,'random',side_effect=AssertionError('gate forbidden')),mock.patch.object(bm,'_rand_bbox',wraps=bm._rand_bbox) as boxes:
            scope['saliency_labeled_masks'],stats=eval(compile(ast.Expression(selector),'production_selector','eval'),scope)
            rgb,target,q,origin=eval(compile(ast.Expression(mixer),'production_mixer','eval'),scope)
        self.assertEqual(boxes.call_count,0) # selector's sampler was passed before wrapping
        self.assertEqual(stats['saliency/component_score_mode_id'],1);self.assertGreater(origin.sum().item(),0);self.assertTrue(q[origin.bool()].eq(1).all())

    def test_selector_exception_aborts_before_legacy_two_stage_fallback(self):
        tree=ast.parse((ROOT/'train_semi.py').read_text());handler=next(n for n in ast.walk(tree)if isinstance(n,ast.ExceptHandler)and any(isinstance(b,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='saliency_selector_exception_batch'for t in b.targets)for b in n.body))
        statement=ast.Try(body=[ast.Raise(exc=ast.Call(func=ast.Name(id='ValueError',ctx=ast.Load()),args=[ast.Constant('synthetic')],keywords=[]),cause=None)],handlers=[handler],orelse=[],finalbody=[])
        scope={'saliency_cutmix_cfg':{'component_score':'area_only'},'saliency_labeled_masks':'unchanged'}
        with self.assertRaisesRegex(RuntimeError,'before any two-stage fallback'):exec(compile(ast.fix_missing_locations(ast.Module(body=[statement],type_ignores=[])),'handler','exec'),scope)
        self.assertEqual(scope['saliency_labeled_masks'],'unchanged')

    def test_ce_threshold_bhw_gt_ignore_weight_unchanged(self):
        z=torch.zeros(1,21,1,4);y=torch.tensor([[[2,2,3,255]]]);q=torch.tensor([[[.95,.949,1.,1.]]])
        loss,active=compute_unsupervised_loss_by_threshold(z,y.clone(),q)
        self.assertAlmostEqual(loss.item(),2*float(np.log(21))/4,places=6);self.assertEqual(active.item(),.5)
        self.assertEqual(ops.expected_config(ops.profiles()[0])['trainer']['unsupervised']['loss_weight'],1.)
if __name__=="__main__":unittest.main()

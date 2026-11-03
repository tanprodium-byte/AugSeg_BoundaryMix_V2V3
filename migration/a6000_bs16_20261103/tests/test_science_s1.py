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
ROOT=PKG/'sources'/'s1'
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
ops=SimpleNamespace(BASE=origins['sources']['s1']['parent_commit'],git=git,profiles=lambda:[r for r in controller.profiles() if r['source']=='s1'],expected_config=controller.expected_config)
torch.set_num_threads(1)

class ScientificContract(unittest.TestCase):
    def test_teacher_input_gradient_without_parameter_or_bn_updates(self):
        # Tiny synthetic CPU teacher, not the segmentation model.
        teacher = torch.nn.Sequential(torch.nn.Conv2d(3, 21, 1), torch.nn.BatchNorm2d(21))
        teacher.train()
        before = {k: v.clone() for k, v in teacher.state_dict().items()}
        x = torch.randn(2, 3, 4, 4); y = torch.ones(2, 4, 4, dtype=torch.long); y[:, 0] = 255
        sal = sc.compute_labeled_teacher_saliency(teacher, x, y)
        self.assertEqual(sal.shape, y.shape); self.assertFalse(sal.requires_grad)
        self.assertTrue(torch.isfinite(sal).all()); self.assertTrue((sal >= 0).all()); self.assertTrue((sal <= 1).all())
        self.assertTrue(teacher.training); self.assertIsNone(x.grad)
        for k, v in teacher.state_dict().items(): self.assertTrue(torch.equal(v, before[k]))
        for p in teacher.parameters(): self.assertIsNone(p.grad); self.assertTrue(p.requires_grad)

    def test_saliency_and_normalization_core_equal_historical_objects(self):
        current = ast.parse((ROOT / "util/saliency_cutmix.py").read_text())
        for commit in ("1a9387f", "c47581b", ops.BASE):
            old = ast.parse(ops.git("show", commit + ":util/saliency_cutmix.py").decode())
            for name in ("compute_labeled_teacher_saliency", "normalize_saliency_per_image", "sample_candidate_boxes_with_baseline_sampler", "box_mean_saliency", "sample_box_by_softmax"):
                get = lambda tree: next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
                self.assertEqual(ast.dump(get(current), include_attributes=False), ast.dump(get(old), include_attributes=False))

    def test_saliency_default_matches_original_selection_and_rng(self):
        old = ModuleType("historical_s1")
        exec(ops.git("show", "c47581b:util/saliency_cutmix.py"), old.__dict__)
        images = torch.zeros(2, 3, 16, 16); labels = torch.ones(2, 16, 16, dtype=torch.long)
        maps = torch.arange(512).reshape(2,16,16).float()/512
        values = []
        for module in (old, sc):
            np.random.seed(19); torch.manual_seed(23)
            with mock.patch.object(module, "compute_labeled_teacher_saliency", return_value=maps):
                result = module.get_saliency_guided_boxes(None, images, labels, bm._rand_bbox, lam_sampler=lambda: np.random.beta(8, 2))
            values.append((result[0], torch.get_rng_state(), np.random.get_state()))
        self.assertTrue(torch.equal(values[0][0], values[1][0])); self.assertTrue(torch.equal(values[0][1], values[1][1]))
        self.assertTrue(np.array_equal(values[0][2][1], values[1][2][1])); self.assertEqual(values[0][2][2:], values[1][2][2:])

    def test_uniform_changes_only_sampling_weights_same_pool_and_draw(self):
        images = torch.zeros(2,3,16,16); labels = torch.ones(2,16,16,dtype=torch.long)
        maps = torch.zeros(2,16,16); maps[:, :8] = 1
        original_pool = sc.sample_candidate_boxes_with_baseline_sampler
        original_draw = torch.multinomial
        pools, distributions, states = [], [], []
        for policy in ("saliency_softmax", "uniform_multinomial"):
            def pool(*a, **kw):
                value = original_pool(*a, **kw); pools.append(value.clone()); return value
            np.random.seed(29); torch.manual_seed(31)
            with mock.patch.object(sc, "compute_labeled_teacher_saliency", return_value=maps) as sal, mock.patch.object(sc, "sample_candidate_boxes_with_baseline_sampler", side_effect=pool), mock.patch.object(sc.torch, "multinomial", wraps=original_draw) as draw, mock.patch.object(bm, "_rand_bbox", wraps=bm._rand_bbox) as boxes:
                result = sc.get_saliency_guided_boxes(None, images, labels, boxes, lam_sampler=lambda: np.random.beta(8,2), candidate_selection=policy)
                self.assertEqual(sal.call_count,1); self.assertEqual(boxes.call_count,8); self.assertEqual(draw.call_count,1)
                distributions.append(draw.call_args.args[0].clone())
                self.assertEqual(result[1]["saliency/candidate_policy_id"], float(policy == "uniform_multinomial"))
            states.append(torch.get_rng_state())
        self.assertTrue(torch.equal(pools[0],pools[1])); self.assertTrue(torch.equal(states[0],states[1]))
        torch.testing.assert_close(distributions[1],torch.full((2,8),1/8))
        self.assertFalse(torch.equal(distributions[0],distributions[1]))

    def test_box_score_is_mean_without_area_or_confidence(self):
        sal = torch.tensor([[[1.,1.],[0.,0.]]])
        boxes = torch.tensor([[[0,0,1,2],[0,0,2,2],[1,0,1,1]]])
        torch.testing.assert_close(sc.box_mean_saliency(sal,boxes),torch.tensor([[1.,.5,0.]]))

    def test_two_stage_donor_permutation_rgb_target_confidence_ignore(self):
        u=torch.stack([torch.ones(3,4,4),torch.full((3,4,4),2.)]); l=torch.stack([torch.full((3,4,4),10.),torch.full((3,4,4),20.)])
        y=torch.zeros(2,4,4,dtype=torch.long);gt=torch.stack([torch.full((4,4),3),torch.full((4,4),4)]);gt[0,0,0]=255
        q=torch.full((2,4,4),.7);selected=torch.tensor([[0,0,2,2],[2,2,4,4]])
        # Stage2 boxes take the GT region from the opposite intermediate.
        boxes=tuple(np.array([a,b]) for a,b in ((0,2),(0,2),(2,4),(2,4)))
        with mock.patch.object(bm.torch,"randperm",return_value=torch.tensor([1,0])) as perm, mock.patch.object(bm,"_rand_bbox",return_value=boxes) as box, mock.patch.object(bm.np.random,"beta",return_value=.5) as beta, mock.patch.object(bm.np.random,"random",side_effect=[.5,.5]) as gate:
            mixed,target,confidence,origin=bm.cut_mix_label_adaptive_with_mask(u.clone(),y.clone(),q.clone(),l,gt,[0.,0.],labeled_boxes=selected)
        self.assertEqual(perm.call_count,1);self.assertEqual(box.call_count,1);beta.assert_called_once_with(4,4);self.assertEqual(gate.call_count,2)
        self.assertTrue(mixed[0,:,:2,:2].eq(10).all());self.assertTrue(mixed[1,:,2:,2:].eq(20).all())
        self.assertEqual(target[0,0,0].item(),255);self.assertTrue(confidence[0,:2,:2].eq(1).all());self.assertTrue(origin[0,:2,:2].eq(1).all())
        self.assertTrue(mixed[0,:,2:,2:].eq(1).all());self.assertTrue(confidence[0,2:,2:].eq(.7).all())

    def test_gate_direction_and_unconditional_stage2(self):
        u=torch.zeros(2,3,4,4);l=torch.ones_like(u);y=torch.zeros(2,4,4,dtype=torch.long);q=torch.full_like(y,.5,dtype=torch.float)
        full=torch.tensor([[0,0,4,4],[0,0,4,4]]);coords=tuple(np.array([v,v]) for v in (0,0,4,4))
        with mock.patch.object(bm.torch,"randperm",return_value=torch.tensor([1,0])),mock.patch.object(bm,"_rand_bbox",return_value=coords),mock.patch.object(bm.np.random,"beta",return_value=.5),mock.patch.object(bm.np.random,"random",side_effect=[.2,.8]) as gate:
            mixed,*_=bm.cut_mix_label_adaptive_with_mask(u,y,q,l,y,[1.,0.],labeled_boxes=full)
        self.assertEqual(gate.call_count,2);self.assertTrue(mixed[0].eq(1).all());self.assertTrue(mixed[1].eq(0).all())

    def test_both_profiles_execute_production_caller_and_two_stage_mixer(self):
        import builtins
        tree=ast.parse((ROOT/'train_semi.py').read_text())
        selector=next(n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='get_saliency_guided_boxes')
        mixer=next(n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='cut_mix_label_adaptive_with_mask' and any(k.arg=='direct_labeled_mix' for k in n.keywords) and not any(k.arg=='return_target_metadata' for k in n.keywords))
        names={n.id for call in (selector,mixer) for n in ast.walk(call) if isinstance(n,ast.Name)}
        for row in ops.profiles():
            cfg=ops.expected_config(row);self.assertTrue(sc.validate_s1_ablation_config(cfg))
            u=torch.zeros(2,3,4,4);l=torch.stack([torch.full((3,4,4),10.),torch.full((3,4,4),20.)]);gt=torch.stack([torch.full((4,4),3),torch.full((4,4),4)]);gt[0,0,0]=255
            scope={n:None for n in names if not hasattr(builtins,n)}
            scope.update(cfg=cfg,np=np,torch=torch,image_l=l,label_l=gt,image_u_aug=u,label_u_aug=torch.zeros_like(gt),logits_u_aug=torch.full((2,4,4),.6),confidence=[0.,0.],saliency_cutmix_cfg=cfg['saliency_cutmix'],fixed_size_csl_destination_cfg={},destination_kwargs={},get_saliency_guided_boxes=sc.get_saliency_guided_boxes,cut_mix_label_adaptive_with_mask=bm.cut_mix_label_adaptive_with_mask,_rand_bbox=bm._rand_bbox)
            pool=torch.tensor([0,0,4,4]).reshape(1,1,4).repeat(2,8,1);coords=tuple(np.array([v,v]) for v in (0,0,4,4))
            with mock.patch.object(sc,'compute_labeled_teacher_saliency',return_value=torch.ones(2,4,4)),mock.patch.object(sc,'sample_candidate_boxes_with_baseline_sampler',return_value=pool),mock.patch.object(bm.torch,'randperm',return_value=torch.tensor([1,0])) as perm,mock.patch.object(bm,'_rand_bbox',return_value=coords) as boxes,mock.patch.object(bm.np.random,'beta',return_value=.5),mock.patch.object(bm.np.random,'random',return_value=.5) as gates:
                scope['saliency_labeled_boxes'],_=eval(compile(ast.Expression(selector),'production_selector','eval'),scope)
                mixed,target,q,*_=eval(compile(ast.Expression(mixer),'production_mixer','eval'),scope)
            self.assertEqual(perm.call_count,1);self.assertEqual(boxes.call_count,1);self.assertEqual(gates.call_count,2)
            self.assertTrue(mixed[0].eq(10).all());self.assertTrue(mixed[1].eq(20).all());self.assertEqual(target[0,0,0].item(),255);self.assertTrue(q.eq(1).all())

    def test_baseline_mixer_default_unchanged(self):
        self.assertEqual((ROOT/'util/boundary_mix.py').read_bytes(),ops.git('show',ops.BASE+':util/boundary_mix.py'))
        u=torch.arange(2*3*8*8).reshape(2,3,8,8).float();y=torch.zeros(2,8,8,dtype=torch.long);q=torch.full((2,8,8),.6);l=u+1000
        result=[]
        for mixer in (cut_mix_label_adaptive,bm.cut_mix_label_adaptive_with_mask):
            np.random.seed(37);torch.manual_seed(41);result.append(mixer(u.clone(),y.clone(),q.clone(),l,y,[.2,.8]))
        for a,b in zip(result[0],result[1][:3]):self.assertTrue(torch.equal(a,b))

    def test_threshold_grid_denominator_gt_ignore_and_weight(self):
        z=torch.zeros(1,21,1,4);y=torch.tensor([[[2,2,3,255]]]);q=torch.tensor([[[.95,.949,1.,1.]]])
        value,active=compute_unsupervised_loss_by_threshold(z,y.clone(),q)
        self.assertAlmostEqual(value.item(),2*float(np.log(21))/4,places=6);self.assertEqual(active.item(),.5)
        for row in ops.profiles():self.assertEqual(ops.expected_config(row)['trainer']['unsupervised']['loss_weight'],1.)

    def test_strict_failure_executes_real_handler_before_fallback(self):
        tree=ast.parse((ROOT/'train_semi.py').read_text())
        handler=next(n for n in ast.walk(tree) if isinstance(n,ast.ExceptHandler) and any(isinstance(b,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='saliency_selector_exception_batch' for t in b.targets) for b in n.body))
        statement=ast.Try(body=[ast.Raise(exc=ast.Call(func=ast.Name(id='ValueError',ctx=ast.Load()),args=[ast.Constant('synthetic')],keywords=[]),cause=None)],handlers=[handler],orelse=[],finalbody=[])
        code=compile(ast.fix_missing_locations(ast.Module(body=[statement],type_ignores=[])),'actual_s1_handler','exec')
        scope={'saliency_cutmix_cfg':{'strict_selector':True},'saliency_labeled_boxes':'unchanged'}
        with self.assertRaisesRegex(RuntimeError,'without pipeline fallback'):exec(code,scope)
        self.assertEqual(scope['saliency_labeled_boxes'],'unchanged')
if __name__=="__main__":unittest.main()

"""CPU synthetic/mocks only; no trainer import, GPU, real checkpoint or remote writes."""
import ast
import copy
import csv
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from util import saliency_cutmix as sc
from util import boundary_mix as bm
from augseg.dataset.augs_ALIA import cut_mix_label_adaptive, rand_bbox
from augseg.utils.loss_helper import compute_unsupervised_loss_by_threshold

spec = importlib.util.spec_from_file_location("s1_operations", ROOT / "tools/s1_two_stage_cohort.py")
ops = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ops)
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


class OperationsContract(unittest.TestCase):
    def test_bs16_batch_and_runtime_loader_guards(self):
        cfg=ops.expected_config(ops.profiles()[0])
        self.assertEqual(cfg['dataset']['train']['batch_size'],16)
        cfg['dataset']['train']['batch_size']=8
        with self.assertRaises(ValueError):sc.validate_s1_ablation_config(cfg)
        tree=ast.parse((ROOT/'train_semi.py').read_text())
        guard=next(n for n in ast.walk(tree) if isinstance(n,ast.If) and
            ast.unparse(n.test).startswith('s1_ablation and') and 'len(train_loader_sup)' in ast.unparse(n.test))
        program=compile(ast.Module(body=[guard],type_ignores=[]),'<actual-runtime-loader-guard>','exec')
        namespace={'s1_ablation':True,'train_loader_sup':range(620),'train_loader_unsup':range(620),'val_loader':range(1449)}
        exec(program,namespace)
        namespace['train_loader_sup']=range(1240)
        with self.assertRaises(RuntimeError):exec(program,namespace)

    def test_configs_only_declared_scientific_difference(self):
        configs=[ops.expected_config(r) for r in ops.profiles()]
        old=yaml.safe_load(ops.git('show',ops.BASE+':'+ops.OLD_CONFIG))
        for cfg in configs:
            self.assertTrue(sc.validate_s1_ablation_config(cfg))
            for group in ('dataset','trainer','net','criterion','csl','csl_cutmix','boundary_mix','boundary_component','boundary_compatibility'):
                self.assertEqual(cfg[group],old[group])
        self.assertFalse(configs[0]['checkpoint']['auto_resume'])
        self.assertEqual(configs[0]['dataset']['train']['batch_size'],16)
        a,b=[copy.deepcopy(c['saliency_cutmix']) for c in configs]
        self.assertNotEqual(a.pop('candidate_selection'),b.pop('candidate_selection'));self.assertEqual(a,b)
        bad=copy.deepcopy(configs[0]);bad['saliency_cutmix']['strict_selector']=False
        with self.assertRaises(ValueError):sc.validate_s1_ablation_config(bad)
        self.assertFalse(sc.validate_s1_ablation_config(old))

    def test_pins_and_independent_identity_without_production_state(self):
        self.assertFalse(ops.STATE.exists());ops.validate()
        rows=ops.profiles()
        for r in rows:ops.local_fresh(r)
        for field in ('run_id','config','output','launch_log','hf_path'):self.assertNotEqual(rows[0][field],rows[1][field])
        self.assertFalse(ops.STATE.exists())

    def test_tamper_and_collision_fail_closed(self):
        original=ops.sha_file
        with mock.patch.object(ops,'sha_file',side_effect=lambda p:'tampered' if str(p).endswith('train_semi.py') else original(p)):
            with self.assertRaises(ops.CohortError):ops.validate()
        with tempfile.TemporaryDirectory() as d:
            row=copy.deepcopy(ops.profiles()[0]);row['output']=d
            with self.assertRaises(ops.CohortError):ops.local_fresh(row)

    def test_resource_gate_preserves_protected_queue_priority(self):
        with mock.patch.object(ops.subprocess,'check_output',side_effect=['32000,0','']),mock.patch.object(ops.subprocess,'run',return_value=SimpleNamespace(returncode=0,stdout='active\n')),mock.patch.object(ops.shutil,'disk_usage',return_value=SimpleNamespace(free=100*1024**3)):
            self.assertFalse(ops.resources()['launch_resource_ready'])
        with mock.patch.object(ops.subprocess,'check_output',side_effect=['16383,0','']),mock.patch.object(ops.subprocess,'run',return_value=SimpleNamespace(returncode=0,stdout='inactive\n')),mock.patch.object(ops.shutil,'disk_usage',return_value=SimpleNamespace(free=100*1024**3)):
            self.assertFalse(ops.resources()['launch_resource_ready'])

    def test_gpu_threshold_boundary_and_other_guards_unchanged(self):
        self.assertEqual(ops.MIN_GPU_FREE_MIB,16384)
        for free,pids,disk,state,expected in (
            (16383,'',100*1024**3,'inactive',False),
            (16384,'',100*1024**3,'inactive',True),
            (16384,'12345',100*1024**3,'inactive',True),
            (16384,'',31*1024**3,'inactive',False),
            (16384,'',100*1024**3,'active',False),
            (16384,'',100*1024**3,'UNKNOWN',False)):
            with mock.patch.object(ops.subprocess,'check_output',side_effect=[str(free)+',0',pids]),mock.patch.object(ops.subprocess,'run',return_value=SimpleNamespace(returncode=0,stdout=state+'\n')),mock.patch.object(ops.shutil,'disk_usage',return_value=SimpleNamespace(free=disk)):
                result=ops.resources()
                self.assertEqual(result['gpu_min_free_mib'],16384)
                self.assertEqual(result['launch_resource_ready'],expected)
                self.assertEqual(result['compute_pids'],pids.splitlines())
                self.assertIn('--id=0',ops.subprocess.check_output.call_args.args[0])

    def test_launch_dispatch_is_single_profile_no_enqueue(self):
        row=ops.profiles()[0]
        with mock.patch.object(sys,'argv',['cohort','launch','--method',row['method']]),mock.patch.object(ops,'launch',return_value={}) as launch,mock.patch('sys.stdout',new=io.StringIO()):
            ops.main();launch.assert_called_once_with(row)

    def test_finalize_without_state_cannot_train_or_create_run(self):
        with mock.patch.object(ops.subprocess,'run',side_effect=AssertionError('training forbidden')):
            with self.assertRaises(ops.CohortError):ops.finalize(ops.profiles()[0])
        self.assertFalse(ops.STATE.exists())

    def test_finalization_failure_separate_from_training(self):
        with tempfile.TemporaryDirectory() as d:
            with mock.patch.object(ops,'STATE',Path(d)),mock.patch.object(ops,'_finalize',side_effect=RuntimeError('synthetic')),mock.patch.object(ops.subprocess,'run',side_effect=AssertionError('training forbidden')):
                with self.assertRaises(RuntimeError):ops.finalize(ops.profiles()[0])
            self.assertTrue((Path(d)/(ops.profiles()[0]['method']+'.finalization_failed')).exists())
            self.assertFalse((Path(d)/(ops.profiles()[0]['method']+'.training_failed')).exists())

    def test_completed80_checks_real_csv_contract_without_checkpoint_load(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'output';out.mkdir();state=Path(d)/'state';state.mkdir()
            row=copy.deepcopy(ops.profiles()[0]);row['output']=str(out)
            (state/(row['method']+'.training_done')).write_text(json.dumps({'run_id':row['run_id']}))
            for name in ops.CANONICAL+('source.diff','reviewed-deployment.json'):(out/name).write_text('synthetic, no tensor storage')
            (out/'config.yaml').write_bytes(Path(row['config']).read_bytes());(out/'run_id.txt').write_text(row['run_id']+'\n')
            pin=json.loads(ops.PIN.read_text());sha=ops.sha_file(ops.PIN)
            (out/'provenance.json').write_text(json.dumps({'execution_id':row['run_id'],'source_pin_sha256':sha,'source_hashes':pin['file_hashes']}))
            for rel in pin['file_hashes']:
                p=out/'source_snapshot'/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/rel).read_bytes())
            epochs=[{'epoch':e,'lr':.000125*(1-((e+1)*620-1)/49600)**.9,'run_id':row['run_id'],'git_commit':ops.BASE[:7],'world_size':1,'miou_stu':.7,'miou_tea':.71,'source_pin_sha256':sha,'s1_candidate_selection':row['mode'],'s1_fallback_policy':'abort'} for e in range(80)]
            def save_epochs():
                with (out/'epoch_metrics.csv').open('w',newline='') as f:
                    w=csv.DictWriter(f,fieldnames=list(epochs[0]));w.writeheader();w.writerows(epochs)
            save_epochs();(out/'manifest.json').write_text(json.dumps({'epoch':79,'run_id':row['run_id']}))
            iterations=[]
            for e in range(80):
                for model,value in (('student',.7),('teacher',.71)):
                    iterations.append({'meta/log_type':'val_epoch','meta/global_iter':(e+1)*620-1,'val/model':model,'val/miou':value})
            iterations.append({'meta/log_type':'train_iter','meta/global_iter':0,'saliency/num_candidates':8,'saliency/temperature':.2,'saliency/fallback_ratio':0,'saliency/candidate_policy_id':0})
            columns=sorted(set().union(*(r.keys() for r in iterations)))
            with (out/'iter_metrics.csv').open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(iterations)
            with mock.patch.object(ops,'STATE',state),mock.patch.object(ops,'checkpoint_scalars',side_effect=lambda p:{'run_id':row['run_id'],'epoch':1 if p.name=='ckpt_best.pth' else 80}):
                self.assertEqual(len(ops.completed(row)),80)
                epochs[79]['lr']=0;save_epochs()
                with self.assertRaisesRegex(ops.CohortError,'LR trajectory'):ops.completed(row)

    def test_snapshot_parent_label_requires_matching_attestation(self):
        from util.run_logging import get_git_commit
        with mock.patch.dict('os.environ',{'AUGSEG_DEPLOYMENT_PIN':str(ops.PIN),'AUGSEG_DEPLOYMENT_PIN_SHA256':ops.sha_file(ops.PIN)}):
            self.assertEqual(get_git_commit(),ops.BASE[:7])
        with mock.patch.dict('os.environ',{'AUGSEG_DEPLOYMENT_PIN':str(ops.PIN),'AUGSEG_DEPLOYMENT_PIN_SHA256':'wrong'}):
            with self.assertRaises(RuntimeError):get_git_commit()

    def test_finalization_receipt_idempotence_with_mock_remote(self):
        with tempfile.TemporaryDirectory() as d:
            state=Path(d)/'state';state.mkdir();out=Path(d)/'out';out.mkdir()
            row=copy.deepcopy(ops.profiles()[0]);row['output']=str(out);row['launch_log']=str(Path(d)/'train.log');Path(row['launch_log']).write_text('synthetic')
            for name in ops.CANONICAL+('provenance.json','source.diff','reviewed-deployment.json'):(out/name).write_text('synthetic')
            (out/'source_snapshot').mkdir();(out/'source_snapshot/source.py').write_text('# synthetic')
            cfg=ops.expected_config(row);cfg['run']['source_pin_sha256']=ops.sha_file(ops.PIN);run=SimpleNamespace(config=cfg,state='finished')
            rows=[{'epoch':str(e),'miou_stu':str(e/100),'miou_tea':str(e/101)} for e in range(80)]
            remote=[];hf=mock.Mock();hf.repo_info.return_value=SimpleNamespace(sha='parent');hf.get_paths_info.side_effect=lambda *a,**k:remote
            def upload(**kw):
                p=Path(kw['path_or_fileobj']);remote.append(SimpleNamespace(size=p.stat().st_size,lfs=SimpleNamespace(sha256=ops.sha_file(p)),last_commit=SimpleNamespace(oid='uploaded')));return SimpleNamespace(oid='uploaded')
            hf.upload_file.side_effect=upload;wb=SimpleNamespace(Api=lambda **kw:SimpleNamespace(run=lambda p:run))
            original=ops.subprocess.run
            def git_only(command,*a,**k):
                if command[0]!='git':raise AssertionError('finalize cannot train')
                return original(command,*a,**k)
            with mock.patch.object(ops,'STATE',state),mock.patch.object(ops,'validate'),mock.patch.object(ops,'completed',return_value=rows),mock.patch.object(ops,'hf_api',return_value=hf),mock.patch.object(ops,'legacy_helpers',return_value=SimpleNamespace(validate_wandb_completion=mock.Mock())),mock.patch.dict(sys.modules,{'wandb':wb}),mock.patch.object(ops.subprocess,'run',side_effect=git_only):
                first=ops._finalize(row);second=ops._finalize(row)
            self.assertEqual(first,second);self.assertEqual(hf.upload_file.call_count,1);self.assertTrue((state/(row['method']+'.done')).exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)

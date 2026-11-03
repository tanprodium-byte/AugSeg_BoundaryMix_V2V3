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

spec = importlib.util.spec_from_file_location("s2_operations", ROOT / "tools/s2_area_only_cohort.py")
ops = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ops)
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


class OperationsContract(unittest.TestCase):
    def test_bs16_batch_and_runtime_loader_guards(self):
        cfg=ops.expected_config(ops.profiles()[0])
        self.assertEqual(cfg['dataset']['train']['batch_size'],16)
        cfg['dataset']['train']['batch_size']=8
        with self.assertRaises(ValueError):sc.validate_s2_area_only_config(cfg)
        tree=ast.parse((ROOT/'train_semi.py').read_text())
        guard=next(n for n in ast.walk(tree) if isinstance(n,ast.If) and
            ast.unparse(n.test).startswith('s2_area_only and') and 'len(train_loader_sup)' in ast.unparse(n.test))
        program=compile(ast.Module(body=[guard],type_ignores=[]),'<actual-runtime-loader-guard>','exec')
        namespace={'s2_area_only':True,'train_loader_sup':range(620),'train_loader_unsup':range(620),'val_loader':range(1449)}
        exec(program,namespace)
        namespace['train_loader_sup']=range(1240)
        with self.assertRaises(RuntimeError):exec(program,namespace)

    def test_config_changes_only_score_and_operational_fields(self):
        cfg=ops.expected_config(ops.profiles()[0]);old=yaml.safe_load(ops.git('show',ops.BASE+':'+ops.OLD_CONFIG))
        self.assertTrue(sc.validate_s2_area_only_config(cfg));self.assertFalse(sc.validate_s2_area_only_config(old))
        for group in ('dataset','trainer','net','criterion','checkpoint','csl','boundary_mix','boundary_component','boundary_compatibility'):
            self.assertEqual(cfg[group],old[group])
        sal=copy.deepcopy(cfg['saliency_cutmix']);sal['component_score']='mean_saliency_area';self.assertEqual(sal,old['saliency_cutmix'])
        bad=copy.deepcopy(cfg);bad['saliency_cutmix']['direct_confidence_gate']=True
        with self.assertRaises(ValueError):sc.validate_s2_area_only_config(bad)

    def test_pins_and_distinct_identity_without_production_state(self):
        self.assertFalse(ops.STATE.exists());ops.validate();row=ops.profiles()[0];ops.local_fresh(row)
        oldroot=ROOT.parent/'s1_two_stage_k8_bs8_20261028'
        reg=yaml.safe_load((oldroot/'configs/experiment_registry_voc662_s1_two_stage_k8_bs8_20261028.yaml').read_text())
        for prior in reg['entries']:
            for field in ('run_id','config','output','launch_log','hf_path','state'):self.assertNotEqual(row[field],prior[field])
        self.assertNotEqual(row['run_id'],ops.CONTROL_ID);self.assertFalse(ops.STATE.exists())

    def test_tamper_and_collision_fail_closed(self):
        original=ops.sha_file
        with mock.patch.object(ops,'sha_file',side_effect=lambda p:'tampered' if str(p).endswith('train_semi.py') else original(p)):
            with self.assertRaises(ops.CohortError):ops.validate()
        with tempfile.TemporaryDirectory() as d:
            row=copy.deepcopy(ops.profiles()[0]);row['output']=d
            with self.assertRaises(ops.CohortError):ops.local_fresh(row)

    def test_input_symlink_retarget_is_rejected(self):
        with mock.patch.object(Path,'readlink',return_value=Path('/synthetic/wrong-dataset')):
            with self.assertRaisesRegex(ops.CohortError,'link changed'):ops.validate()

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
            epochs=[{'epoch':e,'lr':.000125*(1-((e+1)*620-1)/49600)**.9,'run_id':row['run_id'],'git_commit':ops.BASE[:7],'world_size':1,'miou_stu':.7,'miou_tea':.71,'source_pin_sha256':sha,'s2_component_score':row['mode'],'s2_fallback_policy':'random_box_no_eligible;abort_selector_exception'} for e in range(80)]
            def save_epochs():
                with (out/'epoch_metrics.csv').open('w',newline='') as f:
                    w=csv.DictWriter(f,fieldnames=list(epochs[0]));w.writeheader();w.writerows(epochs)
            save_epochs();(out/'manifest.json').write_text(json.dumps({'epoch':79,'run_id':row['run_id']}))
            iterations=[]
            for e in range(80):
                for model,value in (('student',.7),('teacher',.71)):
                    iterations.append({'meta/log_type':'val_epoch','meta/global_iter':(e+1)*620-1,'val/model':model,'val/miou':value})
            iterations.append({'meta/log_type':'train_iter','meta/global_iter':0,'saliency/temperature':.2,'saliency/fallback_ratio':.25,'saliency/component_score_mode_id':1,'saliency_selector_exception_batch':0,'s2/mixing_stages':1,'s2/component_fallback_sample_fraction':.25,'s2/grid_pixels':16*321*321,'s2/gt_origin_pixels':0,'s2/valid_gt_origin_pixels':0,'s2/ignore_gt_origin_pixels':0,'s2/gt_origin_fraction':0,'s2/valid_gt_origin_fraction':0,'s2/ignore_gt_origin_fraction':0})
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

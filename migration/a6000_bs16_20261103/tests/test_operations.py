"""CPU/mocks only; temporary artifacts never use production roots or remote APIs."""
import ast,contextlib,copy,csv,hashlib,io,json,os,shutil,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import yaml
PKG=Path(__file__).resolve().parents[1];sys.path.insert(0,str(PKG))
import controller as c
class Operations(unittest.TestCase):
    def test_package_science_and_identity(self):
        self.assertEqual(c.check_package()['scientific_source_byte_diff'],0)
        self.assertEqual([r['run_id'] for r in c.profiles()],[
            'a6000_20261102T181658Z_baseline_augseg_441920d4',
            'a6000_20261102T181658Z_s1_two_stage_saliency_k8_d1fa9f27',
            'a6000_20261102T181658Z_s1_two_stage_random_k8_4bc977bc',
            'a6000_20261102T181658Z_s2_mask_relocated_area_only_ff7f3cb4'])
    def test_package_tamper_or_unlisted_file(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)/'package';shutil.copytree(PKG,root)
            with mock.patch.object(c,'PKG',root):
                (root/'unexpected.txt').write_text('unexpected')
                with self.assertRaises(c.CohortError):c.check_package()
                (root/'unexpected.txt').unlink();(root/'sources/s1/util/saliency_cutmix.py').write_text('changed')
                with self.assertRaises(c.CohortError):c.check_package()
    def test_environment_metadata_only(self):
        self.assertEqual(c.validate_environment()['torch'],'2.11.0+cu128')
        with mock.patch.object(c.importlib.metadata,'version',return_value='wrong'):
            with self.assertRaises(c.CohortError):c.validate_environment()
    def test_source_snapshot_roots_and_parent_preserved(self):
        self.assertNotEqual(c.REG['profiles'][1]['source_root'],c.REG['profiles'][3]['source_root'])
        self.assertEqual(c.ORIGINS['sources']['s2']['parent_commit'],'9047b1a235541917d77f133401a2547651e7c0e1')
        self.assertTrue((PKG/'historical/git_exports/9047b1a235541917d77f133401a2547651e7c0e1/train_semi.py').is_file())
    def test_wrong_target_bootstrap_fails_before_write(self):
        # Deliberately use a wrong package location on every host, including A6000.
        # Never rely on the real host being different from production.
        with tempfile.TemporaryDirectory() as td, mock.patch.object(c,'PKG',Path(td)), mock.patch.object(c,'check_package'), mock.patch.object(c.shutil,'copytree') as cp:
            with self.assertRaises(c.CohortError):c.bootstrap()
            cp.assert_not_called()
    def test_tags_and_fresh_online_only(self):
        for row in c.profiles():
            cfg=c.expected_config(row)
            self.assertTrue(all(len(t)<=64 for t in cfg['wandb']['tags']))
            self.assertEqual(cfg['wandb']['resume'],'never');self.assertFalse(cfg['checkpoint']['auto_resume'])
            self.assertFalse(cfg['hf']['auto_download']);self.assertFalse(cfg['hf']['upload_every_epoch'])
            self.assertIn('control_execution_id',cfg['run'])
    def test_runtime_defaults_and_eps_equivalence(self):
        for row in c.profiles():
            row['source_root']=str(PKG/'sources'/row['source']);row['output']=str(PKG/'nonexistent_test_output')
            cfg=c.expected_config(row);actual=copy.deepcopy(cfg)
            if 'eps' in actual.get('saliency_cutmix',{}):actual['saliency_cutmix']['eps']='1e-6'
            self.assertTrue(c.validate_runtime_config(actual,cfg,row))
    def test_routing_threshold_loss_and_unknown_key_rejected(self):
        row=c.profiles()[1];row['source_root']=str(PKG/'sources/s1');row['output']=str(PKG/'nonexistent_test_output');cfg=c.expected_config(row)
        for group,key,value in [('trainer','extra_science',True),('saliency_cutmix','candidate_selection','uniform_multinomial'),('boundary_compatibility','enabled',True),('saliency_cutmix','eps','nan')]:
            actual=copy.deepcopy(cfg);actual[group][key]=value
            with self.assertRaises(c.CohortError):c.validate_runtime_config(actual,cfg,row)
        for key,value in [('threshold',.9),('loss_weight',2)]:
            actual=copy.deepcopy(cfg);actual['trainer']['unsupervised'][key]=value
            with self.assertRaises(c.CohortError):c.validate_runtime_config(actual,cfg,row)
    def test_bounded_read_retry_and_persistent_auth_hold(self):
        Auth=type('AuthenticationError',(Exception,),{})
        with mock.patch.object(c.time,'sleep') as sleep,contextlib.redirect_stdout(io.StringIO()):
            call=mock.Mock(side_effect=[Auth('transient'),True]);self.assertTrue(c.readonly_retry(call));self.assertEqual(call.call_count,2);sleep.assert_called_once_with(15)
            call=mock.Mock(side_effect=Auth('persistent'))
            with self.assertRaises(Auth):c.readonly_retry(call)
            self.assertEqual(call.call_count,3)
    def test_scientific_mismatch_not_retried(self):
        call=mock.Mock(side_effect=c.CohortError('routing mismatch'))
        with mock.patch.object(c.time,'sleep') as sleep:
            with self.assertRaises(c.CohortError):c.readonly_retry(call)
            sleep.assert_not_called();self.assertEqual(call.call_count,1)
    def test_resource_thresholds_and_model(self):
        with mock.patch.object(c.subprocess,'check_output',return_value='NVIDIA RTX A6000, 16384\n'),mock.patch.object(c.shutil,'disk_usage',return_value=SimpleNamespace(free=32*1024**3)):
            self.assertTrue(c.resources()['ready'])
        with mock.patch.object(c.subprocess,'check_output',return_value='NVIDIA RTX A6000, 16383\n'),mock.patch.object(c.shutil,'disk_usage',return_value=SimpleNamespace(free=32*1024**3)):
            self.assertFalse(c.resources()['ready'])
        with mock.patch.object(c.subprocess,'check_output',return_value='RTX5090, 32000\n'),mock.patch.object(c.shutil,'disk_usage',return_value=SimpleNamespace(free=99*1024**3)):
            self.assertFalse(c.resources()['ready'])
    def test_predecessor_order(self):
        with mock.patch.object(c,'verify_done') as verify:
            c.predecessor(c.profiles()[0]);verify.assert_not_called()
            for i in range(1,4):c.predecessor(c.profiles()[i]);self.assertEqual(verify.call_args.args[0]['run_id'],c.profiles()[i-1]['run_id'])
    def test_started_and_completed_cannot_retrain(self):
        with tempfile.TemporaryDirectory() as td:
            r=c.profiles()[0];r.update(output=str(Path(td)/'out'),log=str(Path(td)/'log'),state_prefix=str(Path(td)/'run'))
            for key in ['started','training_done','training_failed','done']:
                p=c.marker(r,key);p.write_text('{}')
                with mock.patch.object(c,'preflight',return_value={}),mock.patch.object(c,'lock',return_value=contextlib.nullcontext()),mock.patch.object(c,'validate_installed'),mock.patch.object(c.subprocess,'run') as train:
                    with self.assertRaises(c.CohortError):c.launch(r)
                    train.assert_not_called()
                p.unlink()
    def test_finalize_without_training_cannot_train(self):
        with tempfile.TemporaryDirectory() as td:
            r=c.profiles()[0];r.update(output=str(Path(td)/'out'),state_prefix=str(Path(td)/'run'))
            with mock.patch.object(c,'validate_installed'),mock.patch.object(c,'lock',return_value=contextlib.nullcontext()),mock.patch.object(c.subprocess,'run') as train,mock.patch.object(c,'clients') as remote:
                with self.assertRaises(c.CohortError):c.finalize(r)
                train.assert_not_called();remote.assert_not_called();self.assertTrue(c.marker(r,'finalization_failed').exists())
    def test_finalizer_has_no_training_or_launch_call(self):
        t=ast.parse((PKG/'controller.py').read_text());n=next(x for x in t.body if isinstance(x,ast.FunctionDef) and x.name=='finalize')
        calls=[ast.unparse(x.func) for x in ast.walk(n) if isinstance(x,ast.Call)]
        self.assertFalse(any(x in calls for x in ['launch','train_command','subprocess.run']))
    def test_finalization_receipt_and_recovery_no_second_upload(self):
        with tempfile.TemporaryDirectory() as td:
            r=c.profiles()[0];r.update(output=str(Path(td)/'out'),log=str(Path(td)/'log'),state_prefix=str(Path(td)/'state/run'))
            o=Path(r['output']);o.mkdir();Path(r['log']).write_text('mock log, no trainer')
            for fn in ['ckpt.pth','ckpt_best.pth','config.yaml','run_id.txt','manifest.json','epoch_metrics.csv','iter_metrics.csv','provenance.json','package-SHA256.json']:(o/fn).write_text('SYNTHETIC-NOT-TENSOR')
            (o/'source_snapshot').mkdir();(o/'source_snapshot/code.py').write_text('pass')
            rows=[{'epoch':str(e),'miou_stu':'.70','miou_tea':'.71'} for e in range(80)];run=SimpleNamespace(config={'run':{'name':r['run_id']}},state='finished');stored=[];uploads=[]
            def upload(**kw):
                uploads.append(kw);f=Path(kw['path_or_fileobj']);stored.append(SimpleNamespace(size=f.stat().st_size,lfs=SimpleNamespace(sha256=c.sha_file(f)),last_commit=SimpleNamespace(oid='a'*40)));return SimpleNamespace(oid='a'*40)
            hf=SimpleNamespace(repo_info=lambda *a,**k:SimpleNamespace(sha='b'*40),get_paths_info=lambda *a,**k:stored,upload_file=upload)
            with mock.patch.object(c,'validate_installed'),mock.patch.object(c,'lock',return_value=contextlib.nullcontext()),mock.patch.object(c,'completed',return_value=rows),mock.patch.object(c,'clients',return_value=(None,hf)),mock.patch.object(c,'verify_wandb',return_value=run),mock.patch.object(c.subprocess,'run') as train:
                receipt=c.finalize(r);self.assertEqual(len(uploads),1);self.assertEqual(receipt['run_id'],r['run_id']);self.assertTrue(c.marker(r,'done').exists())
                self.assertEqual(c.finalize(r),receipt);self.assertEqual(len(uploads),1);train.assert_not_called()
                stored.clear()
                with self.assertRaisesRegex(c.CohortError,'uncertain'):c.finalize(r)
                self.assertEqual(len(uploads),1);train.assert_not_called()
    def test_remote_collision_and_auth_error_not_absence(self):
        r=c.profiles()[0];wb=SimpleNamespace(project=lambda *a,**k:object(),run=lambda *a:object());hf=SimpleNamespace()
        with mock.patch.object(c,'clients',return_value=(wb,hf)):
            with self.assertRaisesRegex(c.CohortError,'already exists'):c.remote_fresh(r)
        Auth=type('AuthenticationError',(Exception,),{})
        def fail(*a):raise Auth('unauthorized')
        wb.run=fail
        with mock.patch.object(c,'clients',return_value=(wb,hf)),mock.patch.object(c.time,'sleep'),contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(c.CohortError,'absence not verified'):c.remote_fresh(r)
    def test_upload_error_never_training_retry(self):
        with tempfile.TemporaryDirectory() as td:
            r=c.profiles()[0];r.update(state_prefix=str(Path(td)/'run'))
            with mock.patch.object(c,'validate_installed'),mock.patch.object(c,'lock',return_value=contextlib.nullcontext()),mock.patch.object(c,'completed',return_value=[]),mock.patch.object(c,'clients',side_effect=RuntimeError('remote unavailable')),mock.patch.object(c.subprocess,'run') as train:
                with self.assertRaises(RuntimeError):c.finalize(r)
                self.assertFalse(c.marker(r,'training_failed').exists());self.assertTrue(c.marker(r,'finalization_failed').exists());train.assert_not_called()
if __name__=='__main__':unittest.main()

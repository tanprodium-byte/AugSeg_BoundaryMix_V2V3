"""Portable A6000 operations. No trainer import; no CUDA use until explicit launch.
check-package is host-independent and read-only. preflight is read-only.
bootstrap/launch/finalize are explicit actions; never an automatic queue or retry.
"""
from __future__ import annotations
import pickletools,zipfile
import argparse,ast,contextlib,copy,csv,datetime,fcntl,hashlib,importlib.metadata,importlib.util,json,math,os,re,shutil,subprocess,sys,tarfile,time
from pathlib import Path
PKG=Path(__file__).resolve().parent
REG=json.loads((PKG/'registry.json').read_text())
ORIGINS=json.loads((PKG/'origins.json').read_text())
class CohortError(RuntimeError): pass
def require(v,m):
    if not v: raise CohortError(m)
def utc(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha_file(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for c in iter(lambda:f.read(1024*1024),b''):h.update(c)
    return h.hexdigest()
def read(p): return json.loads(Path(p).read_text())
def atomic(p,v,exclusive=False):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    if exclusive:
        with p.open('x') as f:json.dump(v,f,indent=2,sort_keys=True);f.write('\n')
    else:
        tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n');os.replace(tmp,p)
def same_or_new(p,v):
    p=Path(p)
    if p.exists():require(read(p)==v,'existing artifact differs: '+p.name)
    else:atomic(p,v,True)
def profiles(): return copy.deepcopy(REG['profiles'])
def row_for(method): return next(r for r in profiles() if r['method']==method)
def expected_config(r):
    import yaml
    return yaml.safe_load((PKG/r['package_config']).read_text())
def marker(r,suffix):return Path(r['state_prefix']+'.'+suffix)
def runtime_pin(r):return Path(r['source_root'])/'migration-source-pin.json'
def source_pin(r):return {'source_commit':ORIGINS['sources'][r['source']]['parent_commit'],'deployment_root':r['source_root'],'file_hashes':ORIGINS['sources'][r['source']]['source_files'],'attestation':'package-prepared; NOT historical or A6000 runtime attestation'}
def check_package():
    manifest=read(PKG/'SHA256.json');actual={str(p.relative_to(PKG)) for p in PKG.rglob('*') if p.is_file() and p!=PKG/'SHA256.json'}
    require(actual==set(manifest),'package missing/extra files')
    for rel,h in manifest.items():
        p=PKG/rel;require(not p.is_symlink() and sha_file(p)==h,'package checksum: '+rel)
    require(REG['target_root']=='/home/islabworker2/minhhb/tanvt','wrong tanvt root')
    require(len(profiles())==4 and len({r['run_id'] for r in profiles()})==4,'identity count/collision')
    require(len({r['output'] for r in profiles()})==4 and len({r['hf_path'] for r in profiles()})==4,'output/HF collision')
    for kind,s in ORIGINS['sources'].items():
        for rel,h in s['source_files'].items():
            require(sha_file(PKG/'sources'/kind/rel)==h,'source changed: '+kind+'/'+rel)
            require(sha_file(PKG/'historical'/kind/rel)==h,'source differs from immutable historical export')
    # Historical reviewed pins remain exact bytes, including the S2 parent label.
    for k in ('s1','s2'):
        p=next((PKG/'historical'/k/'configs').glob('*reviewed-deployment.json'))
        require(sha_file(p)==ORIGINS['sources'][k]['historical_pin_sha256'],'historical pin overwritten')
    for r in profiles():
        c=expected_config(r)
        require(c['dataset']['train']['batch_size']==16 and c['dataset']['n_sup']==662,'batch/split mismatch')
        require(c['dataset']['train']['crop']['size']==[321,321] and c['trainer']['epochs']==80 and c['trainer']['sup_only_epoch']==0,'protocol mismatch')
        require(r['updates']==49600 and r['iterations_per_epoch']==620 and r['world_size']==1 and r['seed']==2,'fixed80 protocol')
        require(c['checkpoint']['auto_resume'] is False and c['saver']['auto_resume'] is False,'resume forbidden')
        require(c['wandb']['enable'] and c['wandb']['mode']=='online' and c['wandb']['resume']=='never','online fresh required')
        require(all(0<len(t)<=64 for t in c['wandb']['tags']),'invalid W&B tag')
        require(c['saver']['snapshot_dir']==r['output'] and c['wandb']['name']==r['run_id'],'identity mismatch')
        require(not c['hf']['auto_download'] and not c['hf']['auto_upload'] and not c['hf']['upload_every_epoch'],'HF must finalize separately')
        require(c['run']['control_execution_id']==r['historical_control_execution_id'],'full historical control ID lost')
        pin_bytes=(json.dumps(source_pin(r),indent=2,sort_keys=True)+'\n').encode()
        require(c['run']['source_pin_sha256']==hashlib.sha256(pin_bytes).hexdigest(),'config portable source pin mismatch')
        historical=read_historical_config(r);a=copy.deepcopy(c);z=copy.deepcopy(historical)
        for x in (a,z):
            for part in ('train','val'):x['dataset'][part]['data_root']='INPUT';x['dataset'][part]['data_list']='SPLIT'
            x['net']['encoder']['pretrain']='PRETRAIN'
        for group in ('dataset','net','trainer','criterion','saliency_cutmix','csl','csl_cutmix','boundary_mix','boundary_component','boundary_compatibility'):
            require(a.get(group)==z.get(group),'scientific config changed: '+r['method']+'/'+group)
    # Frozen controls are validation evidence, never production state.
    for name in ('baseline','s2'):
        e=PKG/'evidence/controls'/name
        with (e/'epoch_metrics.csv').open() as f:rr=list(csv.DictReader(f))
        require([int(t['epoch']) for t in rr]==list(range(80)),'control epoch coverage')
        rid=(e/'run_id.txt').read_text().strip();require(all(t['run_id']==rid for t in rr),'control identity')
        require(not read(e/'launch-observations.derived.json')['observations']==[],'control launch observations missing')
        require(not any('Resumed checkpoint' in t['text'] for t in read(e/'launch-observations.derived.json')['observations']),'control resume evidence')
        control_manifest=read(e/'manifest.derived.json')
        parent=ORIGINS['sources']['baseline' if name=='baseline' else 's2']['parent_commit']
        require(control_manifest['run_id']==rid and control_manifest['git_commit']==parent[:7] and control_manifest['epoch']==79 and control_manifest['world_size']==1,'control source binding')
        for t in rr:
            require(int(t['world_size'])==1,'control world size')
            expected_lr=.000125*(1-((int(t['epoch'])+1)*620-1)/49600)**.9
            require(math.isclose(float(t['lr']),expected_lr,rel_tol=1e-10),'control full80 LR mismatch')
        with (e/'iter_metrics.csv').open() as f:iterations=list(csv.DictReader(f))
        for model,col in [('student','miou_stu'),('teacher','miou_tea')]:
            obs=[v for v in iterations if v['meta/log_type']=='val_epoch' and v['val/model']==model]
            require([int(v['meta/global_iter']) for v in obs]==[(i+1)*620-1 for i in range(80)],'control raw validation coverage')
            require(all(math.isclose(float(v['val/miou']),float(t[col]),abs_tol=1e-7) for v,t in zip(obs,rr)),'control validation/raw epoch mismatch')
    return {'package':'PASS','scientific_source_byte_diff':0,'profiles':4,'runtime_A6000_attested':False}
def read_historical_config(r):
    import yaml
    if r['source']=='baseline':p=PKG/'historical/baseline/config.yaml'
    else:
        oldcohort='s1_two_stage_k8_bs16_20261030' if r['source']=='s1' else 's2_mask_relocated_area_only_bs16_20261030'
        p=PKG/'historical'/r['source']/'exps/boundary_mix_v2_v3'/('voc_semi662_'+oldcohort)/r['method']/'config.yaml'
    return yaml.safe_load(p.read_text())
def target_host():
    require(PKG.parents[1].resolve()==Path(REG['target_repo']).resolve(),'run bootstrap/production only inside exact tanvt target checkout')
    require(Path(REG['target_root']).is_dir(),'target root missing')
def bootstrap():
    check_package();target_host()
    for kind in ORIGINS['sources']:
        r=next(r for r in profiles() if r['source']==kind);dst=Path(r['source_root']);src=PKG/'sources'/kind
        if not dst.exists():shutil.copytree(src,dst)
        else:
            expected={str(p.relative_to(src)) for p in src.rglob('*') if p.is_file()}
            for rel in expected:require((dst/rel).is_file() and sha_file(dst/rel)==sha_file(src/rel),'bootstrap collision '+str(dst/rel))
        same_or_new(runtime_pin(r),source_pin(r))
    return {'bootstrap':'READY_FOR_INPUTS_AND_PREFLIGHT','training_launched':False}
def validate_installed(r):
    check_package();target_host();src=Path(r['source_root'])
    expected=source_pin(r);require(read(runtime_pin(r))==expected,'runtime source pin changed')
    for rel,h in expected['file_hashes'].items():require((src/rel).is_file() and not (src/rel).is_symlink() and sha_file(src/rel)==h,'installed source changed '+rel)
    require(Path(r['config']).read_bytes()==(PKG/r['package_config']).read_bytes(),'installed config changed')
    allowed={rel for rel in expected['file_hashes'] if rel.endswith('.py')}
    actual={str(p.relative_to(src)) for p in src.rglob('*.py') if 'log' not in p.relative_to(src).parts and '__pycache__' not in p.relative_to(src).parts}
    require(actual==allowed,'unexpected installed Python source')
def fresh(r):
    for p in [Path(r['output']),Path(r['log'])]+[marker(r,k) for k in ('started','training_done','training_failed','finalization_failed','done','upload_intent.json','upload_receipt.json')]:
        require(not os.path.lexists(p),'existing execution: '+str(p))
def inputs():
    info=read(PKG/'inputs.json')
    for v in info['files']:
        p=Path(v['destination']);require(p.is_file() and p.stat().st_size==v['size'] and sha_file(p)==v['sha256'],'input checksum/size '+p.name)
    root=Path(info['dataset']['destination'])
    require(root.is_dir(),'VOC2012 missing')
    for v in info['dataset']['required_directories']:require((root/v).is_dir(),'dataset directory missing '+v)
    # Check referenced images/labels exist, without decoding images or tensors.
    for rel,n in [('splits/662/labeled.txt',662),('splits/662/unlabeled.txt',9920),('splits/val.txt',1449)]:
        lines=(Path(REG['target_root'])/'augseg_inputs'/rel).read_text().splitlines();require(len(lines)==n,'split count mismatch')
        for line in lines:
            for item in line.split():
                candidate=root/item.lstrip('/')
                require(candidate.is_file(),'referenced dataset file missing '+item)
    return {'inputs':'HASH_AND_EXISTENCE_VERIFIED','dataset_full_hash':'UNKNOWN'}
def validate_environment():
    versions=read(PKG/'runtime/inventory.json')['direct_dependencies_metadata_at_packaging']
    installed={k:importlib.metadata.version(k) for k in versions}
    require(installed==versions,'dedicated environment package versions differ from reviewed source inventory')
    require(sys.version_info[:2]==(3,11),'Python3.11 required')
    return installed

def launch_runtime_inventory():
    # Explicit launch only; no model, CUDA forward or tensors are constructed.
    import torch
    return {'packages':validate_environment(),'python':sys.version,
        'torch_compiled_cuda':torch.version.cuda,'loaded_cudnn_version':torch.backends.cudnn.version(),
        'tf32_matmul_before_trainer':torch.backends.cuda.matmul.allow_tf32,
        'tf32_cudnn_before_trainer':torch.backends.cudnn.allow_tf32,
        'float32_matmul_precision_before_trainer':torch.get_float32_matmul_precision(),
        'source_seed_helper_policy':'cuDNN deterministic=True benchmark=False; recorded before trainer applies seed policy',
        'historical_loaded_library_equivalence':'UNKNOWN; cross-GPU bitwise identity not claimed'}

def resources():
    raw=subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=name,memory.free','--format=csv,noheader,nounits'],text=True,timeout=10).strip();name,free=raw.rsplit(',',1)
    disk=shutil.disk_usage(REG['target_root']).free
    return {'gpu':name.strip(),'gpu_free_mib':int(free),'disk_free_bytes':disk,'ready':REG['gpu_model_contains'] in name and int(free)>=REG['gpu_min_free_mib'] and disk>=REG['disk_min_free_bytes'],'other_compute_processes':'allowed'}
def readonly_retry(call):
    for attempt in range(3):
        try:return call()
        except Exception as exc:
            kind=type(exc).__name__;transient=kind in {'AuthenticationError','ConnectTimeout','ReadTimeout','ConnectionError','ConnectError','ReadError','Timeout'}
            if not transient or attempt==2:raise
            print(json.dumps({'read_retry':attempt+1,'max_attempts':3,'error_type':kind}),flush=True);time.sleep(15)
def clients():
    require(os.environ.get('WANDB_API_KEY') and any(os.environ.get(k) for k in ('HF_TOKEN','HUGGINGFACE_TOKEN','HUGGING_FACE_HUB_TOKEN')),'existing credentials not injected')
    require(os.environ.get('WANDB_MODE')=='online' and not os.environ.get('WANDB_DISABLED'),'online only')
    import wandb
    from huggingface_hub import HfApi
    return readonly_retry(lambda:wandb.Api(timeout=30)),HfApi(token=next(os.environ[k] for k in ('HF_TOKEN','HUGGINGFACE_TOKEN','HUGGING_FACE_HUB_TOKEN') if os.environ.get(k)))
def remote_fresh(r):
    wb,hf=clients();project=readonly_retry(lambda:wb.project(REG['wandb_project'],entity=REG['wandb_entity']));require(project is not None,'W&B project unavailable')
    try:readonly_retry(lambda:wb.run(f"{REG['wandb_entity']}/{REG['wandb_project']}/{r['run_id']}"))
    except Exception as e:
        # Permission/auth/service errors are never treated as absence.
        msg=str(e).lower();require(type(e).__name__=='CommError' and ('could not find run' in msg or 'run not found' in msg),'W&B identity absence not verified')
    else:raise CohortError('W&B execution already exists')
    info=readonly_retry(lambda:hf.repo_info(REG['hf_repo'],repo_type='model'))
    paths=readonly_retry(lambda:hf.get_paths_info(REG['hf_repo'],[r['hf_path']],repo_type='model',revision=info.sha,expand=True));require(not paths,'HF path exists')
    return {'wandb':'ONLINE_READ_VERIFIED_ABSENT','hf_revision':info.sha,'hf_path_absent':True,'write':'UNTESTED'}
def predecessor(r):
    rows=profiles();i=next(i for i,v in enumerate(rows) if v['run_id']==r['run_id'])
    if i:verify_done(rows[i-1])
def preflight(r):
    validate_installed(r);fresh(r);require(Path(sys.executable).resolve()==Path(REG['python']).resolve(),'use dedicated target interpreter')
    validate_environment();inputs();res=resources();require(res['ready'],'GPU/disk guard blocked')
    predecessor(r);remote=remote_fresh(r)
    return {'resources':res,'remote':remote,'training_launched':False}
@contextlib.contextmanager
def lock():
    p=Path(REG['state_root']);p.mkdir(parents=True,exist_ok=True)
    with (p/'runner.lock').open('a+b') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);yield
def train_command(r):return [str(Path(REG['python']).parent/'torchrun'),'--standalone','--nproc_per_node=1','train_semi.py','--config',r['config'],'--seed','2']
def launch(r):
    preflight_result=preflight(r)
    with lock():
        validate_installed(r);fresh(r);require(resources()['ready'],'resource race; no execution created')
        out=Path(r['output']);out.mkdir(parents=True,exist_ok=False);Path(r['log']).parent.mkdir(parents=True,exist_ok=True)
        (out/'run_id.txt').write_text(r['run_id']+'\n')
        pin=source_pin(r);prov={'execution_id':r['run_id'],'control_execution_id':r['historical_control_execution_id'],'cohort_baseline_execution_id':r['cohort_baseline_execution_id'],'deployment_root':r['source_root'],'source_commit':pin['source_commit'],'source_commit_is_parent_only':True,'source_pin_sha256':sha_file(runtime_pin(r)),'source_hashes':pin['file_hashes'],'git_diff_sha256':ORIGINS['sources'][r['source']].get('historical_source_diff_sha256',''),'package_manifest_sha256':sha_file(PKG/'SHA256.json'),'package_commit':subprocess.check_output(['git','-C',str(PKG),'rev-parse','HEAD'],text=True).strip(),'protocol':{k:r[k] for k in ['batch_labeled','batch_unlabeled','world_size','seed','epochs','iterations_per_epoch','updates']},'python':sys.version,'packages':{p:importlib.metadata.version(p) for p in ['torch','torchvision','numpy','wandb','huggingface-hub']},'preflight':preflight_result,'runtime':launch_runtime_inventory(),'timestamp_utc':utc()}
        atomic(out/'provenance.json',prov,True);shutil.copy2(PKG/'SHA256.json',out/'package-SHA256.json')
        # Snapshot an explicit allowlist only: never copy credentials, logs, inputs or arbitrary local files.
        snapshot=out/'source_snapshot';snapshot.mkdir()
        for rel in pin['file_hashes']:
            dst=snapshot/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(Path(r['source_root'])/rel,dst)
        shutil.copy2(runtime_pin(r),snapshot/'migration-source-pin.json')
        env={**os.environ,'CUDA_VISIBLE_DEVICES':'0','PYTHONDONTWRITEBYTECODE':'1','PYTHONNOUSERSITE':'1','WANDB_MODE':'online','WANDB_RESUME':'never','GIT_OPTIONAL_LOCKS':'0','AUGSEG_DEPLOYMENT_PIN':str(runtime_pin(r)),'AUGSEG_DEPLOYMENT_PIN_SHA256':sha_file(runtime_pin(r))}
        for key in ('WANDB_RUN_ID','WANDB_DISABLED','RANK','WORLD_SIZE','LOCAL_RANK','MASTER_ADDR','MASTER_PORT'):env.pop(key,None)
        atomic(marker(r,'started'),{'utc':utc(),'run_id':r['run_id'],'command':train_command(r)},True)
        with Path(r['log']).open('xb') as log:rc=subprocess.run(train_command(r),cwd=r['source_root'],env=env,stdout=log,stderr=subprocess.STDOUT).returncode
        atomic(marker(r,'training_failed' if rc else 'training_done'),{'utc':utc(),'run_id':r['run_id'],'returncode':rc,'automatic_retry':False},True)
        require(rc==0,'training failed; never retry/resume automatically')
    finalize(r)
def completed(r,checkpoint_metadata=True):
    require(marker(r,'training_done').is_file() and not marker(r,'training_failed').exists(),'successful training required')
    out=Path(r['output']);require((out/'run_id.txt').read_text().strip()==r['run_id'],'run ID mismatch')
    require((out/'config.yaml').read_bytes()==Path(r['config']).read_bytes(),'saved config changed')
    prov=read(out/'provenance.json');require(prov['execution_id']==r['run_id'] and prov['source_pin_sha256']==sha_file(runtime_pin(r)),'runtime provenance mismatch')
    for rel,h in source_pin(r)['file_hashes'].items():require(sha_file(out/'source_snapshot'/rel)==h,'runtime snapshot drift')
    with (out/'epoch_metrics.csv').open() as f:rows=list(csv.DictReader(f))
    require([int(t['epoch']) for t in rows]==list(range(80)),'missing/duplicate epochs')
    for t in rows:
        require(t['run_id']==r['run_id'] and int(t['world_size'])==1,'epoch identity')
        e=int(t['epoch']);lr=.000125*(1-((e+1)*620-1)/49600)**.9;require(math.isclose(float(t['lr']),lr,rel_tol=1e-10),'fixed80 LR mismatch')
        require(all(math.isfinite(float(t[k])) and 0<=float(t[k])<=1 for k in ('miou_stu','miou_tea')),'metric scale')
    m=read(out/'manifest.json');require(m['epoch']==79 and m['run_id']==r['run_id'],'manifest mismatch')
    with (out/'iter_metrics.csv').open() as f:iters=list(csv.DictReader(f))
    for model,col in [('student','miou_stu'),('teacher','miou_tea')]:
        vals=[t for t in iters if t.get('meta/log_type')=='val_epoch' and t.get('val/model')==model]
        require([int(t['meta/global_iter']) for t in vals]==[(e+1)*620-1 for e in range(80)],'validation iteration coverage')
        require(all(math.isclose(float(t['val/miou']),float(z[col]),abs_tol=1e-7) for t,z in zip(vals,rows)),'raw validation mismatch')
    if checkpoint_metadata:
        for filename,epoch in [('ckpt.pth',80),('ckpt_best.pth',int(max(rows,key=lambda t:float(t['miou_tea']))['epoch'])+1)]:
            scalars=checkpoint_scalars(out/filename);require(scalars.get('epoch')==epoch and scalars.get('run_id')==r['run_id'],'checkpoint metadata mismatch')
    return rows
def checkpoint_scalars(path):
    """Read at most 4 MiB of pickle opcodes; never unpickle/load tensor storage."""
    with zipfile.ZipFile(path) as archive:
        entries = [i for i in archive.infolist() if i.filename.endswith("/data.pkl")]
        require(len(entries) == 1 and entries[0].file_size <= 4 * 1024**2, "checkpoint metadata size/schema unsupported")
        ops = list(pickletools.genops(archive.read(entries[0])))
    result = {}
    for index, (op, value, _) in enumerate(ops[:-1]):
        if value not in ("epoch", "run_id") or value in result:
            continue
        for nxt, arg, _ in ops[index + 1:index + 4]:
            if nxt.name in ("MEMOIZE", "BINPUT", "LONG_BINPUT"):
                continue
            if value == "epoch" and nxt.name in ("BININT", "BININT1", "BININT2", "INT"):
                result[value] = arg
            if value == "run_id" and nxt.name in ("BINUNICODE", "SHORT_BINUNICODE", "UNICODE"):
                result[value] = arg
            break
    return result

def validate_runtime_config(runtime, expected, row):
    """Resolve only defaults present in the bound trainer, then compare strictly.

    eps alone allows finite numeric string/float serialization equivalence.
    No unknown scientific key, routing switch, threshold or loss is discarded.
    """
    import ast
    import copy
    from decimal import Decimal, InvalidOperation
    groups = {"dataset", "trainer", "net", "criterion", "checkpoint",
              "saliency_cutmix", "csl", "csl_cutmix", "boundary_mix",
              "boundary_component", "boundary_compatibility",
              "u1_saliency_u2u", "u2_cross_view_saliency_u2u",
              "u3_confidence_filtered_cross_view_saliency_u2u",
              "u4_confidence_filtered_self_pseudo_saliency_u2u",
              "fixed_size_csl_destination", "confidence_reliability_control"}
    saved = Path(row["output"]) / "source_snapshot/train_semi.py"
    source = saved if saved.exists() else Path(row["source_root"]) / "train_semi.py"
    attestation = {"file_hashes": ORIGINS["sources"][row["source"]]["source_files"]}
    require(sha_file(source) == attestation["file_hashes"]["train_semi.py"],
            "runtime defaults source is not attested")
    tree = ast.parse(source.read_text())

    def path(node):
        if isinstance(node, ast.Name) and node.id == "cfg":
            return ()
        if isinstance(node, ast.Subscript):
            parent = path(node.value)
            key = ast.literal_eval(node.slice)
            if parent is not None and isinstance(key, str):
                return parent + (key,)
        return None

    constants_source = source.parent / "util/u1_saliency_u2u.py"
    require(sha_file(constants_source) == attestation["file_hashes"]["util/u1_saliency_u2u.py"], "unattested trainer constants")
    constants = {}
    aliases = {"NUM_CANDIDATES": "U1_NUM_CANDIDATES", "TEMPERATURE": "U1_TEMPERATURE", "U1_RNG_POLICY_VERSION": "U1_RNG_POLICY_VERSION"}
    for statement in ast.parse(constants_source.read_text()).body:
        if isinstance(statement, ast.Assign):
            for target in statement.targets:
                if isinstance(target, ast.Name) and target.id in aliases:
                    constants[aliases[target.id]] = ast.literal_eval(statement.value)

    def value(node, cfg):
        try:
            return ast.literal_eval(node)
        except (ValueError, TypeError):
            pass
        if isinstance(node, ast.Name) and node.id == "cfg":
            return cfg
        if isinstance(node, ast.Name) and node.id in constants:
            return constants[node.id]
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get" and not node.keywords:
            receiver = value(node.func.value, cfg)
            require(isinstance(receiver, dict) and 1 <= len(node.args) <= 2,
                    "unsupported trainer default expression")
            return receiver.get(*[value(a, cfg) for a in node.args])
        raise CohortError("unsupported trainer default expression")

    defaults = []
    for node in sorted(ast.walk(tree), key=lambda n: getattr(n, "lineno", -1)):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "setdefault" and len(node.args) == 2):
            continue
        target = path(node.func.value)
        if target is None:
            continue
        key = ast.literal_eval(node.args[0])
        if (target and target[0] in groups) or (not target and key in groups):
            defaults.append((target, key, node.args[1]))

    def resolve(cfg):
        cfg = copy.deepcopy(cfg)
        for target, key, expression in defaults:
            obj = cfg
            for part in target:
                require(isinstance(obj, dict) and part in obj, "missing trainer default parent")
                obj = obj[part]
            require(isinstance(obj, dict), "invalid scientific config mapping")
            if key not in obj:
                obj[key] = value(expression, cfg)
        for group in groups:
            mapping = cfg.get(group, {})
            if "eps" not in mapping:
                continue
            raw = mapping["eps"]
            require(not isinstance(raw, bool) and isinstance(raw, (str, float, int)), "invalid eps representation")
            try:
                eps = Decimal(str(raw))
            except InvalidOperation:
                raise CohortError("invalid eps representation")
            require(eps.is_finite(), "nonfinite eps")
            mapping["eps"] = eps
        return cfg

    actual, wanted = resolve(runtime), resolve(expected)
    for group in sorted(groups):
        require(actual.get(group) == wanted.get(group),
                "W&B scientific routing mismatch after bound defaults: " + group)
    return True

def verify_wandb(r,rows,wb):
    run=readonly_retry(lambda:wb.run(f"{REG['wandb_entity']}/{REG['wandb_project']}/{r['run_id']}"));cfg=expected_config(r)
    spec=importlib.util.spec_from_file_location('portable_tracking_read',PKG/'tracking_read_helpers.py');helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    helper.validate_wandb_completion(run,cfg,r['run_id'],rows,620)
    validate_runtime_config(run.config,cfg,r)
    require(run.config['run'].get('source_pin_sha256')==sha_file(runtime_pin(r)),'W&B source pin mismatch')
    return run

def verify_done(r):
    validate_installed(r);rows=completed(r);out=Path(r['output']);rec=read(marker(r,'done'))
    require(rec==read(marker(r,'upload_receipt.json'))==read(out/'upload_receipt.json'),'receipt disagreement')
    require(rec['run_id']==r['run_id'] and rec['repo_id']==REG['hf_repo'] and rec['path']==r['hf_path'] and rec['wandb_state']=='finished','receipt identity')
    bundle=out/'_hf_bundle/latest.tar.gz';require(bundle.stat().st_size==rec['size'] and sha_file(bundle)==rec['sha256'],'bundle checksum')
    wb,hf=clients();verify_wandb(r,rows,wb)
    for revision in dict.fromkeys([rec['revision'],readonly_retry(lambda:hf.repo_info(REG['hf_repo'],repo_type='model')).sha]):
        files=readonly_retry(lambda:hf.get_paths_info(REG['hf_repo'],[r['hf_path']],repo_type='model',revision=revision,expand=True));require(len(files)==1 and files[0].size==rec['size'] and files[0].lfs.sha256==rec['sha256'],'HF receipt content mismatch')
    return {'run_id':r['run_id'],'state':'VERIFIED_DONE','revision':rec['revision']}
def finalize(r):
    # There is deliberately no call to launch/train_command/subprocess here.
    validate_installed(r)
    with lock():
        try:
            rows=completed(r);out=Path(r['output']);wb,hf=clients();run=verify_wandb(r,rows,wb)
            same_or_new(out/'runtime_config.json',dict(run.config));same_or_new(out/'verified_metrics.json',{model:{'best':max(float(t['miou_'+model]) for t in rows),'best_epoch':int(max(rows,key=lambda t:float(t['miou_'+model]))['epoch']),'final79':float(rows[-1]['miou_'+model])} for model in ['stu','tea']})
            bundle=out/'_hf_bundle/latest.tar.gz';bundle.parent.mkdir(exist_ok=True)
            if not bundle.exists():
                tmp=bundle.with_suffix('.incomplete');require(not tmp.exists(),'incomplete bundle; inspect before recovery')
                with tarfile.open(tmp,'w:gz') as tar:
                    for name in ('ckpt.pth','ckpt_best.pth','config.yaml','run_id.txt','manifest.json','epoch_metrics.csv','iter_metrics.csv','provenance.json','runtime_config.json','verified_metrics.json','package-SHA256.json','source_snapshot'):tar.add(out/name,arcname=name)
                    tar.add(r['log'],arcname='launch.log')
                tmp.rename(bundle)
            digest=sha_file(bundle);size=bundle.stat().st_size;parent=readonly_retry(lambda:hf.repo_info(REG['hf_repo'],repo_type='model')).sha
            items=readonly_retry(lambda:hf.get_paths_info(REG['hf_repo'],[r['hf_path']],repo_type='model',revision=parent,expand=True));intent=marker(r,'upload_intent.json')
            if not items:
                require(not intent.exists(),'prior upload outcome uncertain; no repeat upload')
                atomic(intent,{'run_id':r['run_id'],'path':r['hf_path'],'parent_revision':parent,'sha256':digest,'size':size,'utc':utc()},True)
                revision=hf.upload_file(path_or_fileobj=str(bundle),path_in_repo=r['hf_path'],repo_id=REG['hf_repo'],repo_type='model',parent_commit=parent).oid
                items=readonly_retry(lambda:hf.get_paths_info(REG['hf_repo'],[r['hf_path']],repo_type='model',revision=revision,expand=True))
            else:
                require(intent.exists() and read(intent)['run_id']==r['run_id'] and read(intent)['sha256']==digest,'remote collision without matching upload intent');revision=items[0].last_commit.oid
            require(len(items)==1 and items[0].size==size and items[0].lfs.sha256==digest,'HF post-upload checksum mismatch')
            rec={'run_id':r['run_id'],'repo_id':REG['hf_repo'],'path':r['hf_path'],'revision':revision,'sha256':digest,'size':size,'wandb_state':run.state}
            same_or_new(out/'upload_receipt.json',rec);same_or_new(marker(r,'upload_receipt.json'),rec);same_or_new(marker(r,'done'),rec)
            return rec
        except Exception as exc:
            atomic(marker(r,'finalization_failed'),{'utc':utc(),'error_type':type(exc).__name__,'training_retry':False});raise

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['check-package','bootstrap','preflight','launch','finalize','verify-done']);p.add_argument('--method',choices=[r['method'] for r in profiles()]);a=p.parse_args()
    if a.action=='check-package':result=check_package()
    elif a.action=='bootstrap':result=bootstrap()
    else:
        require(a.method,'one explicit method required');r=row_for(a.method);result={'preflight':preflight,'launch':launch,'finalize':finalize,'verify-done':verify_done}[a.action](r)
    print(json.dumps(result,default=str,sort_keys=True))
if __name__=='__main__':
    try:main()
    except Exception as exc:
        # No credentials, environment dump or potentially secret HTTP messages.
        print('A6000_COHORT_HELD: '+type(exc).__name__+(': '+str(exc) if isinstance(exc,CohortError) else ''),file=sys.stderr);sys.exit(1)

#!/usr/bin/env python3
"""Isolated S1 two-stage K8 cohort. check/paths are read-only; never import trainer.

launch runs one fresh profile; finalize only verifies/packages/uploads its completed
artifacts. A training failure and a finalization failure have separate durable state.
No automatic retry of training, no scheduler resume, no queue/service integration.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import pickletools
import shutil
import subprocess
import sys
import tarfile
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
BASE = "bbe2224c50db2c277e5d76dbb64f6527489d5b5c"
COHORT = "s1_two_stage_k8_bs16_20261030"
MIN_GPU_FREE_MIB = 16384
PYTHON = Path("/home/jupyter-iec2024iot04/.conda/envs/augseg-bm/bin/python")
CONFIG_ROOT = ROOT / "exps/boundary_mix_v2_v3" / ("voc_semi662_" + COHORT)
ORIGIN_REPO = Path("/home/jupyter-iec2024iot04/AugSeg_BoundaryMix_V2V3_legacy_c321_rerun01")
OUTPUT_ROOT = ORIGIN_REPO / "exp_boundary_mix_v2_v3_legacy_c321_rtx5090"
STATE = Path("/home/jupyter-iec2024iot04/.local/state/augseg-legacy-c321-rerun01-rtx5090") / (COHORT + "_seed2")
REGISTRY = ROOT / "configs" / ("experiment_registry_voc662_" + COHORT + ".yaml")
PIN = ROOT / "configs" / (COHORT + "_reviewed-deployment.json")
ENTITY = "tanprodium-uit"
PROJECT = "AugsegLegacyC321Rerun01-RTX5090"
HF_REPO = "tanprodium/augseg-legacy-c321-rerun01-rtx5090"
MODES = {"s1_two_stage_saliency_k8": "saliency_softmax", "s1_two_stage_random_k8": "uniform_multinomial"}
CONTROL_ID = "baseline_augseg_fair80_r101_c321_bs16x1_gbs16_20260911_084310_2a37d013"
OLD_CONFIG = 'exps/boundary_mix_v2_v3/voc_semi662/baseline_augseg_fair80_c321_bs16x1_gbs16/config.yaml'
CONTROL_OUTPUT = Path('/home/jupyter-iec2024iot04/AugSeg_BoundaryMix_V2V3/exp_boundary_mix_v2_v3/baseline_augseg_fair80_r101_c321_bs16x1_gbs16')

PATCHED = {"train_semi.py", "util/saliency_cutmix.py", "util/run_logging.py"}
CANONICAL = ("config.yaml", "run_id.txt", "ckpt.pth", "ckpt_best.pth",
             "epoch_metrics.csv", "iter_metrics.csv", "manifest.json")


class CohortError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise CohortError(message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ORIGIN_REPO,
        env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"})


def digest(data):
    return hashlib.sha256(data).hexdigest()


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def profiles():
    rows = []
    for method, mode in MODES.items():
        name = f"{method}_{COHORT}_rtx5090_voc662_r101_c321_bs16x1_gbs16_seed2_attempt01"
        out = OUTPUT_ROOT / name
        rows.append({"method": method, "mode": mode, "identity": name,
            "run_id": "legacy5090_" + digest(name.encode())[:24], "control_execution_id": CONTROL_ID,
            "config": str(CONFIG_ROOT / method / "config.yaml"), "output": str(out),
            "log_dir": str(CONFIG_ROOT / method / "log"), "launch_log": str(STATE / (method + ".train.log")),
            "state": str(STATE), "lock": str(STATE / "runner.lock"),
            "launcher": str(ROOT / "scripts/run_s1_two_stage_k8.sh"),
            "controller_finalizer_uploader": str(Path(__file__).resolve()), "registry": str(REGISTRY), "reviewed_deployment": str(PIN),
            "artifacts": {f: str(out / f) for f in CANONICAL + ("provenance.json", "source.diff", "runtime_config.json", "verified_metrics.json", "upload_receipt.json", "_hf_bundle/latest.tar.gz")},
            "markers": {suffix: str(STATE / (method + "." + suffix)) for suffix in ("started", "training_done", "training_failed", "finalization_failed", "upload_intent.json", "upload_receipt.json", "done")},
            "hf_path": f"boundary_mix_v2_v3/voc_semi662/{COHORT}/rtx5090/{method}/seed2/attempt01/r101_c321_bs16x1_gbs16/latest.tar.gz",
            "iterations_per_epoch": 620, "updates": 49600, "final_iteration": 49599})
    return rows


def expected_config(row):
    cfg = yaml.safe_load(git("show", BASE + ":" + OLD_CONFIG))
    cfg["name"] = cfg["run"]["name"] = cfg["wandb"]["name"] = row["identity"]
    cfg["saver"]["snapshot_dir"] = row["output"]
    cfg["saver"].update(auto_resume=False, auto_profile_dir=False)
    cfg["checkpoint"]["auto_resume"] = False
    cfg["wandb"].update(enable=True, entity=ENTITY, project=PROJECT, resume="never")
    cfg["hf"].update(enabled=True, repo_id=HF_REPO, repo_type="model", auto_profile_path=False,
        auto_download=False, auto_upload=True, upload_every_epoch=False)
    s = yaml.safe_load(git("show", "c47581b:exps/boundary_mix_v2_v3/voc_semi662/s1_saliency_box_cutmix/config.yaml"))["saliency_cutmix"]
    s.update(ablation_contract="s1_two_stage_k8_fixed80_v1", strict_selector=True, fallback="abort",
        candidate_selection=row["mode"], direct_labeled_mix=False, direct_paste_policy="same_coordinate",
        direct_confidence_gate=False, eps=1e-6)
    cfg["saliency_cutmix"] = s
    cfg["wandb"]["mode"] = "online"
    cfg["wandb"]["group"] = COHORT
    cfg["wandb"]["tags"] = [COHORT, "two_stage", "candidate:" + row["mode"], "seed:2", "fresh80", "control:" + CONTROL_ID]
    cfg["run"].update(control_execution_id=CONTROL_ID, source_parent_commit=BASE,
        deployment_root=str(ROOT), paired_execution_id=next(r["run_id"] for r in profiles() if r["method"] != row["method"]),
        scientific_role="method" if row["mode"] == "saliency_softmax" else "additional_candidate_selection_control")
    cfg["hf"]["path_in_repo"] = row["hf_path"]
    cfg["hf"]["squash_after_upload"] = False
    return cfg


def patch_bytes():
    import difflib
    origins = json.loads((ROOT / "source_origins.json").read_text())
    parts = []
    for rel in sorted(origins["source_files"]):
        # Human-readable diff normalizes EOL; pins hash the exact file bytes.
        original = git("show", BASE + ":" + rel).decode().splitlines()
        current = (ROOT / rel).read_text().splitlines()
        parts.extend(difflib.unified_diff(original, current, fromfile="a/" + rel, tofile="b/" + rel, lineterm=""))
    return ("\n".join(parts) + "\n").encode()


def check_controls(pin):
    for path, expected in pin["protected_controls"].items():
        require(sha_file(path) == expected, "comparator evidence changed: " + path)
    cfg = yaml.safe_load(git("show", BASE + ":" + OLD_CONFIG))
    out = CONTROL_OUTPUT
    require((out / "run_id.txt").read_text().strip() == CONTROL_ID, "comparator ID mismatch")
    require(yaml.safe_load((out / "config.yaml").read_text()) == cfg, "comparator config differs from historical object")
    m = json.loads((out / "manifest.json").read_text())
    require(m["epoch"] == 79 and m["git_commit"] == BASE[:7] and m["world_size"] == 1, "comparator source/protocol mismatch")
    with (out / "epoch_metrics.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    require([int(r["epoch"]) for r in rows] == list(range(80)), "comparator not completed80")
    require(all(math.isclose(float(r["lr"]), .000125 * (1-((int(r["epoch"])+1)*620-1)/49600)**.9, rel_tol=1e-10) for r in rows), "comparator LR differs from fixed80")


    metadata = json.loads(Path(pin["control_metadata"]).read_text())
    argv = metadata["args"]
    require(metadata["git"]["commit"] == BASE and "--seed" in argv and argv[argv.index("--seed")+1] == "2",
            "comparator historical source/seed binding mismatch")
    launch_text = Path(pin["control_launch_log"]).read_text()
    require("Resumed checkpoint" not in launch_text and "Epoch/Iter [80:  0/" in launch_text,
            "comparator is not the recorded fresh80 launch")
    with (out / "iter_metrics.csv").open() as stream:
        observations = [r for r in csv.DictReader(stream) if r.get("meta/log_type") == "val_epoch" and r.get("val/model") == "teacher"]
    require([int(r["meta/global_iter"]) for r in observations] == [(e+1)*620-1 for e in range(80)],
            "comparator actual update coverage mismatch")


def validate():
    pin = json.loads(PIN.read_text())
    require(pin["source_commit"] == BASE and pin["deployment_root"] == str(ROOT) and not (ROOT / ".git").exists(), "wrong isolated deployment")
    origins = json.loads((ROOT / "source_origins.json").read_text())["source_files"]
    require({rel for rel, meta in origins.items() if sha_file(ROOT / rel) != meta["sha256"]} == PATCHED, "scientific snapshot patch scope changed")
    require(digest(patch_bytes()) == pin["git_diff_sha256"] and (ROOT / "source.diff").read_bytes() == patch_bytes(), "source diff changed")
    for rel, expected in pin["file_hashes"].items():
        path = ROOT / rel
        require(path.is_file() and not path.is_symlink() and sha_file(path) == expected, "reviewed snapshot hash mismatch: " + rel)
    protected = json.loads((ROOT / "protected_deployments.json").read_text())
    for path, expected in protected.items():
        require(sha_file(path) == expected, "protected R-entropy bytes changed: " + path)
    registry = yaml.safe_load(REGISTRY.read_text())
    require(registry["entries"] == profiles() and registry["cohort"] == COHORT and registry["status"] == "MANUAL_ONLY" and registry["active_queue"] == [] and registry["automatic_launch_enabled"] is False, "registry identities/activation changed")
    require(registry["input_hashes"] == pin["input_hashes"], "input pin mapping changed")
    for rel, expected in pin["input_hashes"].items():
        require(sha_file(ROOT / rel) == expected, "current split/pretrained bytes changed: " + rel)
    for row in profiles():
        cfg = yaml.safe_load(Path(row["config"]).read_text())
        require(cfg == expected_config(row), "unreviewed config difference: " + row["method"])
        require(Path(cfg["saver"]["snapshot_dir"]) == Path(row["output"]), "absolute output mismatch")
    check_controls(pin)
    return pin


def legacy_helpers():
    spec = importlib.util.spec_from_file_location("tracking_read_helpers", ROOT / "tools/tracking_read_helpers.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def local_fresh(row):
    for value in (row["output"], row["log_dir"], row["launch_log"]):
        require(not os.path.lexists(value), "non-fresh local path: " + value)
    for suffix in ("started", "training_done", "training_failed", "finalization_failed", "upload_intent.json", "upload_receipt.json", "done"):
        require(not os.path.lexists(STATE / (row["method"] + "." + suffix)), "existing production marker: " + suffix)


def remote_read(rows, fresh=True):
    # Match the existing controller's bounded read retries. Never retry uploads here.
    for attempt in range(3):
        try:
            return _remote_read(rows, fresh)
        except Exception as exc:
            transient = type(exc).__name__ in {"ConnectTimeout", "ReadTimeout", "ConnectionError", "ConnectError", "ReadError"}
            # Existing legacy controller treats only these verification/service
            # messages as retryable. No relogin and no credential/config change.
            transient |= type(exc).__name__ == "AuthenticationError" and "verifying the API key" in str(exc)
            transient |= type(exc).__name__ == "CommError" and "service process is busy" in str(exc)
            if not transient or attempt == 2:
                raise
            print(f"read-only metadata retry {attempt + 1}/3: {type(exc).__name__}", file=sys.stderr)
            time.sleep(15)


def _remote_read(rows, fresh=True):
    require(os.environ.get("WANDB_API_KEY"), "existing W&B authentication not injected")
    require(any(os.environ.get(k) for k in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN")), "existing HF authentication not injected")
    import wandb
    wb, hf = wandb.Api(timeout=30), hf_api()
    pin = json.loads(PIN.read_text())
    control = wb.run(pin["control_wandb_path"])
    expected = yaml.safe_load(git("show", BASE + ":" + OLD_CONFIG))
    require(control.state == "finished" and control.commit == BASE, "comparator W&B source/state mismatch")
    require(control.summary.get("_step") == 49599 and control.summary.get("meta/epoch") == 80, "comparator W&B horizon mismatch")
    for key in ("dataset", "net", "trainer", "criterion"):
        require(control.config.get(key) == expected[key], "comparator W&B recipe mismatch: " + key)
    files = hf.get_paths_info(pin["control_hf_repo"], [pin["control_hf_path"]], repo_type="model", expand=True)
    require(len(files) == 1 and files[0].lfs, "comparator HF artifact missing")
    revision = hf.repo_info(HF_REPO, repo_type="model").sha
    if fresh:
        for row in rows:
            legacy_helpers().verify_wandb_fresh(expected_config(row), row["run_id"], wb)
            require(not hf.get_paths_info(HF_REPO, [row["hf_path"]], repo_type="model", revision=revision), "HF path collision")
    return {"status": "READ_VERIFIED", "write_status": "WRITE_UNTESTED", "hf_revision": revision,
        "control": {"run_id": CONTROL_ID, "wandb_state": control.state, "hf_path": files[0].path, "hf_size": files[0].size, "hf_lfs_sha256": files[0].lfs.sha256},
        "new_ids_collision_checked": [r["run_id"] for r in rows] if fresh else []}


def hf_api():
    from huggingface_hub import HfApi
    return HfApi(token=next((os.environ[k] for k in
        ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN") if os.environ.get(k)), None))


def resources():
    values = subprocess.check_output(["/usr/bin/nvidia-smi", "--id=0",
        "--query-gpu=memory.free,utilization.gpu", "--format=csv,noheader,nounits"], text=True, timeout=10).strip()
    free, utilization = map(int, values.split(","))
    # Informational only: sharing GPU0 is allowed. Exact-execution freshness and
    # runner locks remain mandatory in launch(); never stop another user's job.
    pids = subprocess.check_output(["/usr/bin/nvidia-smi", "--id=0", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True, timeout=10).strip()
    queue = subprocess.run(["systemctl", "--user", "show", "augseg-legacy-c321-rerun01-continue-6-13.service", "--property=ActiveState", "--value"], capture_output=True, text=True, timeout=10)
    state = queue.stdout.strip() if queue.returncode == 0 else "UNKNOWN"
    return {"gpu_free_mib": free, "gpu_min_free_mib": MIN_GPU_FREE_MIB, "gpu_utilization": utilization, "compute_pids": pids.splitlines(),
            "disk_free_bytes": shutil.disk_usage(ROOT).free, "protected_queue_state": state,
            "launch_resource_ready": free >= MIN_GPU_FREE_MIB and shutil.disk_usage(ROOT).free >= 32 * 1024**3 and state in ("inactive", "failed")}



def write_once(path, content):
    path = Path(path)
    with path.open("xb") as stream:
        os.chmod(path, 0o600)
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def record(path, value):
    write_once(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def provenance(row, pin, remote):
    return {"execution_id": row["run_id"], "control_execution_id": CONTROL_ID,
        "source_commit": BASE, "source_commit_is_parent_only": True, "deployment_root": str(ROOT), "git_worktree": False,
        "source_pin_sha256": sha_file(PIN), "source_hashes": pin["file_hashes"], "git_diff_sha256": pin["git_diff_sha256"],
        "input_hashes": pin["input_hashes"], "command": training_command(row), "cwd": str(ROOT), "python": sys.version,
        "packages": {p: importlib.metadata.version(p) for p in ("torch", "torchvision", "numpy", "wandb", "huggingface-hub")},
        "protocol": {"seed": 2, "world_size": 1, "batches": [16, 16], "epochs": 80, "iterations_per_epoch": 620, "updates": 49600, "fresh": True, "resume": False},
        "scientific_contract": {"stages": 2, "candidate_selection": row["mode"], "candidate_pool": 8, "random_control_computes_same_saliency": True, "fallback": "abort_no_pipeline_fallback"},
        "metric_schema": {"epoch": "0..79", "miou": "fraction_0_1", "wb_epoch": "1..80", "wb_step": "global_iter; final=49599", "best": "independent maxima per model"},
        "remote_preflight": remote, "timestamp_utc": utc()}


def training_command(row):
    return [str(PYTHON.parent / "torchrun"), "--standalone", "--nproc_per_node=1",
            "train_semi.py", "--config", row["config"], "--seed", "2"]


def launch(row):
    pin = validate()
    local_fresh(row)
    remote = remote_read([row])
    require(sys.executable == str(PYTHON) and os.getcwd() == str(ROOT), "wrong interpreter/cwd")
    require(os.environ.get("WANDB_MODE") == "online" and not os.environ.get("WANDB_DISABLED"), "W&B online environment required")
    require(resources()["launch_resource_ready"], "GPU/disk not ready or protected R-entropy queue active/unknown; no state created")
    time.sleep(15)
    require(resources()["launch_resource_ready"], "resource confirmation failed; no state created")
    STATE.mkdir(mode=0o700, parents=True, exist_ok=True)
    with (STATE / "runner.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        local_fresh(row)
        validate()
        require(resources()["launch_resource_ready"], "resource gate changed")
        out = Path(row["output"])
        out.mkdir(mode=0o700)
        write_once(out / "run_id.txt", (row["run_id"] + "\n").encode())
        write_once(out / "source.diff", patch_bytes())
        record(out / "provenance.json", provenance(row, pin, remote))
        write_once(out / "reviewed-deployment.json", PIN.read_bytes())
        for rel in pin["file_hashes"]:
            target = out / "source_snapshot" / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            write_once(target, (ROOT / rel).read_bytes())
        record(STATE / (row["method"] + ".started"), {"utc": utc(), "run_id": row["run_id"], "command": training_command(row)})
        with Path(row["launch_log"]).open("xb") as stream:
            result = subprocess.run(training_command(row), cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                env={**os.environ, "AUGSEG_DEPLOYMENT_PIN": str(PIN), "AUGSEG_DEPLOYMENT_PIN_SHA256": sha_file(PIN)})
        if result.returncode:
            record(STATE / (row["method"] + ".training_failed"), {"utc": utc(), "returncode": result.returncode, "resume": False})
            raise RuntimeError("training failed; never automatically restart")
        record(STATE / (row["method"] + ".training_done"), {"utc": utc(), "run_id": row["run_id"]})
    # Completion/upload is a separate phase; finalize cannot invoke training.
    finalize(row)


def completed(row):
    out = Path(row["output"])
    require((STATE / (row["method"] + ".training_done")).is_file(), "successful training marker missing")
    require(not (STATE / (row["method"] + ".training_failed")).exists(), "training failed; finalization forbidden")
    for name in CANONICAL + ("provenance.json", "source.diff", "reviewed-deployment.json"):
        p = out / name
        require(p.is_file() and not p.is_symlink() and p.stat().st_size, "required artifact absent: " + name)
    require((out / "config.yaml").read_bytes() == Path(row["config"]).read_bytes(), "saved config mismatch")
    require((out / "run_id.txt").read_text().strip() == row["run_id"], "output execution ID mismatch")
    prov = json.loads((out / "provenance.json").read_text())
    require(prov["source_pin_sha256"] == training_pin_sha(row) and prov["execution_id"] == row["run_id"], "runtime source pin mismatch")
    require(prov["source_hashes"] == json.loads(training_pin_path(row).read_text())["file_hashes"], "incomplete runtime source snapshot mapping")
    for rel, sha in prov["source_hashes"].items():
        require(sha_file(out / "source_snapshot" / rel) == sha, "source snapshot drift")
    with (out / "epoch_metrics.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    require([int(r["epoch"]) for r in rows] == list(range(80)), "not exactly completed80")
    for r in rows:
        require(r["run_id"] == row["run_id"] and r["git_commit"] == BASE[:7] and int(r["world_size"]) == 1, "epoch identity mismatch")
        require(r.get("source_pin_sha256") == training_pin_sha(row) and r.get("s1_candidate_selection") == row["mode"] and r.get("s1_fallback_policy") == "abort", "epoch snapshot/selector provenance mismatch")
        expected_lr = .000125 * (1 - ((int(r["epoch"]) + 1) * 620 - 1) / 49600)**.9
        require(math.isclose(float(r["lr"]), expected_lr, rel_tol=1e-10), "fixed80 LR trajectory mismatch")
        require(all(math.isfinite(float(r[k])) and 0 <= float(r[k]) <= 1 for k in ("miou_stu", "miou_tea")), "metric scale/nonfinite")
    manifest = json.loads((out / "manifest.json").read_text())
    require(manifest["epoch"] == 79 and manifest["run_id"] == row["run_id"], "final manifest mismatch")
    for name, epoch in (("ckpt.pth", 80), ("ckpt_best.pth", int(max(rows, key=lambda r: float(r["miou_tea"]))["epoch"]) + 1)):
        scalars = checkpoint_scalars(out / name)
        require(scalars.get("run_id") == row["run_id"] and scalars.get("epoch") == epoch,
                "checkpoint metadata identity/epoch mismatch: " + name)
    with (out / "iter_metrics.csv").open() as stream:
        iterations = list(csv.DictReader(stream))
    for model, col in (("student", "miou_stu"), ("teacher", "miou_tea")):
        history = [r for r in iterations if r.get("meta/log_type") == "val_epoch" and r.get("val/model") == model]
        require([int(r["meta/global_iter"]) for r in history] == [(e + 1) * 620 - 1 for e in range(80)], "validation coverage mismatch")
        require(all(math.isclose(float(h["val/miou"]), float(r[col]), abs_tol=1e-7) for h, r in zip(history, rows)), "raw CSV metric mismatch")
    train_rows = [r for r in iterations if r.get("meta/log_type") == "train_iter"]
    require(bool(train_rows), "selector telemetry missing")
    for r in train_rows:
        require(float(r.get("saliency/num_candidates", "nan")) == 8 and float(r.get("saliency/temperature", "nan")) == .2 and float(r.get("saliency/fallback_ratio", "nan")) == 0, "selector contract/fallback telemetry mismatch")
        require(float(r.get("saliency/candidate_policy_id", "nan")) == float(row["mode"] == "uniform_multinomial"), "selector policy telemetry mismatch")
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



def training_pin_path(row):
    """Use the attestation actually saved at launch; never relabel historical runs."""
    saved = Path(row["output"]) / "reviewed-deployment.json"
    if not saved.exists():
        return PIN
    live = json.loads(PIN.read_text())
    allowed = {sha_file(PIN)}
    previous = live.get("historical_training_pins", {}).get(row["run_id"])
    if previous:
        allowed.add(previous)
    require(sha_file(saved) in allowed, "unreviewed historical training pin")
    return saved


def training_pin_sha(row):
    return sha_file(training_pin_path(row))


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
    source = saved if saved.exists() else ROOT / "train_semi.py"
    attestation = json.loads(training_pin_path(row).read_text())
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


def _finalize(row):
    validate()
    rows = completed(row)
    import wandb
    run = wandb.Api(timeout=30).run(f"{ENTITY}/{PROJECT}/{row['run_id']}")
    cfg = expected_config(row)
    legacy_helpers().validate_wandb_completion(run, cfg, row["run_id"], rows, 620)
    validate_runtime_config(run.config, cfg, row)
    require(run.config["run"].get("source_pin_sha256") == training_pin_sha(row), "W&B source pin mismatch")
    out = Path(row["output"])
    write_if_absent_or_same(out / "runtime_config.json", (json.dumps(run.config, indent=2, sort_keys=True) + "\n").encode())
    metrics = {}
    for model in ("stu", "tea"):
        best = max(rows, key=lambda r: float(r["miou_" + model]))
        metrics[model] = {"best": float(best["miou_" + model]), "best_epoch": int(best["epoch"]), "final79": float(rows[-1]["miou_" + model])}
    write_if_absent_or_same(out / "verified_metrics.json", (json.dumps(metrics, indent=2, sort_keys=True) + "\n").encode())
    operation = {"training_pin_sha256": training_pin_sha(row), "finalizer_pin_sha256": sha_file(PIN),
                 "controller_sha256": sha_file(Path(__file__)),
                 "recovery_review": json.loads(PIN.read_text())["finalization_recovery_review"]}
    write_if_absent_or_same(out / "finalization_operations.json", (json.dumps(operation, indent=2, sort_keys=True) + "\n").encode())
    bundle = out / "_hf_bundle/latest.tar.gz"
    if not bundle.exists():
        bundle.parent.mkdir(exist_ok=True)
        tmp = bundle.with_suffix(".incomplete")
        require(not tmp.exists(), "incomplete bundle requires inspection; no overwrite")
        with tarfile.open(tmp, "w:gz") as tar:
            for name in CANONICAL + ("provenance.json", "source.diff", "reviewed-deployment.json", "runtime_config.json", "verified_metrics.json", "finalization_operations.json", "source_snapshot"):
                tar.add(out / name, arcname=name)
            tar.add(row["launch_log"], arcname="launch.log")
        tmp.rename(bundle)
    size, sha = bundle.stat().st_size, sha_file(bundle)
    hf = hf_api()
    parent = hf.repo_info(HF_REPO, repo_type="model").sha
    items = hf.get_paths_info(HF_REPO, [row["hf_path"]], repo_type="model", revision=parent, expand=True)
    intent = STATE / (row["method"] + ".upload_intent.json")
    if not items:
        require(not intent.exists(), "previous upload outcome uncertain; inspect remote before another write")
        record(intent, {"run_id": row["run_id"], "path": row["hf_path"], "parent_revision": parent, "sha256": sha, "size": size, "utc": utc()})
        commit = hf.upload_file(path_or_fileobj=str(bundle), path_in_repo=row["hf_path"],
            repo_id=HF_REPO, repo_type="model", parent_commit=parent)
        revision = commit.oid
        items = hf.get_paths_info(HF_REPO, [row["hf_path"]], repo_type="model", revision=revision, expand=True)
    else:
        require(intent.exists(), "remote collision without local upload intent")
        saved_intent = json.loads(intent.read_text())
        require(saved_intent["sha256"] == sha and saved_intent["run_id"] == row["run_id"], "upload intent mismatch")
        revision = items[0].last_commit.oid
    require(len(items) == 1 and items[0].size == size and items[0].lfs.sha256 == sha, "HF content-addressed verification failed")
    receipt = {"run_id": row["run_id"], "repo_id": HF_REPO, "path": row["hf_path"],
               "revision": revision, "sha256": sha, "size": size, "wandb_state": run.state}
    payload = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode()
    write_if_absent_or_same(out / "upload_receipt.json", payload)
    write_if_absent_or_same(STATE / (row["method"] + ".upload_receipt.json"), payload)
    write_if_absent_or_same(STATE / (row["method"] + ".done"), payload)
    return receipt


def write_if_absent_or_same(path, data):
    path = Path(path)
    if path.exists():
        require(path.read_bytes() == data, "immutable artifact collision: " + str(path))
    else:
        write_once(path, data)


def finalize(row):
    require(STATE.is_dir(), "production state absent; finalize never creates a run")
    with (STATE / "runner.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            return _finalize(row)
        except Exception as exc:
            marker = STATE / (row["method"] + ".finalization_failed")
            if not marker.exists():
                record(marker, {"utc": utc(), "error_type": type(exc).__name__, "training_restart": False})
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "paths", "launch", "finalize"))
    parser.add_argument("--method", choices=tuple(MODES))
    parser.add_argument("--remote", action="store_true", help="read exact scoped metadata only")
    parser.add_argument("--resources", action="store_true", help="read GPU/disk telemetry only")
    args = parser.parse_args()
    rows = [r for r in profiles() if not args.method or r["method"] == args.method]
    if args.action in ("launch", "finalize"):
        require(args.method, "one explicit method required; no automatic cohort queue")
        result = launch(rows[0]) if args.action == "launch" else finalize(rows[0])
    elif args.action == "paths":
        result = {"registry": str(REGISTRY), "reviewed_deployment": str(PIN), "profiles": rows}
    else:
        validate()
        for row in rows:
            local_fresh(row)
        result = {"config": "CONFIG_ENABLED", "pins": "MATCH", "controls": "COMPARATOR_AND_R_ENTROPY_PROTECTED",
                  "production_state_created": False, "write_status": "WRITE_UNTESTED", "profiles": rows}
        if args.remote:
            result["remote"] = remote_read(rows)
        if args.resources:
            result["resources"] = resources()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        # Avoid HTTP exception bodies, signed URLs, auth headers or environment.
        if isinstance(exc, CohortError):
            print("COHORT_BLOCKED: " + str(exc), file=sys.stderr)
        else:
            print("COHORT_BLOCKED: " + type(exc).__name__, file=sys.stderr)
        sys.exit(2)

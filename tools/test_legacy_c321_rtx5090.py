import csv
import hashlib
import importlib.util
import io
import json
import os
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import yaml

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("legacy_finalizer", ROOT / "tools/finalize_legacy_hf_upload.py")
fin = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fin)

class LegacyCohortTests(unittest.TestCase):
    def test_order_count_contract_semantics_and_uniqueness(self):
        self.assertEqual(len(fin.METHODS), 13)
        self.assertEqual(len(set(fin.METHODS)), 13)
        runner = (ROOT / "run_legacy_c321_rtx5090.sh").read_text()
        self.assertEqual(subprocess.run(["bash", "-n", str(ROOT / "run_legacy_c321_rtx5090.sh")]).returncode, 0)
        dry = subprocess.check_output(["bash", str(ROOT / "run_legacy_c321_rtx5090.sh"), "--dry-run"], text=True).splitlines()
        self.assertEqual(dry, [str(fin.BASE / m / "config.yaml") for m in fin.METHODS])
        failure = subprocess.run(["bash", str(ROOT / "run_legacy_c321_rtx5090.sh"), "--self-test-failure"], capture_output=True, text=True)
        self.assertEqual(failure.returncode, 23)
        self.assertEqual(failure.stdout.splitlines(), ["attempted: baseline_augseg_fair80_rerun01"])
        self.assertNotIn("retry", runner.lower())
        self.assertNotIn(".method_queue", runner)
        self.assertNotIn("artifacts/slide_assets", runner)
        identities, outputs, hf_paths = set(), set(), set()
        allow = {
            "name", "run.name", "saver.auto_resume", "saver.auto_profile_dir",
            "saver.snapshot_dir", "wandb.enable", "wandb.entity",
            "wandb.project", "wandb.name", "wandb.resume",
            "checkpoint.auto_resume", "hf.enabled", "hf.auto_profile_path",
            "hf.repo_id", "hf.repo_type", "hf.auto_download", "hf.auto_upload",
            "hf.upload_every_epoch", "hf.keep_only_latest", "hf.bundle_name",
            "hf.path_in_repo", "hf.squash_after_upload",
        }
        def differences(a, b, prefix=""):
            if isinstance(a, dict) and isinstance(b, dict):
                result = set()
                for key in set(a) | set(b):
                    result |= differences(a.get(key, object()), b.get(key, object()), f"{prefix}.{key}".lstrip("."))
                return result
            return {prefix} if a != b else set()
        for m in fin.METHODS:
            source_path = Path("exps/boundary_mix_v2_v3/voc_semi662") / m / "config.yaml"
            raw = subprocess.check_output(["git", "show", f"{fin.PARENT}:{source_path}"], cwd=ROOT, text=True)
            source = yaml.safe_load(raw)
            target_path = ROOT / fin.BASE / m / "config.yaml"
            cfg = yaml.safe_load(target_path.read_text())
            self.assertTrue(target_path.is_file())
            changed = differences(source, cfg)
            self.assertTrue(changed <= allow, (m, changed - allow))
            self.assertEqual(cfg["dataset"]["n_sup"], 662)
            self.assertEqual(cfg["dataset"]["train"]["crop"]["size"], [321, 321])
            self.assertEqual(cfg["dataset"]["train"]["batch_size"], 8)
            self.assertEqual(cfg["dataset"]["workers"], 4)
            self.assertEqual(cfg["trainer"]["epochs"], 80)
            self.assertEqual(cfg["trainer"]["optimizer"]["type"], "SGD")
            self.assertEqual(cfg["trainer"]["optimizer"]["kwargs"], {"lr": 0.000125, "momentum": 0.9, "weight_decay": 0.0001})
            self.assertEqual(cfg["trainer"]["lr_scheduler"]["mode"], "poly")
            self.assertEqual(cfg["trainer"]["lr_scheduler"]["kwargs"]["power"], 0.9)
            self.assertEqual(cfg["net"]["ema_decay"], 0.999)
            self.assertIn("resnet101", cfg["net"]["encoder"]["type"])
            self.assertEqual(cfg["trainer"]["unsupervised"]["threshold"], 0.95)
            self.assertEqual(cfg["trainer"]["unsupervised"]["loss_weight"], 1.0)
            for block in ("boundary_mix", "boundary_component", "boundary_compatibility", "saliency_cutmix", "csl", "csl_cutmix", "fixed_size_csl_destination"):
                self.assertEqual(cfg.get(block), source.get(block))
            self.assertFalse(cfg.get("boundary_mix", {}).get("enabled", False))
            self.assertFalse(cfg.get("boundary_component", {}).get("enabled", False))
            self.assertFalse(cfg["checkpoint"]["auto_resume"])
            self.assertFalse(cfg["saver"]["auto_resume"])
            self.assertFalse(cfg["hf"]["auto_download"])
            self.assertFalse(cfg["hf"]["upload_every_epoch"])
            self.assertTrue(cfg["hf"]["enabled"] and cfg["hf"]["auto_upload"])
            self.assertEqual(cfg["wandb"]["project"], fin.PROJECT)
            self.assertEqual(cfg["wandb"]["entity"], fin.ENTITY)
            self.assertEqual(cfg["wandb"]["resume"], "never")
            self.assertEqual(cfg["name"], cfg["run"]["name"])
            self.assertEqual(cfg["name"], cfg["wandb"]["name"])
            self.assertNotIn("suite_id", cfg["run"])
            self.assertEqual(cfg["hf"]["path_in_repo"], __import__("util.hf_auto", fromlist=["_default_path_in_repo"])._default_path_in_repo(cfg["hf"], Path("/unused")))
            identities.add(cfg["wandb"]["name"])
            outputs.add(cfg["saver"]["snapshot_dir"])
            hf_paths.add(cfg["hf"]["path_in_repo"])
            self.assertNotEqual(cfg["run"]["name"], source["run"]["name"])
            self.assertNotEqual(cfg["saver"]["snapshot_dir"], source["saver"]["snapshot_dir"])
            self.assertNotEqual(cfg["hf"]["path_in_repo"], source["hf"]["path_in_repo"])
        self.assertEqual((len(identities), len(outputs), len(hf_paths)), (13, 13, 13))
        self.assertFalse(any("v23_" in m or "v2_" in m or "corrected" in m or m.startswith(("s1_", "c1_", "c2_", "c4_", "u")) for m in fin.METHODS))

    def fixture(self, temp):
        save = Path(temp)
        config = save / "source.yaml"
        config.write_bytes(b"config")
        for name in fin.REQUIRED:
            if name == "config.yaml":
                (save / name).write_bytes(config.read_bytes())
            else:
                (save / name).write_bytes(name.encode())
        with (save / "epoch_metrics.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["epoch"])
            writer.writeheader()
            for epoch in range(80):
                writer.writerow({"epoch": epoch})
        (save / "run_id.txt").write_text("legacy5090_" + "a" * 24)
        (save / "manifest.json").write_text(json.dumps({"epoch": 79, "run_id": "legacy5090_" + "a" * 24, "run_name": "name", "world_size": 1, "git_commit": "abcdef0", "save_path": str(save)}))
        cfg = {"name": "name", "run": {"name": "name"}, "wandb": {"name": "name", "project": fin.PROJECT, "entity": fin.ENTITY},
               "hf": {"repo_id": fin.REPO, "repo_type": "model", "path_in_repo": "x/latest.tar.gz", "bundle_name": "latest.tar.gz"}}
        return save, config, cfg

    def test_local_validation_and_upload_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            save, path, cfg = self.fixture(temp)
            ckpt = {"epoch": 80, "run_id": "legacy5090_" + "a" * 24, "cfg": cfg, "args": {"seed": 2, "config": str(path)}}
            def fake_load(file, **kwargs):
                return dict(ckpt, epoch=79 if str(file).endswith("ckpt_best.pth") else 80)
            with mock.patch.object(fin, "reviewed_config", return_value=("method", path, cfg, save, "abcdef012345")), mock.patch.dict("sys.modules", {"torch": SimpleNamespace(load=fake_load)}):
                self.assertEqual(fin.validate_local(path)[-1], "legacy5090_" + "a" * 24)
                with (save / "epoch_metrics.csv").open("a") as stream:
                    stream.write("79\n")
                with self.assertRaises(ValueError):
                    fin.validate_local(path)
                self.fixture(temp)
                with mock.patch("util.hf_auto.maybe_upload_hf_bundle", return_value={"uploaded": False}):
                    with self.assertRaises(RuntimeError):
                        fin.finalize(path)

    def test_remote_oid_size_and_member_hashes(self):
        with tempfile.TemporaryDirectory() as temp:
            save, path, cfg = self.fixture(temp)
            bundle = save / "_hf_bundle" / "latest.tar.gz"
            bundle.parent.mkdir()
            with tarfile.open(bundle, "w:gz") as tar:
                for name in fin.REQUIRED:
                    source = path if name == "config.yaml" else save / name
                    tar.add(source, arcname=name)
            oid = hashlib.sha256(bundle.read_bytes()).hexdigest()
            api = SimpleNamespace(get_paths_info=lambda **kwargs: [SimpleNamespace(path="x/latest.tar.gz", size=bundle.stat().st_size, lfs={"sha256": oid})])
            downloader = lambda **kwargs: str(bundle)
            self.assertEqual(fin.verify_remote(cfg, save, path, api, downloader)["lfs_oid"], oid)
            bad = SimpleNamespace(get_paths_info=lambda **kwargs: [SimpleNamespace(path="x/latest.tar.gz", size=bundle.stat().st_size, lfs={"sha256": "bad"})])
            with self.assertRaises(ValueError):
                fin.verify_remote(cfg, save, path, bad, downloader)
            (save / "manifest.json").write_text("changed")
            with self.assertRaises(ValueError):
                fin.verify_members(bundle, save, path)

if __name__ == "__main__":
    unittest.main()

"""Actual loader parsing and preflight/upgrade tests; fixtures are temp-only."""
import copy
import contextlib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock

from PIL import Image

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))
import controller as c


class Inputs(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def fixtures(self):
        info = copy.deepcopy(c.read(PKG / "inputs.json"))
        dataset = self.root / "augseg_inputs/VOC2012"
        specs = []
        for i, (rel, archive_path, _) in enumerate(c.SPLIT_SPECS):
            ids = ["2007_00000" + str(i)]
            p = self.root / "augseg_inputs" / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("\n".join(ids) + "\n")
            row = info["files"][i]
            row.update(destination=str(p), size=p.stat().st_size,
                       sha256=c.sha_file(p), archive_path=archive_path)
            specs.append((rel, archive_path, 1))
            for name, extension, mode, color in [("JPEGImages", ".jpg", "RGB", (2, 3, 4)), ("SegmentationClassAug", ".png", "L", 255)]:
                q = dataset / name / (ids[0] + extension)
                q.parent.mkdir(parents=True, exist_ok=True)
                Image.new(mode, (2, 2), color).save(q)
        pretrained = self.root / "augseg_inputs/pretrained/resnet101.pth"
        pretrained.parent.mkdir(parents=True)
        pretrained.write_bytes(b"UNIT_FIXTURE_NOT_A_CHECKPOINT")
        info["files"][3].update(destination=str(pretrained), size=pretrained.stat().st_size,
                                sha256=c.sha_file(pretrained))
        info["dataset"]["destination"] = str(dataset)
        return info, specs, dataset

    def run_inputs(self, info, specs):
        with mock.patch.object(c, "REG", dict(c.REG, target_root=str(self.root))), mock.patch.object(c, "read", return_value=info), mock.patch.object(c, "SPLIT_SPECS", specs):
            return c.inputs()

    def test_actual_loader_parser_accepts_train_unlabeled_validation_for_all_sources(self):
        info, _, _ = self.fixtures()
        for kind in ("baseline", "s1", "s2"):
            path = PKG / "sources" / kind / "augseg/dataset/base.py"
            spec = importlib.util.spec_from_file_location("actual_base_" + kind, path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            # Retain and close the unchanged historical parser's file handles.
            with contextlib.ExitStack() as handles:
                def tracked_open(*args, **kwargs):
                    return handles.enter_context(open(*args, **kwargs))
                with mock.patch.object(module, "open", tracked_open, create=True):
                    for row in info["files"][:3]:
                        dset = module.BaseDataset(row["destination"])
                        image_id = Path(row["destination"]).read_text().strip()
                        self.assertEqual(dset.list_sample, [["JPEGImages/" + image_id + ".jpg", "SegmentationClassAug/" + image_id + ".png"]])

    def test_preflight_accepts_id_splits_and_counts_both_files(self):
        info, specs, _ = self.fixtures()
        result = self.run_inputs(info, specs)
        self.assertEqual((result["unique_images"], result["unique_masks"]), (3, 3))

    def test_missing_image_fails_for_each_split(self):
        info, specs, dataset = self.fixtures()
        for i in range(3):
            p = dataset / "JPEGImages" / ("2007_00000" + str(i) + ".jpg")
            data = p.read_bytes(); p.unlink()
            with self.assertRaisesRegex(c.CohortError, "JPEGImages"):
                self.run_inputs(info, specs)
            p.write_bytes(data)

    def test_missing_mask_fails_including_unlabeled(self):
        info, specs, dataset = self.fixtures()
        for i in range(3):
            p = dataset / "SegmentationClassAug" / ("2007_00000" + str(i) + ".png")
            data = p.read_bytes(); p.unlink()
            with self.assertRaisesRegex(c.CohortError, "SegmentationClassAug"):
                self.run_inputs(info, specs)
            p.write_bytes(data)

    def test_checksum_and_count_guards_remain(self):
        info, specs, _ = self.fixtures()
        broken = copy.deepcopy(info); broken["files"][0]["sha256"] = "0" * 64
        with self.assertRaisesRegex(c.CohortError, "checksum"):
            self.run_inputs(broken, specs)
        wrong_count = list(specs); wrong_count[0] = (*specs[0][:2], 2)
        with self.assertRaisesRegex(c.CohortError, "count"):
            self.run_inputs(info, wrong_count)

    def test_invalid_id_and_manifest_mapping_fail(self):
        info, specs, _ = self.fixtures()
        p = Path(info["files"][0]["destination"]); p.write_text("../escape\n")
        info["files"][0].update(size=p.stat().st_size, sha256=c.sha_file(p))
        with self.assertRaisesRegex(c.CohortError, "invalid VOC split ID"):
            self.run_inputs(info, specs)
        info["files"][0]["archive_path"] = "wrong"
        with self.assertRaisesRegex(c.CohortError, "destination/manifest"):
            self.run_inputs(info, specs)

    def test_pinned_input_bytes_and_protocol_unchanged(self):
        info = c.read(PKG / "inputs.json")
        self.assertEqual([(v["size"], v["sha256"]) for v in info["files"]], [
            (7944, "7dada8ccc5eb6309b4b81e172dfc24f0d83ea72078675083778ca10487182748"),
            (119040, "49b06340ca1534c3f15001ea9ce6d2815dd2ae1302390cfbdcfc0b004e17d960"),
            (17388, "cdc1326d12f69ce5153aa5da04a4d8783e146868d42d19f82f44e74b97ac907d"),
            (179202351, "d25b500a332e97a7bcbc11dc1153ec07c2f5c8334fbac2b36d882fe78a01af98")])
        for row in c.profiles():
            cfg = c.expected_config(row)
            self.assertEqual((row["batch_labeled"], row["batch_unlabeled"], row["iterations_per_epoch"], row["updates"]), (16, 16, 620, 49600))
            self.assertIn("/splits/pascal/", cfg["dataset"]["train"]["data_list"])
            self.assertIn("/splits/pascal/", cfg["dataset"]["val"]["data_list"])

    def upgrade_fixture(self):
        reg = copy.deepcopy(c.REG)
        reg["state_root"] = str(self.root / "state")
        for row in reg["profiles"]:
            row["source_root"] = str(self.root / "deployments" / row["source"])
            row["config"] = str(Path(row["source_root"]) / "exps" / row["method"] / "config.yaml")
            row["output"] = str(self.root / "outputs" / row["run_id"])
            row["log"] = str(self.root / "state" / (row["run_id"] + ".log"))
            row["state_prefix"] = str(self.root / "state" / row["run_id"])
            source = Path(row["source_root"])
            for rel in c.source_pin(row)["file_hashes"]:
                p = source / rel; p.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(PKG / "sources" / row["source"] / rel, p)
            c.runtime_pin(row).write_text(json.dumps(c.source_pin(row)))
            p = Path(row["config"]); p.parent.mkdir(parents=True, exist_ok=True)
            old = (PKG / row["package_config"]).read_bytes().replace(b"/splits/pascal/", b"/splits/")
            p.write_bytes(old)
        return reg

    def upgrade(self, reg, remote=None):
        with mock.patch.object(c, "REG", reg), mock.patch.object(c, "check_package"), mock.patch.object(c, "target_host"), mock.patch.object(c, "remote_fresh", remote or mock.Mock(return_value={})), mock.patch.object(c.subprocess, "run", side_effect=AssertionError("No training")):
            return c.upgrade_input_paths()

    def test_bootstrapped_upgrade_preserves_pins_ids_and_is_idempotent(self):
        reg = self.upgrade_fixture(); before = {r["source_root"]: c.runtime_pin(r).read_bytes() for r in reg["profiles"]}
        remote = mock.Mock(return_value={})
        result = self.upgrade(reg, remote)
        self.assertEqual(result["configs_changed"], 4); self.assertEqual(remote.call_count, 4)
        for row in reg["profiles"]:
            self.assertEqual(Path(row["config"]).read_bytes(), (PKG / row["package_config"]).read_bytes())
            self.assertEqual(c.runtime_pin(row).read_bytes(), before[row["source_root"]])
        self.assertEqual(self.upgrade(reg)["configs_changed"], 0)

    def test_started_collision_blocks_all_config_updates(self):
        reg = self.upgrade_fixture(); row = reg["profiles"][1]
        p = c.marker(row, "started"); p.parent.mkdir(parents=True); p.write_text("{}")
        before = [Path(r["config"]).read_bytes() for r in reg["profiles"]]
        with self.assertRaisesRegex(c.CohortError, "existing execution"):
            self.upgrade(reg)
        self.assertEqual(before, [Path(r["config"]).read_bytes() for r in reg["profiles"]])

    def test_remote_collision_blocks_all_config_updates(self):
        reg = self.upgrade_fixture(); remote = mock.Mock(side_effect=c.CohortError("remote collision"))
        before = [Path(r["config"]).read_bytes() for r in reg["profiles"]]
        with self.assertRaisesRegex(c.CohortError, "remote collision"):
            self.upgrade(reg, remote)
        self.assertEqual(before, [Path(r["config"]).read_bytes() for r in reg["profiles"]])

    def test_unknown_installed_config_and_source_pin_fail(self):
        reg = self.upgrade_fixture(); row = reg["profiles"][0]; p = Path(row["config"])
        original = p.read_bytes(); p.write_bytes(original + b"# unreviewed\n")
        with self.assertRaisesRegex(c.CohortError, "unreviewed installed config"):
            self.upgrade(reg)
        p.write_bytes(original); c.runtime_pin(row).write_text("{}")
        with self.assertRaisesRegex(c.CohortError, "runtime source pin changed"):
            self.upgrade(reg)


if __name__ == "__main__":
    unittest.main()

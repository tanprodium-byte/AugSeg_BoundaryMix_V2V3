#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scheduler.run_train_job import resolve_config_path


def test_absolute_local_path_exists(tmp_root: Path) -> None:
    config = tmp_root / "exps" / "boundary_mix_v2_v3" / "voc_semi662" / "v2_component_weighting" / "config.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("trainer:\n  epochs: 1\n")

    assert resolve_config_path(config, tmp_root) == config


def test_foreign_absolute_exps_path_localizes(tmp_root: Path) -> None:
    local = tmp_root / "exps" / "boundary_mix_v2_v3" / "voc_semi662" / "v3_js_bcr_d1" / "config.yaml"
    local.parent.mkdir(parents=True)
    local.write_text("trainer:\n  epochs: 1\n")
    foreign = "/home/other/AugSeg_BoundaryMix_V2V3/exps/boundary_mix_v2_v3/voc_semi662/v3_js_bcr_d1/config.yaml"

    assert resolve_config_path(foreign, tmp_root) == local


def test_unresolvable_path_has_clear_error(tmp_root: Path) -> None:
    original = "/home/other/AugSeg_BoundaryMix_V2V3/exps/missing/config.yaml"
    try:
        resolve_config_path(original, tmp_root)
    except FileNotFoundError as exc:
        message = str(exc)
        assert "original_config_path=" in message, message
        assert "resolved_config_path=" in message, message
        assert "repo_root=" in message, message
        assert original in message, message
    else:
        raise AssertionError("expected FileNotFoundError")


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)
        for test_fn in (
            test_absolute_local_path_exists,
            test_foreign_absolute_exps_path_localizes,
            test_unresolvable_path_has_clear_error,
        ):
            test_fn(tmp_root)
    print("PASS run_train_job config path resolver tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

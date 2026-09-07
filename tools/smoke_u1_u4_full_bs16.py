#!/usr/bin/env python3
"""Isolated one-step CUDA smoke for the accepted U1-U4 full BS16 path.

Each invocation uses one full labeled batch and one full unlabeled batch of 16,
then validates one real VOC sample with both Student and EMA Teacher. All output
is created below a unique /tmp/augseg_a6000_rerun01_smoke child and removed
after an exact cleanup manifest is printed.
"""

from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import dataclass
import fcntl
import hashlib
import importlib.abc
import json
import math
import multiprocessing
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
from typing import Any, Callable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
TEMP_BASE = Path("/tmp/augseg_a6000_rerun01_smoke")
ACCEPTED_HEAD = "ecced7957895108ff8bae6114fcf9c1da0a3f954"
PRETRAIN_SHA256 = "d25b500a332e97a7bcbc11dc1153ec07c2f5c8334fbac2b36d882fe78a01af98"
SPLIT_SHA256 = {
    "labeled": "7dada8ccc5eb6309b4b81e172dfc24f0d83ea72078675083778ca10487182748",
    "unlabeled": "49b06340ca1534c3f15001ea9ce6d2815dd2ae1302390cfbdcfc0b004e17d960",
    "val": "cdc1326d12f69ce5153aa5da04a4d8783e146868d42d19f82f44e74b97ac907d",
}
METHODS = {
    "u1": {
        "config": "exps/boundary_mix_v2_v3/voc_semi662/u1_self_pseudo_saliency_u2u_cutmix_a6000_rerun01_c321_bs16x1_gbs16/config.yaml",
        "accepted_config": "exps/boundary_mix_v2_v3/voc_semi662/u1_self_pseudo_saliency_u2u_cutmix_c321_bs16x1_gbs16/config.yaml",
        "block": "u1_saliency_u2u",
    },
    "u2": {
        "config": "exps/boundary_mix_v2_v3/voc_semi662/u2_cross_view_saliency_u2u_cutmix_a6000_rerun01_c321_bs16x1_gbs16/config.yaml",
        "accepted_config": "exps/boundary_mix_v2_v3/voc_semi662/u2_cross_view_saliency_u2u_cutmix_c321_bs16x1_gbs16/config.yaml",
        "block": "u2_cross_view_saliency_u2u",
    },
    "u3": {
        "config": "exps/boundary_mix_v2_v3/voc_semi662/u3_confidence_filtered_cross_view_saliency_u2u_cutmix_a6000_rerun01_c321_bs16x1_gbs16/config.yaml",
        "accepted_config": "exps/boundary_mix_v2_v3/voc_semi662/u3_confidence_filtered_cross_view_saliency_u2u_cutmix_c321_bs16x1_gbs16/config.yaml",
        "block": "u3_confidence_filtered_cross_view_saliency_u2u",
    },
    "u4": {
        "config": "exps/boundary_mix_v2_v3/voc_semi662/u4_confidence_filtered_self_pseudo_saliency_u2u_cutmix_a6000_rerun01_c321_bs16x1_gbs16/config.yaml",
        "accepted_config": "exps/boundary_mix_v2_v3/voc_semi662/u4_confidence_filtered_self_pseudo_saliency_u2u_cutmix_c321_bs16x1_gbs16/config.yaml",
        "block": "u4_confidence_filtered_self_pseudo_saliency_u2u",
    },
}
METHOD_BLOCKS = tuple(item["block"] for item in METHODS.values())
METHOD_DIAGNOSTIC_MARKERS = {
    "u1": "u1_saliency_u2u",
    "u2": "u2_cross_view_saliency_u2u",
    "u3": "u3_confidence_filtered_cross_view_saliency_u2u",
    "u4": "u4_confidence_filtered_self_pseudo_saliency_u2u",
}
BASE_DIAGNOSTIC_FIELDS = (
    "u1/total_receivers",
    "u1/total_candidate_draws",
    "u1/generated_invalid_candidates",
    "u1/receivers_with_invalid_candidate",
    "u1/selected_invalid_candidates",
    "u1/paste_attempts",
    "u1/nonempty_paste_attempts",
    "u1/paste_attempt_rate",
    "u1/nonempty_mixed_target_rate",
    "u1/near_flat_images",
    "u1/derangement_attempts",
    "u1/rng_policy",
)
SECRET_ENV_NAMES = (
    "WANDB_API_KEY",
    "HF_TOKEN",
    "HUGGINGFACE_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
)
AUTHORIZED_COMMIT_PATHS = frozenset(
    {spec["config"] for spec in METHODS.values()}
    | {"tools/smoke_u1_u4_full_bs16.py", "run_methods_a6000.sh"}
)


class ExternalIntegrationBlocked(importlib.abc.MetaPathFinder):
    """Fail closed if disabled smoke code unexpectedly imports an integration."""

    BLOCKED_ROOTS = frozenset({"wandb", "huggingface_hub", "hf_xet", "xet"})

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        if fullname.split(".", 1)[0] in self.BLOCKED_ROOTS:
            raise RuntimeError(f"external integration import blocked: {fullname}")
        return None


class SingleMethodAction(argparse.Action):
    def __call__(self, parser: argparse.ArgumentParser, namespace: argparse.Namespace, value: str, option_string: str | None = None) -> None:
        if getattr(namespace, self.dest, None) is not None:
            parser.error("exactly one --method may be supplied")
        setattr(namespace, self.dest, value)


@dataclass(frozen=True)
class OwnedInvocation:
    base: Path
    path: Path
    device: int
    inode: int


class TeardownFailure(RuntimeError):
    def __init__(self, errors: Sequence[tuple[str, BaseException]]) -> None:
        self.errors = tuple(errors)
        detail = "; ".join(f"{stage}: {type(error).__name__}: {error}" for stage, error in errors)
        super().__init__(f"smoke teardown failed: {detail}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one isolated full-ResNet101 U1-U4 BS16 CUDA smoke step."
    )
    parser.add_argument(
        "--method", required=True, choices=tuple(METHODS), action=SingleMethodAction
    )
    parser.add_argument(
        "--device",
        default=0,
        type=int,
        help="single physical CUDA device index to expose (default: 0)",
    )
    return parser


def _resolved(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def require_owned_path(root: Path, candidate: Path) -> Path:
    root_resolved = _resolved(root)
    candidate_resolved = _resolved(candidate)
    if candidate_resolved == root_resolved or root_resolved not in candidate_resolved.parents:
        raise ValueError(f"path escapes owned smoke root: {candidate_resolved}")
    return candidate_resolved


def refuse_official_path(candidate: Path, official_paths: list[Path]) -> None:
    candidate_resolved = _resolved(candidate)
    for official in official_paths:
        official_resolved = _resolved(official)
        if (
            candidate_resolved == official_resolved
            or official_resolved in candidate_resolved.parents
            or candidate_resolved in official_resolved.parents
        ):
            raise ValueError(f"smoke path overlaps official path: {candidate_resolved}")


def validate_bounded_steps(epochs: int, labeled_batches: int, unlabeled_batches: int) -> None:
    if (epochs, labeled_batches, unlabeled_batches) != (1, 1, 1):
        raise ValueError(
            "smoke must remain exactly one epoch and one full labeled/unlabeled batch"
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
    ).strip()


def validate_repository_snapshot(
    method: str,
    *,
    head: str,
    parents: Sequence[str],
    changed_files: set[str],
    selected_tracked: bool,
    tracked_clean: bool,
    staged_clean: bool,
    upstream: str,
    ahead_behind: tuple[int, int],
    live_remote: str,
) -> None:
    if method not in METHODS:
        raise RuntimeError("unknown method in repository guard")
    if tuple(parents) != (ACCEPTED_HEAD,):
        raise RuntimeError("smoke requires exactly one parent: the accepted checkpoint")
    if changed_files != set(AUTHORIZED_COMMIT_PATHS):
        raise RuntimeError("reviewed descendant must contain exactly the six authorized files")
    if not selected_tracked:
        raise RuntimeError("selected A6000 config is not tracked by the reviewed descendant")
    if not tracked_clean or not staged_clean:
        raise RuntimeError("tracked worktree and index must both be clean")
    if upstream != head or ahead_behind != (0, 0):
        raise RuntimeError("local HEAD and configured upstream are not synchronized")
    if live_remote != head:
        raise RuntimeError("live remote branch does not equal local HEAD")


def validate_repository(method: str) -> str:
    if _resolved(Path(_git_output("rev-parse", "--show-toplevel"))) != REPO_ROOT:
        raise RuntimeError("repository root mismatch")
    head = _git_output("rev-parse", "HEAD")
    parent_line = _git_output("rev-list", "--parents", "-n", "1", "HEAD").split()
    parents = parent_line[1:]
    changed_files = set(
        _git_output("diff", "--name-only", f"{ACCEPTED_HEAD}..{head}").splitlines()
    )
    selected_tracked = (
        _git_output("ls-files", "--error-unmatch", METHODS[method]["config"])
        == METHODS[method]["config"]
    )
    tracked_clean = subprocess.run(
        ["git", "diff", "--quiet", "--ignore-submodules", "--"],
        cwd=REPO_ROOT,
        check=False,
    ).returncode == 0
    staged_clean = subprocess.run(
        ["git", "diff", "--cached", "--quiet", "--ignore-submodules", "--"],
        cwd=REPO_ROOT,
        check=False,
    ).returncode == 0
    upstream = _git_output("rev-parse", "@{upstream}")
    ahead_behind_values = _git_output(
        "rev-list", "--left-right", "--count", "HEAD...@{upstream}"
    ).split()
    live_remote_lines = _git_output(
        "ls-remote",
        "--heads",
        "origin",
        "refs/heads/feature/auto-run-12-methods-gpu-suite",
    ).splitlines()
    if len(live_remote_lines) != 1:
        raise RuntimeError("live remote branch lookup did not return exactly one ref")
    validate_repository_snapshot(
        method,
        head=head,
        parents=parents,
        changed_files=changed_files,
        selected_tracked=selected_tracked,
        tracked_clean=tracked_clean,
        staged_clean=staged_clean,
        upstream=upstream,
        ahead_behind=tuple(int(value) for value in ahead_behind_values),
        live_remote=live_remote_lines[0].split()[0],
    )
    return head


def _read_nonempty(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _entry_is_readable(voc_root: Path, sample_id: str) -> bool:
    return os.access(voc_root / "JPEGImages" / f"{sample_id}.jpg", os.R_OK) and os.access(
        voc_root / "SegmentationClassAug" / f"{sample_id}.png", os.R_OK
    )


def select_readable_entries(split: Path, voc_root: Path, count: int) -> list[str]:
    selected = [item for item in _read_nonempty(split) if _entry_is_readable(voc_root, item)][:count]
    if len(selected) != count:
        raise RuntimeError(f"{split} has only {len(selected)} readable image/annotation pairs")
    return selected


def _official_paths() -> list[Path]:
    paths: list[Path] = []
    for spec in METHODS.values():
        config_path = REPO_ROOT / spec["config"]
        accepted_path = REPO_ROOT / spec["accepted_config"]
        paths.extend(
            (config_path.parent / "log", config_path, accepted_path.parent / "log", accepted_path)
        )
    output_root = REPO_ROOT / "exp_boundary_mix_v2_v3"
    paths.append(output_root)
    return paths


def _allocate_invocation_root(method: str) -> tuple[OwnedInvocation, int]:
    if not TEMP_BASE.exists():
        TEMP_BASE.mkdir(mode=0o700)
    if not TEMP_BASE.is_dir() or TEMP_BASE.is_symlink():
        raise RuntimeError(f"unsafe smoke base: {TEMP_BASE}")
    lock_fd = os.open(TEMP_BASE, os.O_RDONLY | os.O_DIRECTORY)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(lock_fd)
        raise RuntimeError("another A6000 cohort smoke invocation is active")
    root = Path(tempfile.mkdtemp(prefix=f"{method}-", dir=TEMP_BASE))
    os.chmod(root, 0o700)
    root = require_owned_path(TEMP_BASE, root)
    metadata = root.lstat()
    ownership = OwnedInvocation(
        base=_resolved(TEMP_BASE),
        path=root,
        device=int(metadata.st_dev),
        inode=int(metadata.st_ino),
    )
    return ownership, lock_fd


def _write_list(path: Path, entries: list[str]) -> None:
    path.write_text("".join(f"{item}\n" for item in entries), encoding="utf-8")


def derive_smoke_config(method: str, root: Path) -> tuple[dict[str, Any], Path]:
    import yaml

    source_path = REPO_ROOT / METHODS[method]["config"]
    cfg = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    cfg = copy.deepcopy(cfg)
    unique_name = f"smoke_{method}_a6000_rerun01_{root.name}"
    split_dir = require_owned_path(root, root / "pascal_smoke_splits")
    split_dir.mkdir()
    voc_root = REPO_ROOT / "data" / "VOC2012"
    canonical = {
        "labeled": REPO_ROOT / "data/splitsall/pascal_u2pl/662/labeled.txt",
        "unlabeled": REPO_ROOT / "data/splitsall/pascal_u2pl/662/unlabeled.txt",
        "val": REPO_ROOT / "data/splitsall/pascal_u2pl/val.txt",
    }
    for key, expected in SPLIT_SHA256.items():
        if _sha256(canonical[key]) != expected:
            raise RuntimeError(f"canonical {key} split identity mismatch")
    pretrain = REPO_ROOT / "pretrained/resnet101.pth"
    if _sha256(pretrain) != PRETRAIN_SHA256:
        raise RuntimeError("pretrained ResNet101 identity mismatch")
    labeled = select_readable_entries(canonical["labeled"], voc_root, 16)
    unlabeled = select_readable_entries(canonical["unlabeled"], voc_root, 16)
    validation = select_readable_entries(canonical["val"], voc_root, 1)
    labeled_path = split_dir / "labeled.txt"
    unlabeled_path = split_dir / "unlabeled.txt"
    val_path = split_dir / "val.txt"
    _write_list(labeled_path, labeled)
    _write_list(unlabeled_path, unlabeled)
    _write_list(val_path, validation)

    output = require_owned_path(root, root / "output")
    runtime_dir = require_owned_path(root, root / "runtime")
    runtime_dir.mkdir()
    for candidate in (output, runtime_dir / "log"):
        refuse_official_path(candidate, _official_paths())

    cfg["name"] = unique_name
    cfg["run"]["name"] = unique_name
    cfg["run"]["log_every"] = 1
    cfg["wandb"].update({"enable": False, "name": unique_name})
    cfg["hf"].update(
        {
            "enabled": False,
            "auto_download": False,
            "auto_upload": False,
            "upload_every_epoch": False,
            "repo_id": "disabled/smoke",
            "path_in_repo": f"disabled/{unique_name}/latest.tar.gz",
        }
    )
    cfg["saver"].update(
        {"auto_resume": False, "auto_profile_dir": False, "snapshot_dir": str(output)}
    )
    cfg["checkpoint"].update(
        {"auto_resume": False, "save_latest": False, "save_best": False}
    )
    cfg["dataset"]["train"].update(
        {"data_root": str(voc_root), "data_list": str(labeled_path), "batch_size": 16}
    )
    cfg["dataset"]["val"].update(
        {"data_root": str(voc_root), "data_list": str(val_path), "batch_size": 1}
    )
    # The loader derives its epoch length as 10582 - n_sup. This yields 16
    # samples in each temporary list and therefore exactly one full BS16 step.
    cfg["dataset"]["n_sup"] = 10566
    cfg["dataset"]["workers"] = 4
    cfg["trainer"]["epochs"] = 1
    cfg["trainer"]["evaluate_student"] = True
    cfg["net"]["encoder"]["pretrain"] = str(pretrain)

    config_path = runtime_dir / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    validate_smoke_config(cfg, method, root)
    return cfg, config_path


def validate_smoke_config(cfg: dict[str, Any], method: str, root: Path) -> None:
    enabled = [block for block in METHOD_BLOCKS if bool(cfg.get(block, {}).get("enabled"))]
    if enabled != [METHODS[method]["block"]]:
        raise ValueError(f"method/config identity mismatch: {enabled}")
    if cfg["dataset"]["train"]["batch_size"] != 16 or cfg["dataset"]["workers"] != 4:
        raise ValueError("smoke requires batch size 16 and four workers")
    if cfg["dataset"]["train"]["crop"]["size"] != [321, 321]:
        raise ValueError("smoke requires crop [321, 321]")
    validate_bounded_steps(cfg["trainer"]["epochs"], 1, 1)
    if cfg["wandb"]["enable"]:
        raise ValueError("W&B must be disabled")
    hf = cfg["hf"]
    if any(bool(hf[key]) for key in ("enabled", "auto_download", "auto_upload", "upload_every_epoch")):
        raise ValueError("all HF operations must be disabled")
    if cfg["checkpoint"]["auto_resume"] or cfg["saver"]["auto_resume"]:
        raise ValueError("all checkpoint resume paths must be disabled")
    output = require_owned_path(root, Path(cfg["saver"]["snapshot_dir"]))
    refuse_official_path(output, _official_paths())


def _blocked_hf_call(*args: Any, **kwargs: Any) -> dict[str, Any]:
    cfg = kwargs.get("cfg")
    hf = (cfg or {}).get("hf", {})
    if hf.get("enabled") or hf.get("auto_download") or hf.get("auto_upload"):
        raise RuntimeError("unexpected HF integration invocation")
    return {"enabled": False, "downloaded": False, "uploaded": False, "reason": "smoke guard"}


def _free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _unwrap(model: Any) -> Any:
    return model.module if hasattr(model, "module") else model


def _assert_full_resnet101(model: Any, label: str) -> None:
    inner = _unwrap(model)
    blocks = sum(1 for module in inner.encoder.modules() if module.__class__.__name__ == "Bottleneck")
    if blocks != 33:
        raise RuntimeError(f"{label} is not full ResNet101: found {blocks} Bottleneck blocks")


def _assert_finite_model(torch: Any, model: Any, label: str) -> None:
    saw_gradient = False
    for name, parameter in model.named_parameters():
        if not bool(torch.isfinite(parameter).all()):
            raise FloatingPointError(f"nonfinite {label} parameter: {name}")
        if parameter.grad is not None:
            saw_gradient = True
            if not bool(torch.isfinite(parameter.grad).all()):
                raise FloatingPointError(f"nonfinite {label} gradient: {name}")
    if label == "Student" and not saw_gradient:
        raise RuntimeError("Student produced no gradients")


def _install_train_probe(train_semi: Any, torch: Any, method: str, result: dict[str, Any]) -> None:
    original_train = train_semi.train

    class DiagnosticLoggerProxy:
        def __init__(self, logger: Any) -> None:
            self._logger = logger

        def __getattr__(self, name: str) -> Any:
            return getattr(self._logger, name)

        def info(self, message: str, *args: Any, **kwargs: Any) -> Any:
            marker_values = frozenset(METHOD_DIAGNOSTIC_MARKERS.values())
            if (
                message.startswith("[%s] epoch=%d step=%d global_iter=%d")
                and args
                and args[0] in marker_values
            ):
                if "method_log_marker" in result:
                    raise RuntimeError("U-method diagnostic marker was logged more than once")
                result["method_log_marker"] = {
                    "marker": args[0],
                    "epoch": args[1],
                    "step": args[2],
                    "global_iteration": args[3],
                }
            return self._logger.info(message, *args, **kwargs)

    def probed_train(*args: Any, **kwargs: Any) -> tuple[float, float]:
        model, teacher = args[0], args[1]
        loader_l, loader_u, cfg = args[5], args[6], args[10]
        validate_smoke_config(cfg, method, Path(cfg["saver"]["snapshot_dir"]).parent)
        if len(loader_l) != 1 or len(loader_u) != 1:
            raise RuntimeError("bounded loader invariant failed")
        if loader_l.batch_size != 16 or loader_u.batch_size != 16:
            raise RuntimeError("full BS16 loader invariant failed")
        if loader_l.num_workers != 4 or loader_u.num_workers != 4:
            raise RuntimeError("four-worker invariant failed")
        _assert_full_resnet101(model, "Student")
        _assert_full_resnet101(teacher, "EMA Teacher")
        torch.cuda.synchronize()
        result["model_initialization"] = {
            "current_allocated": int(torch.cuda.memory_allocated()),
            "current_reserved": int(torch.cuda.memory_reserved()),
            "peak_allocated": int(torch.cuda.max_memory_allocated()),
            "peak_reserved": int(torch.cuda.max_memory_reserved()),
        }
        teacher_before = next(_unwrap(teacher).parameters()).detach().clone()
        torch.cuda.reset_peak_memory_stats()
        call_args = list(args)
        call_args[9] = DiagnosticLoggerProxy(call_args[9])
        losses = original_train(*call_args, **kwargs)
        torch.cuda.synchronize()
        result["first_complete_step"] = {
            "peak_allocated": int(torch.cuda.max_memory_allocated()),
            "peak_reserved": int(torch.cuda.max_memory_reserved()),
        }
        result["steady_bounded_step"] = {
            "measured": False,
            "reason": "one-step bound has no separate steady-state iteration",
        }
        if not all(float("-inf") < float(value) < float("inf") for value in losses):
            raise FloatingPointError(f"nonfinite losses: {losses}")
        _assert_finite_model(torch, model, "Student")
        _assert_finite_model(torch, teacher, "EMA Teacher")
        teacher_after = next(_unwrap(teacher).parameters()).detach()
        if torch.equal(teacher_before, teacher_after):
            raise RuntimeError("EMA Teacher did not update")
        helper_record = result.get("method_diagnostic")
        log_marker = result.get("method_log_marker")
        if not isinstance(helper_record, dict) or not isinstance(log_marker, dict):
            raise RuntimeError("selected U-method helper and log diagnostics were not both observed")
        for position in ("epoch", "step", "global_iteration"):
            if helper_record.get(position) != log_marker.get(position):
                raise RuntimeError(f"U-method helper/log diagnostic {position} mismatch")
        helper_record["marker"] = log_marker["marker"]
        validate_method_diagnostic(method, helper_record)
        result["losses"] = {"supervised": float(losses[0]), "unsupervised": float(losses[1])}
        result["ema_updated"] = True
        result["full_resnet101_student_teacher"] = True
        return losses

    train_semi.train = probed_train


def _finite_number(fields: dict[str, Any], name: str) -> float:
    if name not in fields:
        raise ValueError(f"missing required diagnostic field: {name}")
    value = fields[name]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"diagnostic field is not numeric: {name}")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"nonfinite diagnostic field: {name}")
    return numeric


def _integer_count(fields: dict[str, Any], name: str) -> int:
    numeric = _finite_number(fields, name)
    if not numeric.is_integer():
        raise ValueError(f"diagnostic count is not integral: {name}")
    return int(numeric)


def validate_method_diagnostic(
    method: str,
    record: dict[str, Any],
    *,
    batch_size: int = 16,
    candidates: int = 8,
    crop_size: tuple[int, int] = (321, 321),
) -> None:
    if method not in METHOD_DIAGNOSTIC_MARKERS:
        raise ValueError(f"unknown smoke method: {method}")
    expected_marker = METHOD_DIAGNOSTIC_MARKERS[method]
    marker = record.get("marker")
    if marker != expected_marker:
        raise ValueError(f"wrong method diagnostic marker: expected {expected_marker}, got {marker}")
    if record.get("method") != method:
        raise ValueError("diagnostic method identity mismatch")
    if (record.get("epoch"), record.get("step"), record.get("global_iteration")) != (0, 0, 0):
        raise ValueError("diagnostic is not from epoch 0, step 0, global iteration 0")
    fields = record.get("fields")
    if not isinstance(fields, dict):
        raise ValueError("diagnostic fields are missing")
    for name in BASE_DIAGNOSTIC_FIELDS:
        if name == "u1/rng_policy":
            if fields.get(name) != "u1_rng_policy_v1":
                raise ValueError("incorrect U-method RNG policy")
        else:
            _finite_number(fields, name)

    receivers = _integer_count(fields, "u1/total_receivers")
    candidate_draws = _integer_count(fields, "u1/total_candidate_draws")
    generated_invalid = _integer_count(fields, "u1/generated_invalid_candidates")
    receivers_invalid = _integer_count(fields, "u1/receivers_with_invalid_candidate")
    selected_invalid = _integer_count(fields, "u1/selected_invalid_candidates")
    paste_attempts = _integer_count(fields, "u1/paste_attempts")
    nonempty_pastes = _integer_count(fields, "u1/nonempty_paste_attempts")
    near_flat = _integer_count(fields, "u1/near_flat_images")
    derangement_attempts = _integer_count(fields, "u1/derangement_attempts")
    if receivers != batch_size or candidate_draws != batch_size * candidates:
        raise ValueError("receiver/candidate counts do not match BS16 and K=8")
    if paste_attempts != receivers or not 0 <= nonempty_pastes <= paste_attempts:
        raise ValueError("paste counts are inconsistent with receiver count")
    if not 0 <= generated_invalid <= candidate_draws:
        raise ValueError("invalid-candidate count is out of range")
    if not 0 <= receivers_invalid <= receivers or not 0 <= selected_invalid <= receivers:
        raise ValueError("receiver/selection invalid counts are out of range")
    if selected_invalid > generated_invalid or not 0 <= near_flat <= receivers:
        raise ValueError("selected-invalid or near-flat count is inconsistent")
    if not 1 <= derangement_attempts <= 128:
        raise ValueError("derangement-attempt count is out of range")
    if not math.isclose(_finite_number(fields, "u1/paste_attempt_rate"), paste_attempts / receivers):
        raise ValueError("paste-attempt rate is inconsistent")
    if not math.isclose(
        _finite_number(fields, "u1/nonempty_mixed_target_rate"), nonempty_pastes / receivers
    ):
        raise ValueError("nonempty-paste rate is inconsistent")

    filtered_prefix = "u3" if method == "u3" else ("u4" if method == "u4" else None)
    if filtered_prefix is None:
        if any(name.startswith(("u3/", "u4/")) for name in fields):
            raise ValueError("unfiltered method emitted confidence-filter-only diagnostics")
        return
    required_filtered = (
        "total_probe_pixels",
        "confidence_valid_probe_pixels",
        "ignored_label_pixels",
        "zero_valid_donors",
        "zero_valid_donor_rate",
        "all_valid_donors",
        "all_valid_donor_rate",
        "finite_probe_events",
        "near_flat_donor_rate",
    )
    filtered = {
        name: _finite_number(fields, f"{filtered_prefix}/{name}") for name in required_filtered
    }
    total_pixels = batch_size * crop_size[0] * crop_size[1]
    if int(filtered["total_probe_pixels"]) != total_pixels:
        raise ValueError("confidence-filtered probe pixel count is inconsistent")
    for name in ("total_probe_pixels", "confidence_valid_probe_pixels", "ignored_label_pixels"):
        if not filtered[name].is_integer():
            raise ValueError(f"confidence-filtered count is not integral: {name}")
    if not 0 <= filtered["confidence_valid_probe_pixels"] <= total_pixels:
        raise ValueError("confidence-valid pixel count is out of range")
    if not 0 <= filtered["ignored_label_pixels"] <= total_pixels:
        raise ValueError("ignored-label pixel count is out of range")
    zero_donors = int(filtered["zero_valid_donors"])
    all_donors = int(filtered["all_valid_donors"])
    if any(not filtered[name].is_integer() for name in ("zero_valid_donors", "all_valid_donors", "finite_probe_events")):
        raise ValueError("confidence-filtered donor/event count is not integral")
    if not 0 <= zero_donors <= receivers or not 0 <= all_donors <= receivers:
        raise ValueError("confidence-filtered donor count is out of range")
    if zero_donors + all_donors > receivers or int(filtered["finite_probe_events"]) != 1:
        raise ValueError("confidence-filtered donor/event counts are inconsistent")
    if not math.isclose(filtered["zero_valid_donor_rate"], zero_donors / receivers):
        raise ValueError("zero-valid-donor rate is inconsistent")
    if not math.isclose(filtered["all_valid_donor_rate"], all_donors / receivers):
        raise ValueError("all-valid-donor rate is inconsistent")
    if not math.isclose(filtered["near_flat_donor_rate"], near_flat / receivers):
        raise ValueError("near-flat-donor rate is inconsistent")


def _install_method_diagnostic_probe(
    train_semi: Any, method: str, result: dict[str, Any]
) -> None:
    original_helper = train_semi.apply_u1_saliency_u2u

    def probed_helper(*args: Any, **kwargs: Any) -> Any:
        if "method_diagnostic" in result:
            raise RuntimeError("selected U-method helper executed more than once")
        output = original_helper(*args, **kwargs)
        fields = output[3].as_dict()
        if method == "u4":
            fields = {
                (name.replace("u3/", "u4/", 1) if name.startswith("u3/") else name): value
                for name, value in fields.items()
            }
        record = {
            "method": method,
            "epoch": kwargs.get("epoch"),
            "step": kwargs.get("step"),
            "global_iteration": kwargs.get("absolute_global_iteration"),
            "fields": fields,
        }
        result["method_diagnostic"] = record
        return output

    train_semi.apply_u1_saliency_u2u = probed_helper


def _verify_smoke_outputs(root: Path, result: dict[str, Any], method: str) -> None:
    output = root / "output"
    iter_csv = output / "iter_metrics.csv"
    if not iter_csv.is_file():
        raise RuntimeError("accepted training path did not write isolated iter_metrics.csv")
    with iter_csv.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    train_rows = [row for row in rows if row.get("meta/log_type") == "train_iter"]
    if len(train_rows) != 1:
        raise RuntimeError(f"expected one train_iter diagnostic row, found {len(train_rows)}")
    validate_method_diagnostic(method, result.get("method_diagnostic", {}))
    for forbidden in ("ckpt.pth", "ckpt_best.pth", "_hf_bundle"):
        if (output / forbidden).exists():
            raise RuntimeError(f"forbidden smoke artifact created: {forbidden}")
    for required in ("run_id.txt", "manifest.json", "epoch_metrics.csv"):
        if not (output / required).is_file():
            raise RuntimeError(f"missing isolated training-path evidence: {required}")
    result["diagnostic_train_rows"] = 1


def cleanup_manifest(root: Path) -> list[dict[str, Any]]:
    require_owned_path(TEMP_BASE, root)
    entries: list[dict[str, Any]] = []
    for current, directories, files in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        for name in sorted(directories):
            path = current_path / name
            entries.append({"path": str(path.relative_to(root)), "type": "symlink" if path.is_symlink() else "directory"})
        for name in sorted(files):
            path = current_path / name
            entries.append(
                {
                    "path": str(path.relative_to(root)),
                    "type": "symlink" if path.is_symlink() else "file",
                    "bytes": path.lstat().st_size,
                }
            )
    return entries


def validate_owned_child(ownership: OwnedInvocation, candidate: str | Path) -> Path:
    if candidate is None or not str(candidate).strip():
        raise ValueError("cleanup target is empty")
    raw_target = Path(candidate).expanduser()
    if raw_target == Path("/") or raw_target.is_symlink():
        raise ValueError(f"unsafe cleanup target: {raw_target}")
    base = _resolved(ownership.base)
    target = _resolved(raw_target)
    if base == Path("/") or target == Path("/") or target == base:
        raise ValueError(f"refusing broad cleanup target: {target}")
    if target.parent != base or target != _resolved(ownership.path):
        raise ValueError(f"cleanup target is not the owned direct child: {target}")
    if not target.is_dir() or target.is_symlink():
        raise ValueError(f"owned cleanup target is missing or unsafe: {target}")
    metadata = target.lstat()
    if (int(metadata.st_dev), int(metadata.st_ino)) != (ownership.device, ownership.inode):
        raise ValueError("cleanup target identity changed after allocation")
    refuse_official_path(target, _official_paths())
    return target


def cleanup_owned_root(ownership: OwnedInvocation, candidate: str | Path | None = None) -> None:
    target = validate_owned_child(ownership, ownership.path if candidate is None else candidate)
    shutil.rmtree(target)


def attempt_teardown_stages(
    stages: Sequence[tuple[str, Callable[[], None]]],
) -> tuple[list[str], list[tuple[str, BaseException]]]:
    attempted: list[str] = []
    errors: list[tuple[str, BaseException]] = []
    for name, action in stages:
        attempted.append(name)
        try:
            action()
        except BaseException as error:
            errors.append((name, error))
            print(
                f"TEARDOWN_ERROR stage={name} error={type(error).__name__}: {error}",
                file=sys.stderr,
                flush=True,
            )
    return attempted, errors


def raise_after_teardown(
    primary_error: BaseException | None,
    primary_traceback: Any,
    teardown_errors: Sequence[tuple[str, BaseException]],
) -> None:
    if primary_error is not None:
        for stage, error in teardown_errors:
            primary_error.add_note(
                f"teardown error in {stage}: {type(error).__name__}: {error}"
            )
        raise primary_error.with_traceback(primary_traceback)
    if teardown_errors:
        raise TeardownFailure(teardown_errors)


def run(method: str, device: int) -> int:
    if device < 0:
        raise ValueError("--device must identify exactly one non-negative CUDA device")
    if "SLURM_JOB_ID" in os.environ:
        raise RuntimeError("refusing inherited SLURM distributed context")
    expected_distributed = {"WORLD_SIZE": "1", "RANK": "0", "LOCAL_RANK": "0"}
    for key, expected in expected_distributed.items():
        if key in os.environ and os.environ[key] != expected:
            raise RuntimeError(f"refusing inherited multi-process setting: {key}")

    head = validate_repository(method)
    ownership, lock_fd = _allocate_invocation_root(method)
    root = ownership.path
    child_pids_before = {child.pid for child in multiprocessing.active_children()}
    old_cwd = Path.cwd()
    old_env = dict(os.environ)
    old_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    finder = ExternalIntegrationBlocked()
    result: dict[str, Any] = {"method": method, "git_head": head, "temporary_root": str(root)}
    primary_error: BaseException | None = None
    primary_traceback: Any = None

    def interrupt(signum: int, frame: Any) -> None:
        raise KeyboardInterrupt(f"received signal {signum}")

    def teardown_process_group() -> None:
        torch_module = sys.modules.get("torch")
        if torch_module is None:
            return
        dist_module = getattr(torch_module, "distributed", None)
        if dist_module is not None and dist_module.is_available() and dist_module.is_initialized():
            dist_module.destroy_process_group()

    def teardown_workers() -> None:
        worker_errors: list[tuple[str, BaseException]] = []
        for child in multiprocessing.active_children():
            if child.pid in child_pids_before:
                continue
            try:
                child.terminate()
                child.join(timeout=5)
            except BaseException as error:
                worker_errors.append((f"worker-{child.pid}", error))
        if worker_errors:
            raise TeardownFailure(worker_errors)

    def report_cleanup_manifest() -> None:
        manifest = cleanup_manifest(root)
        print(
            "CLEANUP_MANIFEST="
            + json.dumps({"root": str(root), "entries": manifest}, sort_keys=True),
            flush=True,
        )

    def cleanup_invocation() -> None:
        cleanup_owned_root(ownership)
        print("CLEANUP_COMPLETE=" + str(root), flush=True)

    def release_lock() -> None:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        finally:
            os.close(lock_fd)

    def restore_runtime_state() -> None:
        if finder in sys.meta_path:
            sys.meta_path.remove(finder)
        signal_errors: list[BaseException] = []
        for sig, handler in old_handlers.items():
            try:
                signal.signal(sig, handler)
            except BaseException as error:
                signal_errors.append(error)
        os.chdir(old_cwd)
        os.environ.clear()
        os.environ.update(old_env)
        if signal_errors:
            raise TeardownFailure(
                [(f"restore-signal-{index}", error) for index, error in enumerate(signal_errors)]
            )

    try:
        os.chdir(REPO_ROOT)
        os.environ.update(
            {
                "CUDA_VISIBLE_DEVICES": str(device),
                "WORLD_SIZE": "1",
                "RANK": "0",
                "LOCAL_RANK": "0",
                "MASTER_ADDR": "127.0.0.1",
                "MASTER_PORT": str(_free_local_port()),
                "WANDB_MODE": "disabled",
                "HF_HUB_OFFLINE": "1",
                "TMPDIR": str(root),
            }
        )
        for name in SECRET_ENV_NAMES:
            os.environ.pop(name, None)
        for sig in old_handlers:
            signal.signal(sig, interrupt)
        cfg, config_path = derive_smoke_config(method, root)
        already_loaded = ExternalIntegrationBlocked.BLOCKED_ROOTS.intersection(sys.modules)
        if already_loaded:
            raise RuntimeError(f"external integration was loaded before smoke guard: {sorted(already_loaded)}")
        sys.meta_path.insert(0, finder)
        import torch
        import torch.distributed as dist
        import train_semi

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("smoke requires exactly one visible CUDA device")
        if "RTX A6000" not in torch.cuda.get_device_name(0):
            raise RuntimeError(f"smoke requires RTX A6000, got {torch.cuda.get_device_name(0)!r}")
        torch.cuda.reset_peak_memory_stats(0)
        train_semi.maybe_download_hf_bundle = _blocked_hf_call
        train_semi.maybe_upload_hf_bundle = _blocked_hf_call
        _install_method_diagnostic_probe(train_semi, method, result)
        _install_train_probe(train_semi, torch, method, result)
        args = argparse.Namespace(config=str(config_path), local_rank=0, seed=2, port=int(os.environ["MASTER_PORT"]))
        train_semi.main(args)
        torch.cuda.synchronize(0)
        _verify_smoke_outputs(root, result, method)
    except BaseException as error:
        primary_error = error
        primary_traceback = sys.exc_info()[2]
    finally:
        attempted, teardown_errors = attempt_teardown_stages(
            (
                ("process-group", teardown_process_group),
                ("workers", teardown_workers),
                ("cleanup-manifest", report_cleanup_manifest),
                ("invocation-child", cleanup_invocation),
                ("lock", release_lock),
                ("runtime-state", restore_runtime_state),
            )
        )
        result["teardown_stages_attempted"] = attempted

    raise_after_teardown(primary_error, primary_traceback, teardown_errors)
    result["status"] = "PASS"
    print("SMOKE_RESULT=" + json.dumps(result, sort_keys=True), flush=True)
    return 0


def main() -> int:
    args = build_parser().parse_args()
    return run(args.method, args.device)


if __name__ == "__main__":
    raise SystemExit(main())

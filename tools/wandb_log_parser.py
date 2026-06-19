#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
from typing import Any


SEGMENT_RE = re.compile(r"(?P<method>.+)_e(?P<start>\d+)_to_e(?P<target>\d+)\.log$")
TRAIN_RE = re.compile(
    r"Epoch/Iter\s+\[(?P<target>\d+)\s*:\s*(?P<epoch>\d+)\s*/\s*(?P<iter>\d+)\]\.\s+"
    r"Sup:(?P<sup>[-+0-9.eE]+|nan|inf|-inf)\((?P<sup_avg>[-+0-9.eE]+|nan|inf|-inf)\)\s+"
    r"Uns:(?P<uns>[-+0-9.eE]+|nan|inf|-inf)\((?P<uns_avg>[-+0-9.eE]+|nan|inf|-inf)\)\s+"
    r"Pseudo:(?P<pseudo>[-+0-9.eE]+|nan|inf|-inf)\((?P<pseudo_avg>[-+0-9.eE]+|nan|inf|-inf)\).*?"
    r"LR:(?P<lr>[-+0-9.eE]+|nan|inf|-inf)"
)
TEST_RE = re.compile(
    r"<<Test>>\s*-\s*Epoch:\s*(?P<epoch>\d+)\.\s+MIoU:\s*"
    r"(?P<miou_stu>[-+0-9.eE]+|nan|inf|-inf)\s*/\s*(?P<miou_ema>[-+0-9.eE]+|nan|inf|-inf)\.\s+"
    r".*?Best-STU:\s*(?P<best_stu>[-+0-9.eE]+|nan|inf|-inf)\s*/\s*(?P<best_stu_epoch>\d+).*?"
    r"Best-EMA:\s*(?P<best_ema>[-+0-9.eE]+|nan|inf|-inf)\s*/\s*(?P<best_ema_epoch>\d+)"
)
KEY_VALUE_RE = re.compile(r"(?P<key>[A-Za-z_][A-Za-z0-9_./-]*)=(?P<value>[-+0-9.eE]+|nan|inf|-inf|[A-Za-z_][A-Za-z0-9_.-]*)")
RUN_ID_RE = re.compile(r"\[log\]\s+run_id\s*=\s*(?P<run_id>\S+)")
RESUME_RE = re.compile(r"start_epoch=(?P<start_epoch>\d+),\s*best_miou=(?P<best_miou>[-+0-9.eE]+|nan|inf|-inf)")

FATAL_PATTERNS = (
    re.compile(r"\bTraceback\b"),
    re.compile(r"\bRuntimeError\b"),
    re.compile(r"\b(?:CUDA out of memory|torch\.OutOfMemoryError|out of memory)\b", re.IGNORECASE),
    re.compile(r"\bSup\s*:\s*(?:nan|inf|-inf)\b", re.IGNORECASE),
    re.compile(r"\bUns\s*:\s*(?:nan|inf|-inf)\b", re.IGNORECASE),
    re.compile(r"\bloss\s*=\s*(?:nan|inf|-inf)\b", re.IGNORECASE),
    re.compile(r"\btotal_loss\s*=\s*(?:nan|inf|-inf)\b", re.IGNORECASE),
    re.compile(r"\bL_BCR\s*=\s*(?:nan|inf|-inf)\b", re.IGNORECASE),
    re.compile(r"\bbcr/loss_bcr\s*=\s*(?:nan|inf|-inf)\b", re.IGNORECASE),
    re.compile(r"\bnon-finite\s+(?:loss|gradient|grad)\b", re.IGNORECASE),
    re.compile(r"\b(?:loss|gradient|grad)\s+(?:is|became|produced)\s+(?:nan|inf|-inf|non-finite)\b", re.IGNORECASE),
)


def _float(value: str) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isfinite(parsed):
        return parsed
    return None


def _number_or_string(value: str) -> Any:
    parsed = _float(value)
    if parsed is not None:
        return parsed
    if value.lower() in {"nan", "inf", "-inf"}:
        return None
    return value


def infer_method_and_segment_from_path(path: Path) -> dict[str, Any]:
    match = SEGMENT_RE.match(path.name)
    if match:
        return {
            "method": match.group("method"),
            "segment_start_epoch": int(match.group("start")),
            "segment_target_epoch": int(match.group("target")),
            "log_path": str(path),
        }
    return {
        "method": path.stem,
        "segment_start_epoch": None,
        "segment_target_epoch": None,
        "log_path": str(path),
    }


def is_fatal_line(line: str) -> bool:
    return any(pattern.search(line) for pattern in FATAL_PATTERNS)


def parse_line(line: str, context: dict[str, Any] | None = None) -> dict[str, Any] | None:
    context = context or {}
    if is_fatal_line(line):
        return {"event_type": "fatal", "fatal": True, "message": line.strip()}

    match = RUN_ID_RE.search(line)
    if match:
        return {"event_type": "metadata", "run_id": match.group("run_id")}

    match = RESUME_RE.search(line)
    if match:
        event = {"event_type": "metadata", "resumed_start_epoch": int(match.group("start_epoch"))}
        best = _float(match.group("best_miou"))
        if best is not None:
            event["val/best_mIoU"] = best * 100.0 if best <= 1.0 else best
        return event

    match = TRAIN_RE.search(line)
    if match:
        epoch = int(match.group("epoch"))
        iteration = int(match.group("iter"))
        event = {
            "event_type": "train",
            "epoch": epoch,
            "iter": iteration,
            "step": epoch * 100000 + iteration,
            "segment/target_epoch": int(match.group("target")),
        }
        values = {
            "train/sup_loss": match.group("sup"),
            "train/sup_loss_avg": match.group("sup_avg"),
            "train/unsup_loss": match.group("uns"),
            "train/unsup_loss_avg": match.group("uns_avg"),
            "train/pseudo_ratio": match.group("pseudo"),
            "train/pseudo_ratio_avg": match.group("pseudo_avg"),
            "train/lr": match.group("lr"),
        }
        event.update({key: value for key, raw in values.items() if (value := _float(raw)) is not None})
        if "train/sup_loss" in event and "train/unsup_loss" in event:
            event["train/total_loss"] = event["train/sup_loss"] + event["train/unsup_loss"]
        return event

    match = TEST_RE.search(line)
    if match:
        epoch = int(match.group("epoch"))
        event = {
            "event_type": "val",
            "epoch": epoch,
            "step": epoch * 100000,
            "val/best_STU_epoch": int(match.group("best_stu_epoch")),
            "val/best_EMA_epoch": int(match.group("best_ema_epoch")),
        }
        for key, group in (
            ("val/mIoU_STU", "miou_stu"),
            ("val/mIoU", "miou_ema"),
            ("val/best_mIoU_STU", "best_stu"),
            ("val/best_mIoU", "best_ema"),
        ):
            value = _float(match.group(group))
            if value is not None:
                event[key] = value
        return event

    if "[boundary_compatibility]" in line:
        event: dict[str, Any] = {"event_type": "boundary_compatibility"}
        for match in KEY_VALUE_RE.finditer(line):
            key = match.group("key")
            value = _number_or_string(match.group("value"))
            if value is None:
                continue
            mapped = {
                "L_BCR": "bcr/loss_bcr",
                "mean_JS": "bcr/mean_js",
                "num_pairs_per_image": "bcr/components",
            }.get(key, key)
            event[mapped] = value
        if "epoch" in event:
            event["epoch"] = int(event["epoch"])
        if "step" in event:
            event["iter"] = int(event["step"])
            if "epoch" in event:
                event["step"] = int(event["epoch"]) * 100000 + int(event["iter"])
        return event if len(event) > 1 else None

    if "[component]" in line or "[boundary_component]" in line or "[csl" in line.lower() or "[saliency" in line.lower():
        event = {"event_type": "debug"}
        for match in KEY_VALUE_RE.finditer(line):
            key = match.group("key")
            value = _number_or_string(match.group("value"))
            if value is None:
                continue
            if key == "epoch":
                event["epoch"] = int(value)
            elif key == "step":
                event["iter"] = int(value)
            elif key.startswith(("bcr/", "csl/", "saliency/", "affinity/", "teacher/")):
                event[key] = value
        if "epoch" in event and "iter" in event:
            event["step"] = int(event["epoch"]) * 100000 + int(event["iter"])
        return event if len(event) > 1 else None

    return None


def parse_log_file(path: Path) -> list[dict[str, Any]]:
    meta = infer_method_and_segment_from_path(path)
    events: list[dict[str, Any]] = []
    context = dict(meta)
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, 1):
            event = parse_line(line, context)
            if event is None:
                continue
            event.setdefault("line_no", line_no)
            event.update(meta)
            if event.get("segment/target_epoch") is None and meta.get("segment_target_epoch") is not None:
                event["segment/target_epoch"] = meta["segment_target_epoch"]
            if meta.get("segment_start_epoch") is not None:
                event["segment/start_epoch"] = meta["segment_start_epoch"]
            if event.get("event_type") == "metadata":
                context.update(event)
            events.append(event)
    return events


def metric_keys(events: list[dict[str, Any]]) -> list[str]:
    skip = {"event_type", "fatal", "message", "method", "segment_start_epoch", "segment_target_epoch", "log_path", "line_no"}
    keys = set()
    for event in events:
        for key, value in event.items():
            if key not in skip and isinstance(value, (int, float)):
                keys.add(key)
    return sorted(keys)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Parse AugSeg local logs into W&B-compatible metric events.")
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--head", type=int, default=20)
    args = parser.parse_args(argv)
    events = parse_log_file(args.log)
    print(json.dumps({"log": str(args.log), "events": len(events), "metrics": metric_keys(events)}, indent=2))
    for event in events[: max(0, args.head)]:
        print(json.dumps(event, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

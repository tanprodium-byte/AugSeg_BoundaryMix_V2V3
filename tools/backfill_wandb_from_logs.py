#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any

from wandb_log_parser import metric_keys, parse_log_file


def sanitize_run_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"uploaded_logs": {}}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return {"uploaded_logs": {}}
    data.setdefault("uploaded_logs", {})
    return data


def save_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")


def discover_logs(roots: list[Path], include_glob: str, methods: set[str] | None) -> list[Path]:
    logs: list[Path] = []
    for root in roots:
        if root.is_file() and root.match(include_glob):
            logs.append(root)
            continue
        if root.is_dir():
            logs.extend(sorted(root.rglob(include_glob)))
    if methods:
        filtered = []
        for path in logs:
            parsed = parse_log_file(path)
            method = parsed[0]["method"] if parsed else path.stem
            if method in methods:
                filtered.append(path)
        logs = filtered
    return sorted(dict.fromkeys(logs))


def _path_key(path: Path) -> str:
    try:
        return str(path.resolve())
    except OSError:
        return str(path.absolute())


def running_log_paths_from_postgres(db_url: str, suite_name: str, mode: str) -> set[str]:
    try:
        import psycopg
    except Exception as exc:
        raise RuntimeError("Postgres active-log filtering requires psycopg.") from exc

    paths: set[str] = set()
    with psycopg.connect(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT log_path
                FROM experiment_suite_queue
                WHERE suite_name=%s
                  AND mode=%s
                  AND status='running'
                  AND log_path IS NOT NULL
                  AND log_path <> ''
                """,
                (suite_name, mode),
            )
            for (log_path,) in cur.fetchall():
                paths.add(_path_key(Path(str(log_path))))
    return paths


def active_log_paths(args: argparse.Namespace) -> set[str]:
    paths: set[str] = set()
    if args.db_url_env:
        db_url = os.environ.get(args.db_url_env)
        if db_url:
            paths.update(running_log_paths_from_postgres(db_url, args.suite_name, args.mode))
    if not args.force_active and args.active_mtime_sec > 0:
        now = time.time()
        for path in discover_logs(args.log_root, args.include_glob, set(args.method) if args.method else None):
            try:
                age_sec = now - path.stat().st_mtime
            except OSError:
                continue
            if age_sec < args.active_mtime_sec:
                paths.add(_path_key(path))
    return paths


def upload_events(args: argparse.Namespace, grouped: dict[str, list[tuple[Path, list[dict[str, Any]]]]], manifest: dict[str, Any]) -> None:
    try:
        import wandb
    except Exception:
        print("Need WANDB_API_KEY or wandb login. Do not print the key.", file=sys.stderr)
        raise SystemExit(2)
    api_key = os.environ.get("WANDB_API_KEY") or getattr(getattr(wandb, "api", None), "api_key", None)
    if not api_key:
        print("Need WANDB_API_KEY or wandb login. Do not print the key.", file=sys.stderr)
        raise SystemExit(2)

    for method, log_events in sorted(grouped.items()):
        run_id = sanitize_run_id(f"{args.suite_name}__{method}")
        config = {
            "method": method,
            "suite_name": args.suite_name,
            "source": "local_log_backfill",
            "log_paths": [str(path) for path, _ in log_events],
        }
        run = wandb.init(
            project=args.project,
            entity=args.entity,
            group=args.group,
            name=method,
            id=run_id,
            resume="allow",
            config=config,
            tags=["backfilled_from_local_logs", "segment_scheduler"],
        )
        all_events = []
        for _, events in log_events:
            all_events.extend(event for event in events if not event.get("fatal") and event.get("event_type") != "metadata")
        if any("epoch" in event for event in all_events):
            wandb.define_metric("*", step_metric="epoch")
        for event in sorted(all_events, key=lambda e: (e.get("epoch", 10**9), e.get("step", 10**12), e.get("line_no", 0))):
            payload = {
                key: value
                for key, value in event.items()
                if key not in {"event_type", "method", "segment_start_epoch", "segment_target_epoch", "log_path", "line_no", "message", "fatal"}
                and isinstance(value, (int, float))
            }
            if payload:
                wandb.log(payload)
        run.finish()
        for path, events in log_events:
            manifest["uploaded_logs"][str(path)] = {
                "suite_name": args.suite_name,
                "method": method,
                "run_id": run_id,
                "event_count": len(events),
            }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Backfill W&B runs from local AugSeg log files.")
    parser.add_argument("--project", default="augseg-voc662")
    parser.add_argument("--entity")
    parser.add_argument("--group", required=True)
    parser.add_argument("--suite-name", required=True)
    parser.add_argument("--log-root", action="append", type=Path, required=True)
    parser.add_argument("--method", action="append")
    parser.add_argument("--include-glob", default="*_e*_to_e*.log")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--manifest", type=Path, default=Path("runs/wandb_backfill_manifest.json"))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-active", action="store_true", help="Skip running DB logs and recently modified logs.")
    parser.add_argument("--force-active", action="store_true", help="Disable recent-mtime active log protection.")
    parser.add_argument("--active-mtime-sec", type=int, default=120)
    parser.add_argument("--db-url-env", default="")
    parser.add_argument("--mode", default="full")
    args = parser.parse_args(argv)

    manifest = load_manifest(args.manifest)
    methods = set(args.method) if args.method else None
    logs = discover_logs(args.log_root, args.include_glob, methods)
    active_paths = active_log_paths(args) if args.skip_active else set()
    grouped: dict[str, list[tuple[Path, list[dict[str, Any]]]]] = {}
    zero_event_logs: list[str] = []
    skipped_manifest: list[str] = []
    skipped_active: list[str] = []
    fatal_logs: list[str] = []
    total_events = 0
    all_metric_keys: set[str] = set()

    for path in logs:
        if _path_key(path) in active_paths:
            skipped_active.append(str(path))
            continue
        if not args.force and str(path) in manifest.get("uploaded_logs", {}):
            skipped_manifest.append(str(path))
            continue
        events = parse_log_file(path)
        fatal_events = [event for event in events if event.get("fatal")]
        if fatal_events:
            fatal_logs.append(str(path))
            continue
        parseable_events = [event for event in events if event.get("event_type") != "metadata"]
        if not parseable_events:
            zero_event_logs.append(str(path))
        method = events[0]["method"] if events else path.stem
        grouped.setdefault(method, []).append((path, parseable_events))
        total_events += len(parseable_events)
        all_metric_keys.update(metric_keys(parseable_events))

    print(f"mode={'upload' if args.upload else 'dry-run'}")
    print(f"project={args.project} group={args.group} suite_name={args.suite_name}")
    print(
        f"logs_found={len(logs)} logs_ready={sum(len(v) for v in grouped.values())} "
        f"active_logs_skipped={len(skipped_active)} zero_event_logs_count={len(zero_event_logs)} "
        f"fatal_logs_skipped_count={len(fatal_logs)} manifest_skipped_count={len(skipped_manifest)} "
        f"parseable_events={total_events}"
    )
    print("runs:")
    for method, log_events in sorted(grouped.items()):
        count = sum(len(events) for _, events in log_events)
        print(f"  {sanitize_run_id(f'{args.suite_name}__{method}')} name={method} logs={len(log_events)} events={count}")
        for path, events in log_events:
            keys = ", ".join(metric_keys(events))
            print(f"    {path} events={len(events)} metrics=[{keys}]")
    print("metric_keys=" + ", ".join(sorted(all_metric_keys)))
    print("zero_event_logs=" + json.dumps(zero_event_logs, indent=2))
    print("fatal_logs_skipped=" + json.dumps(fatal_logs, indent=2))
    print("active_logs_skipped=" + json.dumps(skipped_active, indent=2))
    print("manifest_skipped=" + json.dumps(skipped_manifest, indent=2))

    if not args.upload:
        return 0
    upload_events(args, grouped, manifest)
    save_manifest(args.manifest, manifest)
    print(f"manifest_updated={args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

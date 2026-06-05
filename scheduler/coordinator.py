#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scheduler.config import DB_PATH
from scheduler.db import connect, init_db, mark_done, mark_failed, request_job


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    req = sub.add_parser("request-job")
    req.add_argument("--worker-id", required=True)
    req.add_argument("--server-name", required=True)
    req.add_argument("--gpu-id", type=int, required=True)
    req.add_argument("--db", default=str(DB_PATH))

    done = sub.add_parser("done")
    done.add_argument("--job-id", required=True)
    done.add_argument("--worker-id", required=True)
    done.add_argument("--latest-hf-path", default=None)
    done.add_argument("--db", default=str(DB_PATH))

    failed = sub.add_parser("failed")
    failed.add_argument("--job-id", required=True)
    failed.add_argument("--worker-id", required=True)
    failed.add_argument("--error", required=True)
    failed.add_argument("--non-retryable", action="store_true")
    failed.add_argument("--db", default=str(DB_PATH))

    args = parser.parse_args()
    conn = connect(args.db)
    init_db(conn)

    if args.cmd == "request-job":
        job = request_job(conn, args.worker_id, args.server_name, args.gpu_id)
        print(json.dumps(job or {}, indent=2))
        return 0
    if args.cmd == "done":
        ok = mark_done(conn, args.job_id, args.worker_id, args.latest_hf_path)
        print("DONE_ACCEPTED" if ok else "DONE_REJECTED")
        return 0 if ok else 1
    if args.cmd == "failed":
        ok = mark_failed(conn, args.job_id, args.worker_id, args.error, retryable=not args.non_retryable)
        print("FAILED_ACCEPTED" if ok else "FAILED_REJECTED")
        return 0 if ok else 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

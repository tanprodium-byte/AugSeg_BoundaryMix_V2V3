#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import os
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
SETTINGS = ROOT / "scheduler/hf_settings.yaml"
TEST_REL_PATH = "voc5_single_gpu_gbs8/_hf_test/supermaster_test.txt"
TEST_CONTENT = "supermaster hf access test\n"


def main() -> int:
    try:
        from huggingface_hub import HfApi, hf_hub_download
        from huggingface_hub.utils import HfHubHTTPError
    except Exception as exc:
        print(f"HF_ACCESS: FAIL import huggingface_hub: {exc}")
        print("Install with: conda run -n augseg-bm python -m pip install huggingface_hub")
        return 1

    cfg = yaml.safe_load(SETTINGS.read_text())
    repo_id = cfg["repo_id"]
    repo_type = cfg.get("repo_type", "model")
    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)

    try:
        who = api.whoami()
        print(f"login: {who.get('name') or who.get('email') or 'ok'}")
    except Exception as exc:
        print(f"HF_ACCESS: FAIL token/login unavailable: {exc}")
        print("Set HF_TOKEN or run huggingface-cli login in this environment.")
        return 1

    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"repo: access ok {repo_id} ({repo_type})")
    except HfHubHTTPError as exc:
        print(f"HF_ACCESS: FAIL repo inaccessible: {repo_id}: {exc}")
        print("Create the repo or grant access; this script will not create it automatically.")
        return 1
    except Exception as exc:
        print(f"HF_ACCESS: FAIL repo check error: {exc}")
        return 1

    tmp = ROOT / ".codex_smoke/hf_access/supermaster_test.txt"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(TEST_CONTENT)

    try:
        api.upload_file(
            path_or_fileobj=str(tmp),
            path_in_repo=TEST_REL_PATH,
            repo_id=repo_id,
            repo_type=repo_type,
        )
        downloaded = hf_hub_download(
            repo_id=repo_id,
            repo_type=repo_type,
            filename=TEST_REL_PATH,
            token=token,
            force_download=True,
        )
        got = Path(downloaded).read_text()
    except Exception as exc:
        print(f"HF_ACCESS: FAIL upload/download test: {exc}")
        return 1

    if got != TEST_CONTENT:
        print("HF_ACCESS: FAIL downloaded content mismatch")
        return 1

    print(f"uploaded_and_verified: {TEST_REL_PATH}")
    print("HF_ACCESS: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
